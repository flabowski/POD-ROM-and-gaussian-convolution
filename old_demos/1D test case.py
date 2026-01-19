# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:06:11 2023

@author: florianma
"""
from scipy.interpolate import (RegularGridInterpolator, RectBivariateSpline,
                               interpn, griddata, Rbf, interp1d, interp2d)
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, orth, qr
from scipy.optimize import curve_fit
from numpy import sin, cos, pi
cmap = plt.cm.plasma


def plot_paramspace(X):
    plt.imshow(X, interpolation="none")
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("$x$")
    ax.set_aspect(1/10)
    return fig, ax


def bump(x, mu):
    w = 15
    y = np.zeros_like(x)
    i = np.argmin(np.abs(x-mu))
    y[(i-w):(i+w-1)] = 1.0
    return y


def predict(VT, grid, r, xi):
    method = "linear"
    V_interpolated = np.zeros((r, 1))
    for mode in range(r):
        interpolator = interp1d(grid[0], VT[mode, :], kind=method)
        V_interpolated[mode] = interpolator(xi)
    return V_interpolated


m, n = 400, 10
xi = 0.35
sigma = 15
x = np.linspace(0, 1, m, endpoint=False)
mu = np.linspace(0, 1, n, endpoint=False)
grid = [mu, ]

X = np.zeros((m, n))
X_c = np.zeros_like(X)
for j, mu_j in enumerate(mu):
    X[:, j] = bump(x, mu_j)
U, S, VT = np.linalg.svd(X, False)
VT_intp = predict(VT, grid, min(m, n), xi)
X_intp = (U*S) @ VT_intp

# ---------------------------
sigmas = np.linspace(1e-6, 50, 1000)
sigmas[-1] = 18.768
e1 = np.zeros_like(sigmas)
e2 = np.zeros_like(sigmas)
e3 = np.zeros_like(sigmas)
for i, sigma in enumerate(sigmas):
    for j, mu_j in enumerate(mu):
        X_c[:, j] = gaussian_filter1d(X[:, j], sigma)
    U_c, S_c, VT_c = np.linalg.svd(X_c, False)

    VT_intp_c = predict(VT_c, grid, min(m, n), xi)
    X_intp_c = (U_c*S_c) @ VT_intp_c

    ind = np.argpartition(X_intp_c[:, 0], -29)[-29:]
    X_intp_c_post = np.zeros_like(X_intp_c)
    X_intp_c_post[ind, 0] = 1
    print(np.sum(bump(x, 0.35) > .5), np.sum(X_intp_c_post[:, 0] > .5))

    e1[i] = np.sum((bump(x, 0.35)-X_intp[:, 0])**2)**.5
    e2[i] = np.sum((bump(x, 0.35)-X_intp_c[:, 0])**2)**.5
    e3[i] = np.sum((bump(x, 0.35)-X_intp_c_post[:, 0])**2)**.5
# ---------------------------

fig, ax = plt.subplots()
ax.plot(sigmas, e1, "r.", label="standard ROM")
ax.plot(sigmas, e2, "g.", label="convolution + ROM")
ax.plot(sigmas, e3, "b.", label="convolution + ROM + post-processing")
plt.xlabel("$\sigma$")
plt.ylabel("$L_2- error$")
plt.legend()
plt.show()

fig, ax = plt.subplots()
plot_paramspace(X)
plt.show()

fig, ax = plt.subplots()
plot_paramspace(X_c)
plt.show()

fig, ax = plt.subplots()
plt.plot(x, bump(x, xi), "g.-", label="true solution")
plt.plot(x, X_intp[:, 0], "b.-", label="ROM")
plt.plot(x, X[:, 3], "r--", lw=1, label="closest snapshot (left)")
plt.plot(x, X[:, 4], "r--", lw=1, label="closest snapshot (right)")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig, ax = plt.subplots()
plt.plot(x, bump(x, xi), "g.-", label="true solution")
plt.plot(x, X_intp_c[:, 0], "b.-", label="convolution + ROM")
plt.plot(x, X_c[:, 3], "r--", lw=1, label="closest smoothened snapshot (left)")
plt.plot(x, X_c[:, 4], "r--", lw=1,
         label="closest smoothened snapshot (right)")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
