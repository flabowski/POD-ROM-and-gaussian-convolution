# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from smooth_POD_ROM.reduced_order_model import pulse, delta_n_width
from smooth_POD_ROM.pre_processing import gaussian, gaussian_f, convolve_f
from scipy.ndimage import gaussian_filter
cmap = plt.cm.plasma


page_width_pt = 455.24
pt2in = 0.01389
pt2cm = 0.0352777778
cm2in = 1/2.54
page_width_cm = 13.70499
# TODO: work with textwidth
plot_width_in = page_width_pt*pt2in/2
page_width_in = page_width_cm*cm2in
print(plot_width_in/cm2in)

fs = 10
fs_lbl = 6
plt.rcParams["figure.figsize"] = (plot_width_in, plot_width_in/1.61803398875)
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.size'] = fs
plt.rcParams['axes.titlesize'] = fs
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['legend.labelspacing'] = 0.0
plt.rcParams['legend.fontsize'] = fs_lbl
plt.rcParams['legend.handlelength'] = 1.0

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', size=fs, serif='Computer Modern Roman')
pth = "../Plots/"

markevery = 45

m, n = 2500, 2500
# w = 30  # width of the pulse in # nodes
x = np.linspace(0, 1, m, endpoint=False)+1/(2*m)
mu_train = np.linspace(0.0, 1.0, n, endpoint=False)+1/(2*n)
# mu_test = np.array([0.2, 0.22, 0.225, 0.23,  0.24, 0.25])
# mu_validation = np.linspace(0, 1, m, endpoint=False)
sigma = 0.05
mode = "constant"
rank = 10
dx = 1/m
pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/convolution_paper/"


X_train = np.empty((m, len(mu_train)))
X_train_s1 = np.empty((m, len(mu_train)))
X_train_s2 = np.empty((m, len(mu_train)))
X_train_s3 = np.empty((m, len(mu_train)))
X_train_s4 = np.empty((m, len(mu_train)))
X_train_s5 = np.empty((m, len(mu_train)))


def test1():
    sigma = 0.05
    mu = 0.25
    x = np.linspace(0, 1, 101, endpoint=False)
    dx = x[1]-x[0]
    y = pulse(x, mu)
    g_x = gaussian(x-0.5+dx, sigma, shift=False)
    g_f = gaussian_f(x, sigma)
    ys1 = convolve_f(y, g_f)
    # ! boundary effects cant be avoided!
    ys2 = np.convolve(y, g_x, mode="same")
    ys3 = gaussian_filter(y, sigma=sigma/dx, truncate=12, mode="wrap")
    print(np.allclose(ys1, ys2, atol=1e-6))
    print(np.allclose(ys1, ys3, atol=1e-6))
    print(np.allclose(ys2, ys3, atol=1e-6))
    plt.plot(x, y, "C0.")
    plt.plot(x, ys1, "C1.")
    plt.plot(x, ys2, "C2.")
    plt.plot(x, ys3, "C3.")
    return


n1 = int(0.05*m)
ma1 = np.ones(n1,)/n1


for j, mu_j in enumerate(mu_train):
    y = pulse(x, mu_j)
    X_train[:, j] = y
    X_train_s1[:, j] = gaussian_filter(y, sigma=0.005/dx, truncate=12)
    X_train_s2[:, j] = gaussian_filter(y, sigma=0.05/dx, truncate=12)
    X_train_s3[:, j] = np.convolve(y, ma1, mode="same")

U, S, VT = np.linalg.svd(X_train, False)
U, S1, VT = np.linalg.svd(X_train_s1, False)
U, S2, VT = np.linalg.svd(X_train_s2, False)
U, S3, VT = np.linalg.svd(X_train_s3, False)
delta_N = delta_n_width(S, m, n)
delta_N1 = delta_n_width(S1, m, n)
delta_N2 = delta_n_width(S2, m, n)
delta_N3 = delta_n_width(S3, m, n)

fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
ax.plot(x, X_train[:, n//2], "C0-", lw=1, label="pulse")
ax.plot(x, X_train_s3[:, n//2], "C3-", lw=1, label="moving average")
ax.plot(x, X_train_s1[:, n//2], "C1-", lw=1, label="smoothed, $\sigma=0.005$")
ax.plot(x, X_train_s2[:, n//2], "C2-", lw=1, label="smoothed, $\sigma=0.05$")
plt.grid(which="minor")
plt.grid(which="major")
ax.set_xlim([0.3, .7])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x; 0.5)$")
plt.legend()
plt.tight_layout()
plt.savefig(pth+"Pulse_smoothing.png")
plt.show()

fig, ax = plt.subplots()
plt.plot(delta_N / (1/np.pi*np.arange(len(S))**(-1/2)), ".")
plt.show()

fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
ax.plot(delta_N, "C0.", ms=1, label="pulse")
ax.plot(delta_N3, "C3.", ms=1, label="moving average")
ax.plot(delta_N1, "C1.", ms=1, label="smoothed, $\sigma=0.005$")
ax.plot(delta_N2, "C2.", ms=1, label="smoothed, $\sigma=0.05$")
# ax.plot(1/np.pi*np.arange(len(S))**(-1/2), "k--")
ax.set_yscale('log')
ax.set_xlim([0, len(S)])
ax.set_ylim([1e-16, 1])
plt.grid(which="minor")
plt.grid(which="major")
ax.set_xlabel("$N$")
ax.set_ylabel("$\delta_N$")
plt.legend()
plt.tight_layout()
plt.savefig(pth+"Pulse_smoothing_deltaN.png")
plt.show()
