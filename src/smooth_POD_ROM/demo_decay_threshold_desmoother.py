# -*- coding: utf-8 -*-
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
# from smooth_POD_ROM.post_processing import post_process, richardson_lucy
# from smooth_POD_ROM.post_processing import get_sigma
from smooth_POD_ROM.pre_processing import convolve_f, gaussian_f, gaussian, smoothen
from smooth_POD_ROM.reduced_order_model import delta_n_width, train_ROM, L2_error
from smooth_POD_ROM.plots_paper import Fig3, Fig4, Fig5, Fig6, Fig7, Fig8, Fig11
import warnings
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d as conv2
from scipy.interpolate._rgi_cython import evaluate_linear_2d, find_indices
from scipy.interpolate import (RegularGridInterpolator, RectBivariateSpline,
                               interpn, griddata, Rbf, interp1d, interp2d)
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.signal import wiener, butter, freqs, convolve, freqz
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, orth, qr
from scipy.optimize import curve_fit, direct, Bounds
from numpy import sin, cos, pi
cmap = plt.cm.plasma

# removed the if 1 in shape: ravel() in ezyrbs predict!


def rectangular_pulse(x, mu, w=0.075+1e-6):
    n = int(len(x)*0.075//1)
    y_dummy = 1/((x-mu)**2+.1)+1/((x-mu-1)**2+.1)+1/((x-mu+1)**2+.1)
    largest = np.argsort(y_dummy)[-n:]
    y = np.zeros_like(x, dtype=np.float64)
    y[largest] = 1
    # y[(0 <= (x-mu)) & ((x-mu) <= w)] = 1.0
    # y[(0 <= (x-mu-1)) & ((x-mu-1) <= w)] = 1.0
    # y[(0 <= (x-mu+1)) & ((x-mu+1) <= w)] = 1.0
    return y


def get_datasets(x, g, mu_train, mu_test, mu_val, sigma, dx):
    n_x = len(x)
    X_train = np.empty((n_x, len(mu_train)))
    X_test = np.zeros((n_x, len(mu_test)))
    X_val = np.zeros((n_x, len(mu_val)))

    for _mu, _X in zip([mu_train, mu_test, mu_val], [X_train, X_test, X_val]):
        for j, mu_j in enumerate(_mu):
            y = g(x, mu_j)
            _X[:, j] = y
    return X_train, X_test, X_val


def post_process(x, data):
    n = int(len(x)*0.075//1)  # a priori knowledge

    deconvolved = np.zeros_like(data)
    for j in range(data.shape[1]):
        ss = data[:, j].copy()
        largest = np.argsort(ss)[-n:]
        # ss *= 0
        # ss[largest] = 1
        deconvolved[largest, j] = 1
    return deconvolved


def zielfunktion(params, x, mu_train, X_train, mu_test, X_test, rank, shape):
    # if counter[0] > maxiter:
    #     counter[0] += 1
    #     return 0., 0., 0., 0.
    dx = x[1] - x[0]
    sigma = params[0]

    standard_rom = train_ROM(mu_train, X_train, rank=rank)
    X_test_ROM = standard_rom.predict(mu_test).snapshots_matrix.T
    e_ROM = L2_error(X_test_ROM, X_test)
    mean_ROM = np.mean(e_ROM)

    X_train_s = smoothen(X_train, sigma/dx, shape)
    smooth_rom = train_ROM(mu_train, X_train_s, rank=rank)
    X_test_sROM = smooth_rom.predict(mu_test).snapshots_matrix.T
    e_sROM = L2_error(X_test_sROM, X_test)
    # mean_sROM = np.mean(e_sROM)
    X_test_sROMs = post_process(x, X_test_sROM)
    e_sROMs = L2_error(X_test_sROMs, X_test)
    mean_sROMs = np.mean(e_sROMs)
    improvement = 100*mean_sROMs/mean_ROM-100
    # print("{:.8f}, {:.8f},".format(sigma, c),
    #       "{:.0f}, {:.8f}, {:.8f}, {:.8f}, {:.4f} %".format(
    #           counter[0], mean_ROM, mean_sROM, mean_sROMs, improvement))
    # counter[0] += 1
    print(len(mu_train), sigma, improvement)

    sigs[0].append(sigma)
    imprmts[0].append(improvement)
    line3.set_xdata(sigs[0])
    line3.set_ydata(imprmts[0])
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    return improvement, X_test_ROM, X_test_sROM, X_test_sROMs


fig2, ax2 = plt.subplots()
sigs = [[]]
imprmts = [[]]
line3, = ax2.plot(sigs[0], imprmts[0], "C0.")
ax2.set_ylim([-100, 0])
ax2.set_xlim([1e-7, 0.3])


def optimize_hyperparameters(x, mu_train, mu_test, X_train, X_test):
    # return [0.025]
    sigma_opt = 1/mu_train.size
    x0 = np.array([sigma_opt])
    counter = np.array([0])

    sigs[0] = []
    imprmts[0] = []
    # ax2.set_xlim([1e-7, 5*sigma_opt])

    def zf(params): return zielfunktion(params, x, mu_train, X_train, mu_test,
                                        X_test, rank, shape)[0]
    print(len(mu_train))
    bounds = [(1e-7, 0.3)]
    res = direct(zf, bounds, eps=1e-6, len_tol=min(0.005, sigma_opt/100),
                 maxiter=10)
    # res = minimize(zf, x0, method='SLSQP',
    #                bounds=[(0.001, 10*sigma_opt)],
    #                options={'disp': True, "eps": np.array([sigma_opt*0.5]),
    #                         "maxiter": 25, "ftol": 0.00005})
    # _i, _R, _sR, _sRs = zielfunktion(
    #     res["x"], x, mu_train, X_train, mu_test, X_test, rank, shape, num_iter2, clip=clip)
    # n_iter = 1000, sig_opt = 0.06579520, c_opt = 1.13641229, improvement = -46.5767 %
    # n_iter = 500, sig_opt = 0.06784999, c_opt = 1.05239696, improvement = -44.7313 % or -43.3101?
    return res.x

# for j in range(25):
#     fig, ax = plt.subplots()
#     ax.plot(x, X_test[:, j], "C0.-")
#     ax.plot(x, _sR[:, j], "C1.-")
#     ax.plot(x, _sRs[:, j], "C2.-")
#     ax.plot(x, X_test[:, j]-_sRs[:, j], "r.-")
#     plt.show()
# impr = np.empty(shape=(100,))
# for i, sigm in enumerate(np.linspace(0.02, 0.03, 100)):
#     params = [sigm]
#     impr[i] = zielfunktion(params, x, mu_train, X_train, mu_test,
#                                         X_test, rank, shape)[0]
#     print(i, sigm, impr[i])
# plt.plot(np.linspace(0.02, 0.03, 100), impr)


# def estimate_dN(g, x, mu_train, mu_test, X_train, X_test, shape,
#                 num_iter1, num_iter2, clip, rank):
if True:
    n_x = 2500
    n_test = 25
    shape = (n_x,)
    g = rectangular_pulse
    x = np.linspace(0, 1, n_x, endpoint=False)

    # data = get_datasets(x, g, mu_train, mu_test, mu_val, sigma, dx)

    n_x = len(x)
    # num_iter = 200
    NN = np.arange(5, 150, 5)
    dN_ROM = np.zeros(len(NN))
    dN_sROM = np.zeros(len(NN))
    dN_sROMs = np.zeros(len(NN))
    dN_sROMs2 = np.zeros(len(NN))
    dN_sig = np.zeros(len(NN))
    # dN_c = np.zeros(len(NN))
    mu_test = np.random.rand(n_test,)[:, None]

    page_width_in = 5.395665354330708
    fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
    line1, = plt.plot(NN, dN_ROM, "C0.-", ms=2, label="$u_{\mu,rb}$")
    #plt.plot(NN, esROM, "C1--", ms=2, label="$u_{rb,S}$")
    line2, = plt.plot(NN, dN_sROMs, "C2.-", ms=2, label="$u_{\mu,Srb,D}$")
    plt.plot(NN, 0.5*1/NN**.5, "C0--", label=".5/sqrt(N)")
    plt.plot(NN, 30/NN**2, "C2--", label="25/(N**2)")
    plt.plot(NN, .9/NN, "C1--", label=".9/(N)")
    plt.legend()
    plt.ylabel("$\|u_{\mu,rb}-u_{\mu}\|_{L_2}$")
    plt.xlabel("$N$")
    # plt.ylim(0, 0.4)
    plt.legend()
    ax.set_xticks(np.linspace(0, 250, 26, endpoint=True), minor=True)
    ax.set_yticks(np.linspace(0, .4, 19, endpoint=True), minor=True)
    plt.grid(True, which='minor', linestyle='--', lw=.25)
    plt.grid(True, which='major', linestyle='-')
    ax.set_yscale('log')
    plt.xlim(0, 150)
    plt.ylim(1e-3, .3)

    for i, rank in enumerate(NN):
        n_train = rank
        mu_train = np.linspace([0.0], [1.0], n_train+1, endpoint=True)
        # mu_test = np.linspace(mu_train[1], mu_train[2], n_test, endpoint=False)
        X_train = np.empty((n_x, len(mu_train)))
        X_test = np.zeros((n_x, len(mu_test)))
        for _mu, _X in zip([mu_train, mu_test], [X_train, X_test]):
            for j, mu_j in enumerate(_mu):
                _X[:, j] = g(x, mu_j)

        print(np.sum(X_train, axis=0))
        # optimize sigma
        # smooth using sigma_opt
        res = optimize_hyperparameters(x, mu_train, mu_test, X_train, X_test)
        _i, _R, _sR, _sRs = zielfunktion(res, x, mu_train, X_train, mu_test,
                                         X_test, rank, shape)

        dN_ROM[i] = np.mean(L2_error(_R, X_test))
        dN_sROM[i] = np.mean(L2_error(_sR, X_test))
        dN_sROMs[i] = np.mean(L2_error(_sRs, X_test))
        dN_sig[i] = res[0]

        line1.set_ydata(dN_ROM)
        line2.set_ydata(dN_sROMs)
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.show()
    fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
    plt.scatter(NN, dN_sig, c=dN_sROMs/dN_ROM, vmin=0, vmax=.5)
    plt.show()

    # plt.plot(NN, dN_sROMs2, "C3.-", ms=2, label="$u_{\mu,rb,D_{1000}}$")
    #plt.plot(mu_val, esROMs2, "C3<-", ms=4, markevery=(20, markevery), label="$u_{rb,D_{100}}$")

    plt.show()
    # return NN, dN_ROM, dN_sROM, dN_sROMs, dN_sROMs2, dN_sig, dN_c
