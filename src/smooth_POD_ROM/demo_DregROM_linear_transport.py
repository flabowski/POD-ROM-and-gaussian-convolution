from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from smooth_POD_ROM.post_processing import post_process, richardson_lucy
from smooth_POD_ROM.post_processing import get_sigma
from smooth_POD_ROM.pre_processing import convolve_f, gaussian_f, gaussian, smoothen
from smooth_POD_ROM.reduced_order_model import delta_n_width, train_ROM, L2_error, zielfunktion
from smooth_POD_ROM.plots_paper import Fig3, Fig4, Fig5, Fig6, Fig7
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
from scipy.optimize import curve_fit
from numpy import sin, cos, pi
cmap = plt.cm.plasma

# removed the if 1 in shape: ravel() in ezyrbs predict!


def rect_pulse_sin(x, mu, w=0.075+1e-6):
    w = 1/14
    y = np.zeros_like(x, dtype=np.float64)
    y[(0-1e-6 < (x-mu)) & ((x-mu) < w)] = 0.9
    ys = (np.sin((x-mu)*2*np.pi / w/2) + 1)/2 * 0.1
    return y+ys


def rect_pulse(x, mu, w=0.075+1e-6):
    y = np.zeros_like(x, dtype=np.float64)
    y[(0-1e-6 < (x-mu)) & ((x-mu) < w)] = 1.0
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
    X_train_s = smoothen(X_train, sigma/dx, x.shape, truncate=12)
    X_test_s = smoothen(X_test, sigma/dx, x.shape, truncate=12)
    X_val_s = smoothen(X_val, sigma/dx, x.shape, truncate=12)
    return X_train, X_test, X_val, X_train_s, X_test_s, X_val_s


def brute_force_hyperparams(x, mu_train, mu_test, X_train, X_test,
                            shape, rank, dx, clip, num_iter1, num_iter2):
    standard_rom = train_ROM(mu_train, X_train, rank=rank)
    X_test_ROM = standard_rom.predict(mu_test).T
    n_test = X_test.shape[1]
    sgms = np.linspace(0, 0.1, 21)
    c_all = np.linspace(0, 10, 21)
    eROM = L2_error(X_test_ROM, X_test)
    esROM = np.zeros((n_test, 21))
    esROMs = np.zeros((n_test, 21, 21))
    esROMs2 = np.zeros((n_test, 21, 21))
    for i, sigma_s in enumerate(sgms):
        print(sigma_s)
        X_train_s = smoothen(X_train, sigma_s/dx, shape)
        smooth_rom = train_ROM(mu_train, X_train_s, rank=rank)
        X_test_sROM = smooth_rom.predict(mu_test).T
        esROM[:, i] = L2_error(X_test_sROM, X_test)
        for j, c in enumerate(c_all):
            # de smoothing
            X_test_sROMs = post_process(x, X_test_sROM, sigma_s, c, mu_test,
                                        mu_train, num_iter1, shape=shape,
                                        clip=clip)
            esROMs[:, i, j] = L2_error(X_test_sROMs, X_test)
            # de smoothing with a higher number of iterations
            X_test_sROMs2 = post_process(x, X_test_sROM, sigma_s, c, mu_test,
                                         mu_train, num_iter2, shape=shape,
                                         clip=clip)
            esROMs2[:, i, j] = L2_error(X_test_sROMs2, X_test)
    return eROM, esROM, esROMs, esROMs2, c_all, sgms, c_all


def optimize_hyperparameters(x, mu_train, mu_test, X_train, X_test, shape, num_iter2, clip, rank):

    sigma_opt = 1/(rank*2)
    x0 = np.array([sigma_opt, 2.0])
    counter = np.array([0])

    def zf(params): return zielfunktion(params, x, mu_train, X_train, mu_test,
                                        X_test, rank, shape, num_iter2, clip=clip,
                                        counter=counter)[0]
    print("sigma, c, iteration count, mean_ROM, mean_sROM, mean_sROMs, improvement")
    res = minimize(zf, x0, method='SLSQP',
                   bounds=[(0.001, 10*sigma_opt), (-1, 10*2)],
                   options={'disp': True, "eps": np.array([.0005, 0.05]),
                            "maxiter": 25, "ftol": 0.0005})
    _i, _R, _sR, _sRs = zielfunktion(
        res["x"], x, mu_train, X_train, mu_test, X_test, rank, shape, num_iter2, clip=clip)
    # n_iter = 1000, sig_opt = 0.06579520, c_opt = 1.13641229, improvement = -46.5767 %
    # n_iter = 500, sig_opt = 0.06784999, c_opt = 1.05239696, improvement = -44.7313 % or -43.3101?
    return res["x"]


def optimize_hyperparameters_single(x, mu_train, mu_test, X_train, X_test, shape, num_iter2, clip, rank):

    sigma_opt = 1/(rank*2)
    x0 = np.array([sigma_opt, 2.0])
    counter = np.array([0])

    def zf(params): return zielfunktion(params, x, mu_train, X_train, mu_test[5, None],
                                        X_test[:, 5, None], rank, shape, num_iter2, clip=clip,
                                        counter=counter)[0]
    print("sigma, c, iteration count, mean_ROM, mean_sROM, mean_sROMs, improvement")
    res = minimize(zf, x0, method='SLSQP',
                   bounds=[(0.001, 10*sigma_opt), (-1, 10*2)],
                   options={'disp': True, "eps": np.array([.0005, 0.05]),
                            "maxiter": 25, "ftol": 0.0005})
    _i, _R, _sR, _sRs = zielfunktion(
        res["x"], x, mu_train, X_train, mu_test, X_test, rank, shape, num_iter2, clip=clip)
    # n_iter = 1000, sig_opt = 0.06579520, c_opt = 1.13641229, improvement = -46.5767 %
    # n_iter = 500, sig_opt = 0.06784999, c_opt = 1.05239696, improvement = -44.7313 % or -43.3101?
    return res["x"]


# def zf(params, x, mu_train, X_train, mu_test, X_test, rank, shape, num_iter,
#        clip=True, counter=np.array([0]), maxiter=150):
#     if counter[0] > maxiter:
#         counter[0] += 1
#         return 0., 0., 0., 0.
#     dx = x[1] - x[0]
#     sigma_S, sigma_D = params[0], params[1]
#     print("{:.8f}, {:.8f},".format(sigma_S, sigma_D), end=" ")

#     standard_rom = train_ROM(mu_train, X_train, rank=rank)
#     X_test_ROM_i = standard_rom.predict(mu_test[5])[:, None]
#     mean_ROM = L2_error(X_test_ROM_i, X_test[:, 5][:, None])

#     X_train_s = smoothen(X_train, sigma_S/dx, shape)
#     smooth_rom = train_ROM(mu_train, X_train_s, rank=rank)
#     X_test_sROM_i = smooth_rom.predict(mu_test[5])[:, None]
#     mean_sROM = L2_error(X_test_sROM, X_test[:, 5][:, None])

#     res, res_i = richardson_lucy(x, X_test_sROM_i.reshape(shape), sigma_D,
#                                  num_iter=num_iter, truth=None, damping=2,
#                                  clip=clip, mode="wrap",
#                                  monitor_convergence=True)
#     mean_sROMs = L2_error(res[:, None], X_test[:, 5][:, None])
#     improvement = 100*mean_sROMs/mean_ROM-100
#     print("{:.0f}, {:.8f}, {:.8f}, {:.8f}, {:.4f} %".format(
#         counter[0], mean_ROM, mean_sROM, mean_sROMs, improvement))
#     counter[0] += 1
#     return improvement, X_test_ROM, X_test_sROM, X_test_sROMs

if __name__ == "__main__":
    # TODO: save figures in case folder
    g = rect_pulse_sin
    n_x = 1000
    n_train = 20
    n_test = 10
    num_iter1 = 10
    num_iter2 = 10

    sigma, c = 0.5, 2  # 10 snapshots
    # sigma, c = 0.06544676, 1.13190924  # 10 snapshots, 1000 iterations
    sigma, c = 0.03859704, 0.72075422  # 20 snapshots, 1000 iterations, -51.1889 %
    sigma, c = 0.03344710, 0.97572576  # 20 snapshots, 500 iterations, -51.9384 %
    sigma, c = 0.02573402, 1.79707181  # 20 snapshots, 200 iterations, -53.5236 %
    sigma, c = 0.02175071, 2.76512484  # 20 snapshots, 100 iterations, -53.8829 %
    sigma, c = 0.01713914, 4.09847149  # 20 snapshots, 50 iterations, -51.6148 %
    sigma, c = 0.01195216, 6.51344694  # 20 snapshots, 20 iterations, -44.7214 %
    sigma, c = 0.01100714, 6.78930197  # 20 snapshots, 10 iterations, -37.6233 %

    rank = n_train
    dx = 1/n_x
    shape = (n_x,)
    clip = True

    x = np.linspace(0, 1, n_x, endpoint=False)
    mu_train = np.linspace([0.0], [1.0], n_train, endpoint=False)
    mu_test = np.linspace(mu_train[1], mu_train[2], n_test, endpoint=False)
    mu_val = x.copy()
    interpolatable = (mu_train[0] < mu_val) & (mu_val < mu_train[-1])
    mu_val = mu_val[interpolatable][:, None]

    data = get_datasets(x, g, mu_train, mu_test, mu_val, sigma, dx)
    X_train, X_test, X_val, X_train_s, X_test_s, X_val_s = data
    Fig3(x, mu_train, X_train, X_train_s)

    standard_rom = train_ROM(mu_train, X_train, rank=rank)
    smooth_rom = train_ROM(mu_train, X_train_s, rank=rank)
    X_test_ROM = standard_rom.predict(mu_test).T
    X_val_ROM = standard_rom.predict(mu_val).T
    X_test_sROM = smooth_rom.predict(mu_test).T
    X_val_sROM = smooth_rom.predict(mu_val).T
    Fig4(x, 5, mu_train, mu_test, X_train,
         X_train_s, X_test, X_test_ROM, X_test_sROM)

    X_test_sROMs = post_process(x, X_test_sROM, sigma, c, mu_test, mu_train,
                                num_iter1, shape, clip=clip, progress=False)
    X_test_sROMs2 = post_process(x, X_test_sROM, sigma, c, mu_test, mu_train,
                                 num_iter2, shape, clip=clip, progress=False)
    Fig5(x, X_test, X_test_s, X_test_ROM, X_test_sROM, X_test_sROMs2)

    # parameter optimization for the given rank
    sig_opt, c_opt = optimize_hyperparameters(x, mu_train, mu_test, X_train,
                                              X_test, shape, num_iter2, clip,
                                              rank)

    X_val_sROMs = post_process(x, X_val_sROM, sig_opt, c_opt, mu_val, mu_train,
                               num_iter1, shape, clip=clip, progress=False)
    X_val_sROMs2 = post_process(x, X_val_sROM, sig_opt, c_opt, mu_val, mu_train,
                                num_iter2, shape, clip=clip, progress=False)
    eROM = L2_error(X_val_ROM, X_val)
    esROM = L2_error(X_val_sROM, X_val)
    esROMs = L2_error(X_val_sROMs, X_val)
    esROMs2 = L2_error(X_val_sROMs2, X_val)
    Fig6(mu_val, eROM, esROM, esROMs, esROMs2)

    res = brute_force_hyperparams(x, mu_train, mu_test, X_train, X_test, shape,
                                  rank, dx, clip, num_iter1, num_iter2)
    eROM_mat, esROM_mat, esROMs_mat, esROMs2_mat, c_all, sgms, c_all = res
    Fig7(eROM_mat, esROM_mat, esROMs_mat, c_all, sgms)
    Fig7(eROM_mat, esROM_mat, esROMs2_mat, c_all, sgms)

    # TODO: parameter optimization per rank
    NN = np.arange(3, 150, 2)

    # TODO: Fig. 8 (error vs rank)
    # TODO: Fig. 9 (sigma vs rank)

    # TODO: convergence Richardson-Lucy
