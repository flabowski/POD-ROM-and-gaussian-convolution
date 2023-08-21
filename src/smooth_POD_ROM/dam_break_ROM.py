# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:10:15 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from scipy.interpolate import griddata
from smooth_POD_ROM.pre_processing import (
    on_regular_grid, _gauss_2d, to_frequency, to_space, add_padding)
from smooth_POD_ROM.io import get_data, get_field
from smooth_POD_ROM.plotting import plot_mesh, plot_field, plot_structured_field
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import wiener, convolve2d
from skimage import restoration
from skimage.restoration import richardson_lucy
import tvtk
import cv2
from datetime import datetime
# TODO: ROM in frequency domain?
# TODO: upsample in space
# TODO: smoothen in time too
ts = 25


def get_snapshots():
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/"
    mu_all = [10, 100, 1000, 2000, 5000]   # 10*2**np.arange(1, 10, 2)
    rho_all = [10, 50, 100, 400, 800, 1200]  # 10*2**np.arange(1, 8, 1)
    t_all = [ts]  # [30]  # np.arange(2, 102)
    X = []
    params = []
    for m in mu_all:
        print(m)
        for r in rho_all:
            for t in t_all:
                file = "damBreak_{:04d}_{:04d}_1_{:d}.vtk".format(m, r, t)
                data = get_field(pth+file, 'alpha.water')
                X.append(data)
                params.append([m, r, t])
    return np.array(X).T, np.array(params)


def gen_test_params():
    # mu = 10*2**np.arange(1, 10, 2)
    # rho = 10*2**np.arange(1, 8, 1)
    N = 20
    mu_test = 2**(np.random.rand(N)*8.96+3.32)
    rho_test = 2**(np.random.rand(N)*6.9+3.32)
    for i in range(N):
        print("{:.4f}, {:.4f}".format(mu_test[i], rho_test[i]))

    mu_all = [10, 100, 1000, 2000, 5000]   # 10*2**np.arange(1, 10, 2)
    rho_all = [10, 50, 100, 400, 800, 1200]  # 10*2**np.arange(1, 8, 1)
    M, R = np.meshgrid(mu_all, rho_all)
    fig, ax = plt.subplots()
    plt.plot(M.ravel(), R.ravel(), "b.")
    plt.plot(mu_test, rho_test, "rx")
    plt.show()
    fig, ax = plt.subplots()
    plt.plot(M.ravel(), R.ravel(), "b.")
    plt.plot(mu_test, rho_test, "rx")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


def deconvolution_gauss(data):
    k = _gauss_2d(sigma)
    s = (k.shape[0]-1) // 2
    k[s, s] = 0.0
    k[s, s] = np.sum(-k)
    return convolve2d(data, k, boundary='fill', mode='same')


def laplacian_deconvolution(data):
    laplacian = cv2.Laplacian(data, cv2.CV_64F)
    laplacian /= np.max(laplacian)
    # laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
    sharpened = data - laplacian
    return sharpened


def _find_largest_elements_2d(window, k=5):
    flattened = window.flatten()
    indices = np.argpartition(flattened, -k)[-k:]
    values = flattened[indices]
    sorted_indices = np.argsort(-values)
    sorted_values = values[sorted_indices]
    sorted_indices = indices[sorted_indices]
    ind2D = np.unravel_index(sorted_indices, shape=window.shape, order='C')
    return sorted_values, ind2D


def deconvolve_exact(data_on_grid_smooth_s, psf, psf_f):
    # (151, 159), (57, 57), (151, 159)
    data_on_grid_smooth_f = fft2(data_on_grid_smooth_s)
    data_on_grid_f = data_on_grid_smooth_f / psf_f
    data_on_grid_s = to_space(data_on_grid_f)
    return data_on_grid_s


def local_mass_conservation(data, s):
    i, j = 45, 60
    i, j = 20, 80
    m, n = data.shape
    # s = 4*sigma
    # data_pp = np.zeros_like(data)
    data_t = data.copy()
    for s in [int(s)]:
        data_pp = np.zeros_like(data_t)
        n_evals = np.zeros_like(data_t)
        for i in range(-s+1, m):
            for j in range(-s+1, n):
                # TODO: add padding
                s1 = np.clip(i, 0, m)
                e1 = np.clip(i+s, 0, m)
                s2 = np.clip(j, 0, n)
                e2 = np.clip(j+s, 0, n)
                window = data_t[s1:e1, s2:e2]
                # plt.imshow(window)
                k = np.int32(np.round(np.sum(window), decimals=0))
                assert k <= window.size
                if k > 0:
                    v, inds = _find_largest_elements_2d(window, k)
                    assert len(inds[0]) == k
                    data_pp[s1+inds[0], s2+inds[1]] += np.sum(window)/k
                n_evals[s1:e1, s2:e2] += 1
        data_t = data_pp.copy()/n_evals
    return data_t


def crop(data_full, kernel):
    # psf = Point Spread Function
    px, py = kernel.shape
    px = (px+1)//2
    py = (py+1)//2
    return data_full[px:-px, py:-py]


if __name__ == "__main__":
    # TODO fix error by figuring out all shapes flat and 2D. snapshots on cells, points, grid, smooth in space, frequency
    # points: (9281, 3)
    # triangles: (18144, 3)
    # point_data: (9281,)
    # data_on_grid: (93, 101)
    # data_on_grid_f: (151, 159)
    # data_on_grid_s: (151, 159)

    # psf (Point Spread Function): (57, 57)
    # psf_f (frequencies): (151, 159)

    sigma = 7

    file = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/damBreak_1000_0100_1_30.vtk"
    points, triangles, point_data, cell_data = get_data(file)
    polys = (points[:, :2])[triangles]

    X, Y, data_on_grid = on_regular_grid(
        points[:, :2],  point_data['alpha.water'])
    # data_on_grid_s = gaussian_filter(data_on_grid, sigma=sigma)
    psf = _gauss_2d(sigma, truncate=4, size=False)
    # ------------------------------------------------------------------------
    psf_p = add_padding(psf, data_on_grid.shape)
    psf_f = to_frequency(psf_p, shift=True)
    data_on_grid_p = add_padding(data_on_grid, psf.shape)
    data_on_grid_f = to_frequency(data_on_grid_p)
    data_on_grid_smooth_f = data_on_grid_f * psf_f
    data_on_grid_s = to_space(data_on_grid_smooth_f)
    # ------------------------------------------------------------------------
    # data_on_grid_s = convolve2d(data_on_grid, psf,
    #                             boundary='fill', mode='full')
    # ------------------------------------------------------------------------
    # plotting:
    fig, ax = plt.subplots()
    plot_mesh(points, polys)
    plt.show()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey=True)
    ax1.tripcolor(points[:, 0], points[:, 1], triangles,
                  point_data['alpha.water'])
    ax1.set_title("point_data")
    ax2.tripcolor(points[:, 0], points[:, 1], triangles,
                  cell_data['alpha.water'])
    ax2.set_title("cell_data")
    ax3.pcolormesh(X, Y, data_on_grid)
    ax3.set_title("grid_data")
    ax4.pcolormesh(X, Y, crop(data_on_grid_s, psf))
    ax4.set_title("grid_data (smooth)")
    plt.show()

    # project snapshots on grid and smoothen them
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
          "/Documents/data/VTK_Legacy_NEW/"
    if True:
        snapshots, p = get_snapshots()
        m, n = snapshots.shape
        X_grid = np.empty((data_on_grid.size, n))
        # FIXME
        X_grid_smooth = np.empty((data_on_grid_s.size, n))
        for j in range(n):
            print(j)
            _on_grid = griddata(points[:, :2], snapshots[:, j], (X, Y))
            # _on_grid_s = gaussian_filter(_on_grid, sigma=sigma)
            # _on_grid_s = convolve2d(_on_grid, psf,
            #                         boundary='fill', mode='full')
            # TODO: grid size is odd in every dimension
            _on_grid_p = add_padding(_on_grid, psf.shape)
            _on_grid_f = to_frequency(_on_grid_p)
            _on_grid_smooth_f = _on_grid_f * psf_f
            _on_grid_s = to_space(_on_grid_smooth_f)
            X_grid[:, j] = _on_grid.ravel()
            X_grid_smooth[:, j] = _on_grid_s.ravel()
        np.save(pth+"p.npy", p)
        np.save(pth+"data_on_grid.npy", X_grid)
        np.save(pth+"data_on_grid_s.npy", X_grid_smooth)
        np.save(pth+"snapshots.npy", snapshots)
    else:
        p = np.load(pth+"p.npy")
        X_grid = np.load(pth+"data_on_grid.npy")
        X_grid_smooth = np.load(pth+"data_on_grid_s.npy")
        snapshots = np.load(pth+"snapshots.npy")

    # ROM
    reg = RegularGrid()
    pod = POD()
    x_test = np.array([[500, 75, ts]])
    # x_test = np.array([[1000, 100, ts]])
    predictions = []
    for ss in (snapshots.T, X_grid.T, X_grid_smooth.T):
        print(p[:, :2].shape, ss.shape)
        db = Database(p[:, :2], ss)
        rom = ROM(db, pod, reg)
        rom.fit()
        predictions.append(rom.predict(x_test[:, :2]))

    smoothROMprediction = predictions[2].copy().reshape(data_on_grid_s.shape)
    # res_pp = deconvolution_gauss(data)
    # res_pp = laplacian_deconvolution(data)
    # res_pp = wiener(res_pp, mysize=(11, 11))
    # res_pp = local_mass_conservation(data)
    # data_decon = richardson_lucy(smoothROMprediction, psf, num_iter=30)
    data_decon = deconvolve_exact(smoothROMprediction, psf, psf_f)

    data_decon2 = local_mass_conservation(data_decon, 20)
    data_decon3 = local_mass_conservation(data_decon2, 10)
    data_decon4 = local_mass_conservation(data_decon3, 5)
    predictions[3:3] = [data_decon]
    predictions[4:4] = [data_decon4]

    # plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey=True)
    # ax1.tripcolor(points[:, 0], points[:, 1], triangles,
    #               predictions[0], vmin=0, vmax=1)
    # ax1.set_title("standard ROM with point_data")
    ax1.pcolormesh(X, Y, predictions[1].reshape(X.shape), vmin=0, vmax=1)
    ax1.set_title("standard ROM with grid_data")

    C = crop(predictions[2].reshape(data_on_grid_s.shape), psf)
    ax2.pcolormesh(X, Y, C, vmin=0, vmax=1)
    ax2.set_title("standard ROM with smooth grid_data")
    C = crop(predictions[3].reshape(data_on_grid_s.shape), psf)
    ax3.pcolormesh(X, Y, C, vmin=0, vmax=1)
    ax3.set_title("standard ROM with smooth grid_data + richardson lucy")
    C = crop(predictions[4].reshape(data_on_grid_s.shape), psf)
    ax4.pcolormesh(X, Y, C, vmin=0, vmax=1)
    ax4.set_title(
        "standard ROM with smooth grid_data + richardson lucy + local mass conservation")
    plt.show()

    asd

    # mu_s, mu_e = 10, 100  # [10, 100, 1000, 2000, 5000]
    # rho_s, rho_e = 10,  50  # [10, 50, 100, 400, 800, 1200]
    # t_s, t_e = 0,  10  # [0, ..., 99]
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/data/VTK_Legacy_NEW/"
    # snapshots, parameters, cases = Get_SnapsParam(
    #     pth, 10, 10000, 10,  5000, 30, 30)
    asd
    parameters = parameters[:, [0, 1, 3]]

    plot_unstructured_and_structured_field(cases[5], "alpha.water")

    db = Database(parameters[:, [0, 1]], snapshots, space=cases[0].cellCentres)
    pod = POD('svd')  # reduction
    rbf = RBF()  # approximation
    lin = Linear()  # approximation
    # rg = RegularGrid()  # approximation
    rom = ROM(db, pod, lin)
    rom.fit()

    new_mu = [500, 75]
    pred_sol = rom.predict(new_mu)
    new_case = deepcopy(cases[0])
