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
from smooth_POD_ROM.pre_processing import on_regular_grid
from smooth_POD_ROM.io import get_data, get_field
from smooth_POD_ROM.plotting import plot_mesh, plot_field, plot_structured_field
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener, convolve2d
import tvtk
import cv2


def get_snapshots():
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/"
    mu_all = [10, 100, 1000, 2000, 5000]   # 10*2**np.arange(1, 10, 2)
    rho_all = [10, 50, 100, 400, 800, 1200]  # 10*2**np.arange(1, 8, 1)
    t_all = [30]  # np.arange(100)
    X = []
    params = []
    for m in mu_all:
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


def gauss_2d(sigma):
    cutoff = 3*sigma
    s = np.int32(np.round((cutoff-1)/2, decimals=0))
    x = np.arange(-s, s+1, 1)
    y = np.arange(-s, s+1, 1)
    x, y = np.meshgrid(x, y, indexing="ij")
    mu_x = mu_y = 0
    sigma_x = sigma_y = sigma
    exponent = ((x - mu_x)**2 / (2 * sigma_x**2)) + \
        ((y - mu_y)**2 / (2 * sigma_y**2))
    coefficient = 1 / (2 * np.pi * sigma_x * sigma_y)
    res = -coefficient * np.exp(-exponent)
    res[s, s] = 0.0
    res[s, s] = np.sum(-res)
    # plt.imshow(res)
    return res


def find_largest_elements_2d(window, k=5):
    flattened = window.flatten()
    indices = np.argpartition(flattened, -k)[-k:]
    values = flattened[indices]
    sorted_indices = np.argsort(-values)
    sorted_values = values[sorted_indices]
    sorted_indices = indices[sorted_indices]
    ind2D = np.unravel_index(sorted_indices, shape=window.shape, order='C')
    return sorted_values, ind2D


def aascsa(data):
    i, j = 45, 60
    i, j = 20, 80
    m, n = data.shape
    s = 4*sigma
    # data_pp = np.zeros_like(data)
    data_t = data.copy()
    for s in [4*sigma, 3*sigma, 2*sigma, sigma]:
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
                    v, inds = find_largest_elements_2d(window, k)
                    assert len(inds[0]) == k
                    data_pp[s1+inds[0], s2+inds[1]] += np.sum(window)/k
                n_evals[s1:e1, s2:e2] += 1
        data_t = data_pp.copy()/n_evals
    return data_t
    # fig, (a1, a2) = plt.subplots(2, sharex=True, sharey=True)
    # a1.imshow(data.T, origin="lower", vmin=0, vmax=1)
    # a2.imshow(data_pp.T/s**2, origin="lower", vmin=0, vmax=1)
    # print(np.sum(data))
    # print(np.sum(data_t))


if __name__ == "__main__":
    sigma = 5
    snapshots, p = get_snapshots()
    file = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/damBreak_0010_0010_1_30.vtk"
    points, triangles, point_data, cell_data = get_data(file)
    polys = (points[:, :2])[triangles]

    X, Y, data_on_grid = on_regular_grid(
        points[:, :2],  point_data['alpha.water'])
    data_on_grid_s = gaussian_filter(data_on_grid, sigma=sigma)

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
    ax4.pcolormesh(X, Y, data_on_grid_s)
    ax4.set_title("grid_data (smooth)")
    plt.show()

    # project snapshots on grid and smoothen them
    m, n = snapshots.shape
    X_grid = np.empty((X.size, n))
    X_grid_smooth = np.empty((X.size, n))
    for j in range(n):
        print(j)
        data_on_grid = griddata(points[:, :2], snapshots[:, j], (X, Y))
        data_on_grid_s = gaussian_filter(data_on_grid, sigma=sigma)
        X_grid[:, j] = data_on_grid.ravel()
        X_grid_smooth[:, j] = data_on_grid_s.ravel()

    # ROM
    reg = RegularGrid()
    pod = POD()
    x_test = [[500, 75]]
    res = []
    for ss in (snapshots.T, X_grid.T, X_grid_smooth.T):
        print(p[:, :2].shape, ss.shape)
        db = Database(p[:, :2], ss)
        rom = ROM(db, pod, reg)
        rom.fit()
        res.append(rom.predict(x_test))

    data = res[2].copy().reshape(X.shape)
    res_pp = aascsa(data)
    # fig, ax = plt.subplots()
    # plt.imshow(data.T, origin="lower")
    # laplacian = cv2.Laplacian(res_pp, cv2.CV_64F)
    # laplacian /= np.max(laplacian)
    # # laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
    # sharpened = res_pp - laplacian
    # fig, (a1, a2, a3) = plt.subplots(1, 3)
    # a1.imshow(res_pp.T, origin="lower")
    # a2.imshow(laplacian.T, origin="lower")
    # a3.imshow(sharpened.T, origin="lower")
    # plt.show()
    # k = gauss_2d(sigma)
    # res_pp = convolve2d(res_pp, k, boundary='fill', mode='same')
    # res_pp = wiener(res_pp, mysize=(11, 11))
    # threshold = 0.02
    # res_pp[res_pp > threshold] = 1
    # res_pp[res_pp < 0.05] = 0
    res[3:3] = [res_pp]

    # plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey=True)
    ax1.tripcolor(points[:, 0], points[:, 1], triangles,
                  res[0], vmin=0, vmax=1)
    ax1.set_title("standard ROM with point_data")
    ax2.pcolormesh(X, Y, res[1].reshape(X.shape), vmin=0, vmax=1)
    ax2.set_title("standard ROM with grid_data")
    ax3.pcolormesh(X, Y, res[2].reshape(X.shape), vmin=0, vmax=.1)
    ax3.set_title("standard ROM with smooth grid_data")
    ax4.pcolormesh(X, Y, res[3].reshape(X.shape), vmin=0, vmax=.001)
    ax4.set_title("standard ROM with smooth grid_data + post_processing")
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
