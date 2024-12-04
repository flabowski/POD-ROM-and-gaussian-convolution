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
    on_regular_grid, _gauss_2d, to_frequency, to_space, add_padding, smoothen,
    remove_wall, threshold)
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
# ts = 0.3


def get_snapshotsJ():
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/"
    nu_all = [10, 100, 1000, 2000, 5000]   # 10*2**np.arange(1, 10, 2)
    rho_all = [10, 50, 100, 400, 800, 1200]  # 10*2**np.arange(1, 8, 1)
    t_all = [ts]  # np.arange(2, 102)
    X = []
    params = []
    for n in nu_all:
        print(n)
        for r in rho_all:
            for t in t_all:
                file = "damBreak_{:04d}_{:04d}_1_{:d}.vtk".format(n, r, t)
                data = get_field(pth+file, 'alpha.water')
                X.append(data)
                params.append([n, r, t])
    return np.array(X).T, np.array(params)


def get_snapshots(ts):
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/damBreak_results/"
    nu_all = 10*2**np.arange(0, 10, 1)
    rho_all = 10*2**np.arange(0, 8, 1)
    nu_all = [320,  640]
    rho_all = [40,   80]
    # nu_all = [160, 1280]
    # rho_all = [20,  160]
    # nu_all = [160, 1280]
    # rho_all = [20, 40, 80, 160]
    t_all = [ts]  # np.linspace(0.01, 1, 100)
    X = []
    params = []
    for n in nu_all:
        for r in rho_all:
            for t in t_all:
                file = "damBreak_m{:02.1f}_r{:02.1f}_t{:.2f}/internal.vtu".format(
                    n, r, t)
                data = get_field(pth+file, 'alpha.water')
                X.append(data)
                params.append([n, r, t])
    points3D = get_field(pth+file, 'points')
    forbidden_points = points3D[:, 2] > 0.0
    snapshots3D = np.array(X).T
    snapshots = snapshots3D[~forbidden_points, :]
    # snapshots[snapshots > .5] = 1.0
    # snapshots[snapshots < .5] = 0.0
    return snapshots, np.array(params)


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


def sharpen(data, psf=False, limit=False):
    # assert s % 2 == 1, "must be odd"
    # data_sharpened = data.copy()
    if isinstance(psf, np.ndarray):
        k = -psf.copy()
        n = k.shape[0]//2
        k[n, n] = 0
        k[n, n] = 1-np.sum(k)
    else:
        k = np.array([[0, -1, 0],
                      [-1,  5, -1],
                      [0, -1, 0]])
    # for i in range(s):
    data_sharpened = convolve2d(data, k, boundary='fill', mode='same')
    if limit:
        data_sharpened[data_sharpened < 0.0] = 0.0
        data_sharpened[data_sharpened > 1.0] = 1.0
    return data_sharpened


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
    # for s in [int(s)]:
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
            # # -----------------------------------------------------------------
            # med = np.median(window)
            # if np.sum(window) > 1:
            #     w2 = window.copy()
            #     w2[w2 < med] = 0
            #     w2[w2 > med] = 1
            #     data_pp[s1:e1, s2:e2] += w2
            # # -----------------------------------------------------------------
            k = np.int32(np.round(np.sum(window), decimals=0))
            # print(k, window.size)
            # assert k <= window.size, "{:.0f}, {:.0f}".format(
            #     k, window.size)
            # TODO: ValueError: kth(=-1) out of bounds (3)
            if k > 0:
                if k > window.size:
                    print("{:.0f}, {:.0f}".format(k, window.size))
                    k = window.size
                v, inds = _find_largest_elements_2d(window, k)
                assert len(inds[0]) == k
                data_pp[s1+inds[0], s2+inds[1]] += np.sum(window)/k
            n_evals[s1:e1, s2:e2] += 1
    data_t = data_pp.copy()/n_evals
    return data_t


def lmc(p_smooth):
    lmc3 = local_mass_conservation(p_smooth.copy(), 6)
    lmc2 = local_mass_conservation(lmc3.copy(), 5)
    lmc1 = local_mass_conservation(lmc2.copy(), 4)
    p_lmc = local_mass_conservation(lmc1.copy(), 3)
    p_lmc = crop(p_lmc.reshape(data_on_grid_s.shape), psf)
    # p_lmc = threshold(p_lmc)
    p_lmc = remove_wall(p_lmc, X, Y)
    return p_lmc


def crop(data_full, kernel):
    # psf = Point Spread Function
    px, py = kernel.shape
    px = (px+1)//2
    py = (py+1)//2
    cropped = data_full[px:-px, py:-py]
    return cropped


def make2D(points3D, point_data):
    forbidden_points = points3D[:, 2] > 0.0
    points = points3D[~forbidden_points][:, :2]
    for key, val in point_data.items():
        point_data[key] = point_data[key][~forbidden_points]
    return points, point_data


if __name__ == "__main__":
    # TODO: upsample?
    # TODO: optimize sharpening using test data
    pth_data = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/damBreak_results/damBreak_m450.0_r55.0_t"
    # snapshots on cells, points, grid, smooth in space, frequency
    # points: (9281, 3)  18562, 3
    # triangles: (18144, 3)
    # point_data.shape: (9281,)
    # p_test.shape: (93, 101)
    # data_on_grid_f.shape: (151, 159)
    # data_on_grid_s.shape: (151, 159)
    # psf (Point Spread Function): (57, 57)
    # psf_f (frequencies): (151, 159)
    timesteps = [.05, .10, .15, .2, .25, .3, .35, .4, .45, .5,
                 # 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
                 ]
    timesteps = np.linspace(0.01, 0.8, 80)
    sigmas = [0.25, 0.5, .75, 1, 1.125, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    n_sharpenings = [0, 1, 2, 3, 4, 5, 6, 7]
    # timesteps, sigmas, n_sharpenings = [0.45], [2.5], [5]
    # timesteps, sigmas, n_sharpenings = [0.3], [2.5], [5]
    # timesteps, sigmas, n_sharpenings = [0.3], [2.0], [3]
    # timesteps, sigmas, n_sharpenings = [0.3], [1.25], [0, 1, 2]
    # timesteps = [0.3]
    sigmas, n_sharpenings = [1.5], [0, 1, 2]
    e_ROM = np.zeros((len(timesteps), 2))
    e_sROM = np.zeros((len(timesteps), len(sigmas), 2))
    e_sROMs = np.zeros((len(timesteps), len(sigmas), len(n_sharpenings), 3))
    e_sROMlmc = np.zeros((len(timesteps), len(sigmas), 2))
    # e_sROMsc = np.zeros((len(timesteps), len(sigmas), len(n_sharpenings)))
    for i, ts in enumerate(timesteps):
        x_test = np.array([[450, 55, ts]])
        file = pth_data + '{:.2f}/internal.vtu'.format(ts)

        snapshots, p = get_snapshots(ts)
        # file = pth_data+"VTK_Legacy_NEW/damBreak_1000_0100_1_30.vtk"
        points3D, _, point_data_dict3, cell_data = get_data(file)
        points, pt_data_dct = make2D(points3D, point_data_dict3)
        point_data = pt_data_dct['alpha.water']
        X, Y, p_test = on_regular_grid(points,  pt_data_dct['alpha.water'],
                                       method="nearest")
        p_test = remove_wall(p_test, X, Y)
        x = np.unique(X)
        dx = x[1]-x[0]
        extent = (0-dx/2, 0.584+dx/2, 0-dx/2, 0.584+dx/2)
        plt.figure()
        plt.imshow(p_test.T, origin="lower",
                   interpolation="nearest", extent=extent)
        plt.plot(X.ravel(), Y.ravel(), "r.")
        plt.plot(points[:, 0], points[:, 1], "k.")
        plt.show()

        # p_test = threshold(p_test)
        p_test = remove_wall(p_test, X, Y)

        m, n = snapshots.shape
        X_grid = np.empty((p_test.size, n))
        for j in range(n):
            gridded = griddata(
                points[:, :2], snapshots[:, j], (X, Y), method="nearest")
            # gridded = threshold(gridded)
            gridded = remove_wall(gridded, X, Y)
            X_grid[:, j] = gridded.ravel()
        reg = RegularGrid()
        pod = POD()
        params = np.log2(p[:, :2]/10)
        db = Database(params, X_grid.T)
        rom = ROM(db, pod, reg)
        rom.fit()
        xi = np.log2(x_test[:, :2]/10)
        standard_ROM = rom.predict(xi).reshape(gridded.shape)

        e_ROM[i, 0] = np.mean((standard_ROM-p_test)**2)**.5
        standard_ROM2 = remove_wall(threshold(standard_ROM), X, Y)
        e_ROM[i, 1] = np.mean((standard_ROM2-p_test)**2)**.5
        for j, sigma in enumerate(sigmas):
            print(i, j)
            psf = _gauss_2d(sigma, truncate=4, size=False)
            data_on_grid_s = smoothen(p_test, psf)
            shape_padded = data_on_grid_s.shape

            X_grid_smooth = np.empty((data_on_grid_s.size, n))
            for jj in range(n):
                _on_grid = X_grid[:, jj].reshape(p_test.shape)
                _on_grid_s = smoothen(_on_grid, psf)
                X_grid_smooth[:, jj] = _on_grid_s.ravel()
            # ROM
            reg = RegularGrid()
            pod = POD()
            db = Database(params, X_grid_smooth.T)
            rom = ROM(db, pod, reg)
            rom.fit()
            # xi = np.log2(x_test[:, :2]/10)
            p_smooth_uncrpd = rom.predict(xi).reshape(shape_padded)
            p_smooth = p_smooth_uncrpd.copy()
            p_smooth = crop(p_smooth, psf)
            p_smooth = remove_wall(p_smooth, X, Y)

            e_sROM[i, j, 0] = np.mean((p_smooth-p_test)**2)**.5
            p_smooth2 = remove_wall(threshold(p_smooth), X, Y)
            e_sROM[i, j, 1] = np.mean((p_smooth2-p_test)**2)**.5

            p_lmc = lmc(p_smooth_uncrpd)
            # p_lmc = p_test
            e_sROMlmc[i, j, 0] = np.mean((p_lmc-p_test)**2)**.5
            p_lmc2 = remove_wall(threshold(p_lmc), X, Y)
            e_sROMlmc[i, j, 1] = np.mean((p_lmc2-p_test)**2)**.5

            # res_pp = deconvolution_gauss(data)
            # res_pp = laplacian_deconvolution(data)
            # res_pp = wiener(res_pp, mysize=(11, 11))
            # res_pp = local_mass_conservation(data)
            # p_sharp = richardson_lucy(p_smooth, psf, num_iter=30)
            # p_sharp = deconvolve_exact(p_smooth, psf, psf_f)
            p_sharp = p_smooth_uncrpd.copy()
            p_sharp_t = p_smooth_uncrpd.copy()
            for k, ns in enumerate(n_sharpenings):
                assert k == ns, "we are reusing results!"
                p_sharp_crpd = crop(p_sharp, psf)
                p_sharp_crpd = remove_wall(p_sharp_crpd, X, Y)
                e_sROMs[i, j, k, 0] = np.mean((p_sharp_crpd-p_test)**2)**.5

                p_sharp_crpd2 = remove_wall(threshold(p_sharp_crpd), X, Y)
                e_sROMs[i, j, k, 1] = np.mean((p_sharp_crpd2-p_test)**2)**.5

                p_sharp_t_crpd = crop(p_sharp_t, psf)
                p_sharp_t_crpd = threshold(p_sharp_t_crpd)
                p_sharp_t_crpd = remove_wall(p_sharp_t_crpd, X, Y)
                e_sROMs[i, j, k, 2] = np.mean((p_sharp_t_crpd-p_test)**2)**.5

                p_sharp = sharpen(p_sharp, psf=False)
                p_sharp_t = sharpen(p_sharp_t, psf=False, limit=True)

                # print(sigma, ts, ns,
                #       e_ROM[i],
                #       e_sROMs[i, j, k],
                #       e_sROMsc[i, j, k],
                #       # error_smooth_ROM_pred_sharp[i, j, k],
                #       np.min(p_sharp),
                #       np.max(p_sharp), sep="\t")
    # print(e)
    # plt.imshow(X_grid_smooth)
    # p_sharp = sharpen(p_smooth, 2)
    # p_sharp = crop(p_sharp.reshape(data_on_grid_s.shape), psf)
    # e = np.mean((p_sharp-p_test)**2)**.5
    # print(e)
    # # p_sharp = crop(predictions[1].reshape(p_test.shape), psf)
    # e = np.mean((predictions[1].reshape(
    #     p_test.shape)-p_test)**2)**.5
    # print(e)

    # p_sharp_crpd = local_mass_conservation(p_sharp, 7)
    # pred_sharp3 = local_mass_conservation(p_sharp_crpd, 10)
    # pred_sharp4 = local_mass_conservation(pred_sharp3, 5)
    # predictions[3:3] = [p_sharp]
    # predictions[4:4] = [p_sharp_crpd]

    # plotting
    # ground truth, standard ROM, s + ROM, s+ROM+s, s+ROM+lmc
    fig, ((ax11, ax12, ax13, ax14, ax15), (ax21, ax22, ax23, ax24, ax25),
          (ax31, ax32, ax33, ax34, ax35)) = plt.subplots(
        3, 5, sharex=True, sharey=True)
    # ax1.tripcolor(points[:, 0], points[:, 1], triangles,
    #               predictions[0], vmin=0, vmax=1)
    # ax1.set_title("standard ROM with point_data")
    # crop(standard_ROM.reshape(p_test.shape), psf)

    C1 = p_test
    # ax4.pcolormesh(X, Y, C, vmin=0, vmax=1)
    ax11.imshow(C1.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax11.set_title("ground truth")

    C12 = standard_ROM.reshape(p_test.shape)
    ax12.imshow(C12.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax12.set_title("standard ROM, e={:.4f}".format(np.mean((C12-C1)**2)**.5))
    C22 = remove_wall(threshold(C12), X, Y)
    ax22.imshow(C22.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax32.imshow(C22.T-C1.T, origin="lower", vmin=-
                1, vmax=1, interpolation="nearest")
    ax22.set_title("standard ROM, threshold, e={:.4f}".format(
        np.mean((C22-C1)**2)))

    C13 = p_smooth  # crop(p_smooth.reshape(data_on_grid_s.shape), psf)
    C13 = remove_wall(C13, X, Y)
    ax13.imshow(C13.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax13.set_title(
        "smoothing + ROM, e={:.4f}".format(np.mean((C13-C1)**2)**.5))
    C23 = remove_wall(threshold(C13), X, Y)
    ax23.imshow(C23.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax33.imshow(C23.T-C1.T, origin="lower", vmin=-
                1, vmax=1, interpolation="nearest")
    ax23.set_title(
        "smoothing + ROM, e={:.4f}".format(np.mean((C23-C1)**2)**.5))

    C14 = p_sharp_crpd
    ax14.imshow(C14.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax14.set_title(
        "sROM + sharpening, e={:.4f}".format(np.mean((C14-C1)**2)**.5))
    C24 = p_sharp_t_crpd
    ax24.imshow(C24.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax34.imshow(C24.T-C1.T, origin="lower", vmin=-
                1, vmax=1, interpolation="nearest")
    ax24.set_title(
        "sROM + sharpening, e={:.4f}".format(np.mean((C24-C1)**2)**.5))

    C15 = p_lmc
    ax15.imshow(C15.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax15.set_title(
        "sROM + lmc, e={:.4f}".format(np.mean((C15-C1)**2)**.5))
    C25 = remove_wall(threshold(C15), X, Y)
    ax25.imshow(C25.T, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax35.imshow(C25.T-C1.T, origin="lower", vmin=-
                1, vmax=1, interpolation="nearest")
    ax25.set_title(
        "sROM + lmc, e={:.4f}".format(np.mean((C25-C1)**2)**.5))

    # C6 = p_sharp.copy()
    # # ax4.pcolormesh(X, Y, C, vmin=0, vmax=1)
    # ax6.imshow(C6.T, origin="lower", vmin=0, vmax=1)
    # ax6.set_title(
    #     "smoothing + ROM + sharpening + threshold, e={:.4f}".format(np.mean((C6-C1)**2)))
    plt.show()

    fig, ax = plt.subplots()
    for k, ns in enumerate(n_sharpenings):
        plt.plot(sigmas, np.mean(
            e_sROMs[:, :, k, 0], axis=0), label="n={:2d}".format(ns))
        # plt.plot(sigmas, np.mean(
        #     e_sROMs[:, :, k, 1], axis=0), label="n={:2d} (thr)".format(ns))
    plt.plot(sigmas, np.ones_like(sigmas)*np.mean(e_ROM[:, 0]), "ro")
    # plt.plot(sigmas, np.ones_like(sigmas)*np.mean(e_ROM[:, 1]), "go")
    plt.xlabel("sigma")
    plt.legend()
    plt.ylim([0.05, 0.1])
    plt.show()

    fig, ax = plt.subplots()
    # for j, sigma in enumerate(sigmas):
    #     for k, ns in enumerate(n_sharpenings):
    #         e1, e2 = e_sROMs[:, j, k, 0], e_sROMs[:, j, k, 1]
    #         print(sigma, ns, np.mean(e1), np.mean(e2), np.sum(
    #             e1 < e_ROM[:, 0]), np.sum(e2 < e_ROM[:, 1]), sep="\t")
    #         plt.plot(timesteps, e1, label="n={:2d}".format(ns))
    #         plt.plot(timesteps, e2, marker="o", label="n={:2d} (thr)".format(ns))
    # plt.plot(timesteps, e_ROM[:, 0], "r--", label="standard ROM")
    plt.plot(timesteps, e_sROMs[:, 0, 1, 1], marker="o",
             label="smoothin+ROM+sharpening".format(ns))
    # plt.plot(timesteps, e_sROMs[:, 0, 1, 2], marker="o",
    #          label="smoothin+ROM+bounded sharpening".format(ns))
    plt.plot(timesteps, e_ROM[:, 1], marker="o", label="standard ROM")
    # plt.ylim([.8, 1.2])
    plt.xlabel("time")
    plt.legend()
    plt.show()

    for i, ts in enumerate(timesteps):
        ind_flat = np.argmin(e_sROMs[i, :, :, 1])
        j, k = np.unravel_index(ind_flat, e_sROMs[i, :, :, 1].shape)
        print(i, ts, sigmas[j], n_sharpenings[k],
              e_sROMs[i, j, k, 0],
              e_sROMs[i, j, k, 1],
              # error_smooth_ROM_pred_sharp[i, j, k],
              e_ROM[i, 0], sep="\t")

    for j, sigma in enumerate(sigmas):
        for k, ns in enumerate(n_sharpenings):
            print(j, k, sigmas[j], n_sharpenings[k],
                  np.min(e_sROMs[:, j, k, 1] / e_ROM),
                  np.max(e_sROMs[:, j, k, 1] / e_ROM),
                  np.mean(e_sROMs[:, j, k, 1] / e_ROM),
                  # np.min(e_sROMlmc[:, j, k, 1] / e_ROM),
                  # np.max(e_sROMlmc[:, j, k, 1] / e_ROM),
                  # np.mean(e_sROMlmc[:, j, k, 1] / e_ROM),
                  sep="\t")

    # j, k = 0, 1
    # for i, ts in enumerate(timesteps):
    #     print(j, k, ts,
    #           e_sROMs[i, j, k] / e_ROM[i],
    #           sep="\t")
    asd
    # optimize local mass conservation
    p_sharp = sharpen(p_smooth, ns)
    p_sharp_crpd = sharpen(p_smooth, ns, crop=True)
    p_sharp = crop(p_sharp.reshape(data_on_grid_s.shape), psf)
    p_sharp_crpd = crop(p_sharp_crpd.reshape(data_on_grid_s.shape), psf)
    e_sROMs[i, j, k] = np.mean((p_sharp-p_test)**2)**.5
    e_sROMsc[i, j, k] = np.mean((p_sharp_crpd-p_test)**2)**.5
    # for repert in [1, 2, 3, 4]:
    for s in [3, 5, 7, 9, 11, 15]:
        p_sharp = local_mass_conservation(p_smooth, s)
        p_sharp = crop(p_sharp.reshape(data_on_grid_s.shape), psf)
        e = np.mean((p_sharp-p_test)**2)**.5
        print(s, e)

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
