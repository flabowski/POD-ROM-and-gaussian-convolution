import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from scipy.interpolate import griddata
from smooth_POD_ROM.pre_processing import (
    on_regular_grid, _gauss_2d, to_frequency, to_space, add_padding)
from smooth_POD_ROM.dam_break_ROM import make2D
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
from scipy.interpolate import griddata

ts = .35
pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
    "/Documents/data/damBreak_results/"


def get_snapshots():

    nu_all = 10*2**np.arange(0, 10, 1)
    rho_all = 10*2**np.arange(0, 8, 1)
    t_all = [ts]  # np.linspace(0.01, 1, 100)
    X = []
    params = []
    pts = [[320, 40],
           [320, 80],
           [640, 40],
           [640, 80]]
    for nu, rho in zip(pts):
        file = "damBreak_m{:02.1f}_r{:02.1f}_t{:.2f}/internal.vtu".format(
            nu, rho, ts)
        data = get_field(pth+file, 'alpha.water')
        X.append(data)
        params.append([nu, rho, ts])
    return np.array(X).T, np.array(params)


if __name__ == "__main__":

    pts = [[320, 40],
           [320, 80],
           [640, 40],
           [640, 80],
           [450, 55]]
    to_plot = []

    nu_all = 10*2**np.arange(0, 10, 1)
    rho_all = 10*2**np.arange(0, 8, 1)
    # nu_all = 10*2**np.arange(0, 10, 3)
    # rho_all = 10*2**np.arange(0, 8, 1)
    # nu_all = [320, 640]
    # rho_all = [40, 80]
    nu_all = [160, 1280]
    rho_all = [20, 40, 80, 160]

    nu_rand = 10*2**(np.random.rand(80)*9)
    rho_rand = 10*2**(np.random.rand(80)*7)
    queue = [[450, 55]]
    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(nu_all, rho_all)
    ax.plot(xx.ravel(), yy.ravel(), "bx", label="training dataset")
    ax.plot([450], [55], "g*", ms=10, label="test dataset")
    ax.plot(nu_rand, rho_rand, "r+", label="validation dataset")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(nu_all)
    plt.yticks(rho_all)
    ax.set_xticklabels(nu_all)
    ax.set_yticklabels(rho_all)
    ax.grid()
    ax.legend()
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.show()
    # ax.grid(which="minor", ls=':', color='red', linewidth=0.6)

    fig, axs = plt.subplots(len(nu_all), len(rho_all),
                            sharex=True, sharey=True)

    for i, nu in enumerate(nu_all):
        for j, rho in enumerate(rho_all):
            ax = axs[i][j]
            file = "damBreak_m{:02.1f}_r{:02.1f}_t{:.2f}/internal.vtu".format(
                nu, rho, ts)
            points3D, _, point_data, cell_data = get_data(pth+file)
            points, point_data = make2D(points3D, point_data)
            # to_plot.append(point_data['alpha.water'])
        # for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.tripcolor(points[:, 0], points[:, 1],  # triangles,
                         point_data['alpha.water'], vmin=0, vmax=1)
