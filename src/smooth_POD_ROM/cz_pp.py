# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from scipy.interpolate import griddata
from scipy import spatial
from smooth_POD_ROM.pre_processing import on_regular_grid, _gauss_2d
from smooth_POD_ROM.io import get_data, get_field
from smooth_POD_ROM.plotting import plot_mesh, plot_field, plot_structured_field
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener, convolve2d
from skimage import restoration
from skimage.restoration import richardson_lucy
import tvtk
import cv2
from datetime import datetime
np.set_printoptions(suppress=True)


def get_snapshots(include_mesh=True):
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/workdir3/"
    # ps_all = [0, 1.6, 1.7, 2.0, 2.4, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4, 17]
    ps_all = [1.6, 1.7, 2.0, 2.4, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4]
    p2_all = [50, 100, 200, 300, 400]
    t_all = [19]  # np.arange(0, 20)
    X = []
    params = []
    for ps in ps_all:
        print(ps)
        for p2 in p2_all:
            for t in t_all:
                # file = "damBreak_{:04d}_{:04d}_1_{:d}.vtk".format(m, r, t)
                if ps == 0:
                    file = "pull_speed_0_mmpm_{:d}/result.{:04d}.vtk".format(
                        p2, t)
                elif ps == 17:
                    file = "pull_speed_17_mmpm_{:d}/result.{:04d}.vtk".format(
                        p2, t)
                else:
                    file = "pull_speed_{:.1f}_mmpm_{:d}/result.{:04d}.vtk".format(
                        ps, p2, t)
                print(file)
                if os.path.isfile(pth+file):
                    data = get_field(pth+file, 'Frac_Solid')
                    data = get_field(pth+file, 'Frac_Solid')
                    print(data.shape)
                    X.append(data)
                    params.append([ps, p2, t])
                else:
                    print(file+" is not a file")
    return np.array(X).T, np.array(params)


if __name__ == "__main__":
    sigma = 7
    snapshots, p = get_snapshots()
    file = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/workdir3/pull_speed_2.4_mmpm_300/result.0019.vtk"
    # "/Documents/data/workdir2/pull_speed_0_mmpm_50/result.0001.vtk"
    points, triangles, point_data, cell_data = get_data(file)
    # TODO: points are ordered differently for each parameter
    print(points.shape)
    print(triangles.shape)
    print(snapshots.shape)
    for key in point_data.keys():
        print(point_data[key].shape)
    fig, axs = plt.subplots(10, 5, sharex="all", sharey="all")
    for i, param in enumerate(p):
        print(param)
        ax = axs.ravel()[i]
        # plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.scatter(points[:, 0], points[:, 1], c=snapshots[:, i], s=40)
        ax.set_title("{:.0f}".format(param[1]))
        ax.set_xlim(.10, .16)
        ax.set_ylim(.19, .21)
        ax.set_ylabel("{:.1f}".format(param[0]))
    # plt.tight_layout()
    plt.show()
    pt_s = np.array([0.15, 0.20343532, 0.0])  # 3859
    pt_m = np.array([0.15, 0.2, 0.0])
    pt_l = np.array([0.151875, 0.2, 0.0])  # 3545
    marker = np.zeros_like(p[:, 0])  # 3859, 3545
    for pt, desired_fraction in zip([pt_s, pt_l], [1, 0]):
        # print(spatial.KDTree(points).query(pt))
        pi = spatial.KDTree(points).query(pt)[1]
        print(pi)
        # p[:, 2] = snapshots[pi]
        # print(p)
        marker += snapshots[pi] == desired_fraction
        fig, ax = plt.subplots()
        l = snapshots[pi] == desired_fraction
        plt.plot(p[:, 0], p[:, 1], "ro")
        plt.plot(p[l, 0], p[l, 1], "go")
        plt.show()
    print(marker)
    all_solid = (snapshots[3859] == 1) & (snapshots[3545] == 1)
    all_liquid = (snapshots[3859] == 0) & (snapshots[3545] == 0)
    as_desired = (snapshots[3859] == 1) & (snapshots[3545] == 0)
    partial_liquid = (0 < snapshots[3859]) & (
        snapshots[3859] < 1) & (snapshots[3545] == 0)
    fig, ax = plt.subplots()
    plt.plot(p[all_solid, 1], p[all_solid, 0], "bo", label="both solid")
    plt.plot(p[all_liquid, 1], p[all_liquid, 0], "ro", label="both liquid")
    plt.plot(p[as_desired, 1], p[as_desired, 0],
             "go", label="one liquid, one solid")
    plt.plot(p[partial_liquid, 1], p[partial_liquid, 0], 'o',
             color="orange", label="one partially liquid, one solid")
    ax.invert_yaxis()
    plt.legend()
    # ps_all = [1.6, 1.7, 2.0, 2.4, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4]
    # p2_all = [50, 100, 200, 300, 400]
    ax.set_yticks([1.6, 1.7, 2.0, 2.4, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4])
    ax.set_xticks([50, 100, 200, 300, 400])
    ax.set_ylabel("pull speed [mm/s]")
    ax.set_title("state of corner nodes at t = {:.1f}".format(19))
    plt.show()
    # points[3859], points[3545], points[3543]
    # snapshots[3859], snapshots[3545]

    # points[spatial.KDTree(points).query(pt)[1]] # <-- the nearest point
