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


def get_snapshots():
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/"
    mu_all = [10, 100, 1000, 2000, 5000]
    rho_all = [10, 50, 100, 400, 800, 1200]
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


if __name__ == "__main__":
    snapshots, p = get_snapshots()
    file = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/damBreak_0010_0010_1_30.vtk"
    points, triangles, point_data_dict = get_data(file)
    polys = (points[:, :2])[triangles]
    data = np.array(point_data_dict['alpha.water'])

    fig, ax = plt.subplots()
    plot_mesh(points, polys)
    plt.show()

    fig, ax = plt.subplots()
    plot_field(polys, data)
    plt.show()

    X, Y, data_on_grid = on_regular_grid(points[:, :2],  data)
    data_on_grid_s = gaussian_filter(data_on_grid, sigma=5)

    fig, ax = plt.subplots()
    plot_structured_field(X, Y, data_on_grid)
    plt.show()

    fig, ax = plt.subplots()
    plot_structured_field(X, Y, data_on_grid_s)
    plt.show()

    m, n = snapshots.shape
    X_s = np.empty((X.size, n))
    for j in range(n):
        print(j)
        data_on_grid = griddata(points[:, :2], snapshots[:, j], (X, Y))
        data_on_grid_s = gaussian_filter(data_on_grid, sigma=5)
        X_s[:, j] = data_on_grid_s.ravel()
    fig, ax = plt.subplots()
    plot_structured_field(X, Y, data_on_grid)
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
