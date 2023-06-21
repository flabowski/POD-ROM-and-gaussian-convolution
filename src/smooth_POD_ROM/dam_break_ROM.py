# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:10:15 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from matplotlib.collections import LineCollection
from turbulucid import Case, plot_field, add_colorbar  # , plot_vectors
from smooth_POD_ROM.pre_processing import Get_SnapsParam, get_tri_mesh, on_regular_grid
from copy import deepcopy
from smooth_POD_ROM.io import get_field


def get_snapshots():
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/data/VTK_Legacy_NEW/"
    mu_all = [10, 100, 1000, 2000, 5000]
    rho_all = [10, 50, 100, 400, 800, 1200]
    t_all = np.arange(100)
    X = []
    for m in mu_all:
        for r in rho_all:
            for t in t_all:
                file = "damBreak_{:04d}_{:04d}_1_{:d}.vtk".format(m, r, t)
                data = get_field(pth+file, 'alpha.water')
                X.append(data)
    return np.array(X)


if __name__ == "__main__":
    # mu_s, mu_e = 10, 100  # [10, 100, 1000, 2000, 5000]
    # rho_s, rho_e = 10,  50  # [10, 50, 100, 400, 800, 1200]
    # t_s, t_e = 0,  10  # [0, ..., 99]
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/data/VTK_Legacy_NEW/"
    snapshots, parameters, cases = Get_SnapsParam(
        pth, 10, 10000, 10,  5000, 30, 30)
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
