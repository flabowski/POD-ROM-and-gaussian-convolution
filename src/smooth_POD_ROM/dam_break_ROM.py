# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:10:15 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, Linear
from ezyrb import ReducedOrderModel as ROM
from matplotlib.collections import LineCollection
from turbulucid import Case, plot_field, add_colorbar  # , plot_vectors
from smooth_POD_ROM.pre_processing import Get_SnapsParam, get_polys, on_regular_grid


def plot_mesh(case):
    points = case.cellCentres
    # data = case[field]
    polys = get_polys(case)
    ax1 = plt.gca()
    collection = LineCollection(polys, colors='black')
    ax1.add_collection(collection)
    ax1.plot(points[:, 0], points[:, 1], "b.", ms=1, zorder=2)
    ax1.set_xlim([np.min(polys), np.max(polys)])
    ax1.set_ylim([np.min(polys), np.max(polys)])
    # plt.show()
    return


def plot_structured_field(case, field):
    x, y, data_on_grid = on_regular_grid(case, field)
    X, Y = np.meshgrid(x, y)
    points = case.cellCentres

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.sca(ax1)
    plot_field(case, field, edgecolor='k')
    ax1.plot(points[:, 0], points[:, 1], "b.", ms=1)
    # plot_mesh(case)

    plt.sca(ax2)
    # dont use imshow!
    quadmesh = ax2.pcolormesh(X, Y, data_on_grid, edgecolors='k', lw=1)
    add_colorbar(quadmesh)
    ax2.plot(X.ravel(), Y.ravel(), "b.", ms=1)
    ax2.set_aspect("equal")
    plt.show()
    return


if __name__ == "__main__":
    # mu_s, mu_e = 10, 100  # [10, 100, 1000, 2000, 5000]
    # rho_s, rho_e = 10,  50  # [10, 50, 100, 400, 800, 1200]
    # t_s, t_e = 0,  10  # [0, ..., 99]
    pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/data/VTK_Legacy_NEW/"
    snapshots, parameters, cases = Get_SnapsParam(
        pth, 10, 100, 10,  50, 0,  10)
    parameters = parameters[:, [0, 1, 3]]
    db = Database(parameters, snapshots)
    pod = POD('svd')  # reduction
    # issue: transform is expensive. why not keep it?
    rbf = RBF()  # approximation
    lin = Linear()  # approximation
    rom = ROM(db, pod, lin)
    rom.fit()

    new_mu = [50, 25, 1]
    pred_sol = rom.predict(new_mu)

    x, y, data_on_grid = on_regular_grid(cases[5], "alpha.water")
    plot_structured_field(cases[5], "alpha.water")
