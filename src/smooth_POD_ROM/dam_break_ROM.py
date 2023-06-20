# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:10:15 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ezyrb import POD, RBF, Database, Linear
from ezyrb import ReducedOrderModel as ROM
from matplotlib.collections import LineCollection
import glob
import re
from turbulucid import Case, plot_field, add_colorbar  # , plot_vectors


def Get_SnapsParam(pth, mu_s, mu_e, rho_s, rho_e, t_s, t_e):
    """
    Data file names have the following format:

    damBreak_0010_0050_1_22.vtk

    The first number is the Viscosity ratio * 10
    The second number is the  Density ratio
    The third number is  always 1
    The fourth number is the Time sequence number.
    """

    files = glob.glob(pth+'*.vtk')
    snaps = []
    params = []
    cases = []

    for file in files:
        nu_r, rho_r, one, t = [int(i) for i in re.findall(r'\d+', file)]
        # print(nu_r, rho_r, one, t)
        nu_in_range = mu_s <= nu_r <= mu_e
        rho_in_range = rho_s <= rho_r <= rho_e
        t_in_range = t_s <= t <= t_e

        if nu_in_range and rho_in_range and t_in_range:
            # ['cellID', 'p', 'p_rgh', 'alpha.water', 'U', 'UMag']
            snapshot = Case(file)
            snapshot["UMag"] = np.linalg.norm(snapshot["U"], axis=1)

            params.append([nu_r, rho_r, one, t])
            snaps.append(snapshot['alpha.water'])
            cases.append(snapshot)

    snapshots = np.vstack(snaps)  # make a 2d_array of the snapshots
    parameters = np.vstack(params)
    # param = param.transpose()    #make a 2d_array (nt,1) of time instances
    print(snapshots.shape, parameters.shape)

    return snapshots, parameters, cases

# def to_snapshot_matrix(cases):
    # return


def get_polys(case):
    clippedData = case.vtkData
    # fig, ax = plt.subplots()
    # ax = plt.gca()
    polys = []
    for i in range(clippedData.GetNumberOfCells()):
        cell = clippedData.GetCell(i)
        nPoints = cell.GetNumberOfPoints()
        pts = np.zeros((nPoints, 2))
        for pointI in range(nPoints):
            pts[pointI, :] = cell.GetPoints().GetPoint(pointI)[:2]
        polys.append(pts)
    return polys


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


def on_regular_grid(case, field):
    points = case.cellCentres  # centers of triangles
    data = case[field]

    # Define the regular grid
    # grid_size = 100  # Adjust this value as needed
    # x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), grid_size)
    # y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), grid_size)

    edges = np.array(get_polys(case)).reshape(-1, 2)
    x = np.unique(np.round(edges[:, 0], decimals=8))
    y = np.unique(np.round(edges[:, 1], decimals=8))
    xcenter = (x[:-1]+x[1:]) / 2
    ycenter = (y[:-1]+y[1:]) / 2
    X, Y = np.meshgrid(xcenter, ycenter)

    # Project the velocity data onto the regular grid
    data_on_grid = griddata(points[:, :2], data, (X, Y), method='linear')
    return xcenter, ycenter, data_on_grid


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
