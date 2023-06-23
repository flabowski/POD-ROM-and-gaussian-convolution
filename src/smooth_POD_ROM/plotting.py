import matplotlib.pyplot as plt
import numpy as np
# import vtk
# from vtk.numpy_interface import dataset_adapter as dsa
# from vtk.util.numpy_support import vtk_to_numpy
# from mpl_toolkits import axes_grid1
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
# from smooth_POD_ROM.io import on_regular_grid


def plot_field(polys, data, **kwargs):
    polyCollection = PolyCollection(polys, **kwargs)
    if "edgecolor" not in kwargs:
        polyCollection.set_edgecolor("face")
    polyCollection.set_array(data)

    ax = plt.gca()
    ax.add_collection(polyCollection)
    ax.set_aspect('equal')
    return polyCollection


def plot_mesh(points, polys):
    ax1 = plt.gca()
    collection = LineCollection(polys, colors='black')
    ax1.add_collection(collection)
    ax1.plot(points[:, 0], points[:, 1], "b.", ms=1, zorder=2)
    ax1.set_xlim([np.min(polys), np.max(polys)])
    ax1.set_ylim([np.min(polys), np.max(polys)])
    return


def plot_structured_field(X, Y, data_on_grid, mesh=False):
    # X, Y = np.meshgrid(x, y, indexing="ij")
    # dont use imshow!
    ax = plt.gca()
    if mesh:
        quadmesh = plt.pcolormesh(X, Y, data_on_grid, edgecolors='k', lw=1)
        ax.plot(X.ravel(), Y.ravel(), "b.", ms=1)
    else:
        quadmesh = plt.pcolormesh(X, Y, data_on_grid)
    ax.set_aspect("equal")
    return


def plot_unstructured_and_structured_field(case, field):
    x, y, data_on_grid = on_regular_grid(case, field)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = case.cellCentres

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.sca(ax1)
    # data = get_data(case, field)
    # polys = get_tri_mesh(case, field)
    plot_field(case, field, edgecolor='k')
    ax1.plot(points[:, 0], points[:, 1], "b.", ms=1)
    # plot_mesh(case)

    plt.sca(ax2)
    plot_structured_field(x, y, data_on_grid)
    plt.show()
    return


def plot_t30_matrix():
    directory = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/data/"
    snapshots, parameters, cases = Get_SnapsParam(
        directory+"/VTK_Legacy_NEW/", 10, 10000, 10,  5000, 30, 30)
    parameters = parameters[:, [0, 1, 3]]

    fig, axs = plt.subplots(5, 6)
    for i, ax in enumerate(axs.ravel()):
        # ax.axis("off")
        plt.sca(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plot_field(cases[i], "alpha.water",
                   plotBoundaries=False, colorbar=False)
    plt.tight_layout()
    plt.savefig(directory+"dam_break_pngs/000.pdf")
    plt.savefig(directory+"/dam_break_pngs/000.png", dpi=300)

    plt.close("all")
    for i in range(len(cases)):
        m, r, t = parameters[i][0], parameters[i][1], parameters[i][2]
        # x, y, data_on_grid = on_regular_grid(cases[i], "alpha.water")
        # fig, ax = plt.subplots()
        # plot_structured_field(x, y, data_on_grid)
        if t in np.arange(10, 100, 10):
            plot_field(cases[i], "alpha.water")
            plt.savefig(directory+"/dam_break_pngs/{:03d}_{:03d}_{:03d}.png".format(
                m, r, t))
            plt.close("all")
