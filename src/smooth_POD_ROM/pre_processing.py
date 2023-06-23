import numpy as np
from scipy.interpolate import griddata


def get_centers(points, triangles):
    return np.mean(points[triangles], axis=1)


def on_regular_grid(points, data, **kvargs):
    x = np.unique(np.round(points[:, 0], decimals=8))
    y = np.unique(np.round(points[:, 1], decimals=8))
    # xcenter = (x[:-1]+x[1:]) / 2
    # ycenter = (y[:-1]+y[1:]) / 2
    # assert len(xcenter) > 1, "not enough data"
    # assert len(ycenter) > 1, "not enough data"
    X, Y = np.meshgrid(x, y, indexing="ij")
    data_on_grid = griddata(points[:, :2], data,
                            (X, Y), **kvargs)
    return X, Y, data_on_grid
