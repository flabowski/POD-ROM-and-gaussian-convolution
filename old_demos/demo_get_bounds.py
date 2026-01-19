# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:33:08 2024

@author: florianma
"""
from scipy.interpolate._rgi_cython import evaluate_linear_2d, find_indices
from scipy.interpolate._rgi import _check_points, _ndim_coords_from_arrays
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
from scipy.interpolate._rgi import RegularGridInterpolator  # we do cheat
# from scipy.interpolate import RegularGridInterpolator  # dont use
import scipy.interpolate as ip


# points defining the regular grid in n dimensions
ax1 = np.linspace(0, 1, 10)  # have to be
points = ax1[:, None]
grid = (ax1, )
values = np.sin(points)
xi = np.random.rand(3, 1)

rgi = RegularGridInterpolator(grid, values)

f_xi = rgi(xi)

plt.plot(points[:, 0], values[:, 0], "C0.")
plt.plot(xi[:, 0], f_xi[:, 0], "C1.")

# this method might fail without error
indices, norm_distances = find_indices(grid, xi.T)
norm_distances2 = 1-norm_distances
norm_distances_nn = np.min(np.r_[norm_distances, norm_distances2], axis=0)


def relative_distance_nearest_neighbour(single_point, points):
    distances = distance.cdist(points, single_point, 'euclidean').flatten()
    distances.sort()
    dim = points.shape[1]
    n = 2**dim  # 1D: 2NN; 2D: 4NN, 3D:8NN
    return distances[0] / (np.sum(distances[:n])/dim)


print(norm_distances_nn)

for pt in xi:
    print(relative_distance_nearest_neighbour(pt[None], points))

asd
# test 2D

ax2 = np.linspace(-40, 40, 3)  # have to be
xx, yy = np.meshgrid(ax1, ax2, indexing="ij")
points = np.c_[xx.ravel(), yy.ravel()]
grid = (ax1, ax2)
values = np.sin(xx)+yy
xi = np.random.rand(3, 2)
xi[:, 1] *= 40

rgi = RegularGridInterpolator(grid, values)

f_xi = rgi(xi)

indices, norm_distances = find_indices(grid, xi.T)
norm_distances2 = 1-norm_distances
norm_distances_nn = np.min(np.r_[norm_distances, norm_distances2], axis=0)
print(norm_distances_nn)

for pt in xi:
    print(relative_distance_nearest_neighbour(pt[None], points))

plt.plot(xx.ravel(), yy.ravel(), "C0.")
plt.plot(xi[:, 0], xi[:, 1], "C1.")

asd
a1 = np.random.rand(1, 120)
a2 = np.sin(4*np.linspace(0, 2*np.pi, 1000, endpoint=False))
a3 = np.zeros(1000)
a3[250:300] = 1.2
a4 = np.zeros(1000)
a4[250:500] = 1.2

for a in [a4, a3, a2, a1]:
    w = 5

    def get_bounds(a, w=5):
        shape = a.shape
        a = np.atleast_2d(a)
        # w: window width in percent
        mina = np.zeros_like(a)
        maxa = np.zeros_like(a)
        w1 = int(a.shape[0]*w/100/2)
        w2 = int(a.shape[1]*w/100/2)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ri = np.arange(i-w1, i+w1+1) % a.shape[0]
                ci = np.arange(j-w2, j+w2+1) % a.shape[1]
                mina[i, j] = np.min(a[ri, :][:, ci])
                maxa[i, j] = np.max(a[ri, :][:, ci])
        mina.shape = maxa.shape = a.shape = shape
        return mina, maxa

    mina, maxa = get_bounds(a)
    # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # axs[0].imshow(a, interpolation="nearest")
    # axs[1].imshow(mina, interpolation="nearest")
    # axs[2].imshow(maxa, interpolation="nearest")
    # plt.show()

    plt.figure()
    plt.plot(a)
    plt.plot(mina)
    plt.plot(maxa)
    plt.show()
