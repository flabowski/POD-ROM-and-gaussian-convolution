# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:58:28 2023

@author: florianma
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the four points of the unit square
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Define the two triangles
triangles = [[0, 1, 2], [1, 3, 2]]

# Define the CG2 basis functions


def phi_1(x, y):
    return (2*x - 1)*x


def phi_2(x, y):
    return (2*y - 1)*y


def phi_3(x, y):
    return 4*x*y


def phi_4(x, y):
    return 4*x*(1-x-y)


def phi_5(x, y):
    return 4*y*(1-x-y)


def phi_6(x, y):
    return 4*x*y


basis_functions = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6]

# Plot the CG2 basis functions for each point in the triangle
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, phi in enumerate(basis_functions):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = phi(X, Y)

    # Mask values outside the triangle
    for triangle in triangles:
        if np.any(points[triangle, 0] + points[triangle, 1] > 1):
            Z[(X + Y) > 1+1e-6] = np.nan
    print(np.nanmin(Z), np.nanmax(Z))
    ax = axs[i//3, i % 3]
    c = ax.contourf(X, Y, Z, 50, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f"Basis function {i+1}")
    fig.colorbar(c, ax=ax)

plt.tight_layout()
plt.show()


# Define the local basis functions for CG2 on a reference triangle
def phi_local(i, xi, eta):
    if i == 1:
        return 2*xi**2 - xi
    elif i == 2:
        return 2*eta**2 - eta
    elif i == 3:
        return 4*xi*eta
    elif i == 4:
        return 4*xi*(1-xi-eta)
    elif i == 5:
        return 4*eta*(1-xi-eta)
    elif i == 6:
        return 4*xi*eta


def transform_to_global(triangle, xi, eta):
    # Triangle vertices
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]

    # Compute the Jacobian matrix
    J = np.array([[x2 - x1, x3 - x1],
                  [y2 - y1, y3 - y1]])

    # Compute the global coordinates
    x, y = np.dot(J, [xi, eta]) + triangle[0]

    return x, y


def transform_to_local(triangle, x, y):
    # Triangle vertices
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]

    # Compute the Jacobian matrix
    J = np.array([[x2 - x1, x3 - x1],
                  [y2 - y1, y3 - y1]])

    # Compute the inverse of the Jacobian matrix
    J_inv = np.linalg.inv(J)

    # Compute the local coordinates
    xi, eta = np.dot(J_inv, [x - x1, y - y1])

    return xi, eta
