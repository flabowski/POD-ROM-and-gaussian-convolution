
import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from scipy.interpolate import griddata
# from smooth_POD_ROM.pre_processing import (
#     on_regular_grid, _gauss_2d, to_frequency, to_space, add_padding, smoothen)
from smooth_POD_ROM.io import get_data, get_field
from smooth_POD_ROM.plotting import plot_mesh, plot_field, plot_structured_field
from copy import deepcopy
from scipy.ndimage import gaussian_filter
# from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.signal import wiener, convolve2d
from skimage import restoration
from skimage.restoration import richardson_lucy
import tvtk
import cv2
from datetime import datetime
from scipy.interpolate import griddata


t = np.linspace(0, 1, 11)
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)

X, Y = np.meshgrid(x, y, indexing="ij")
X_ss = np.zeros((len(t), len(x), len(y)))
for i in range(len(t)):
    c = t[i]
    mask = ((X-c)**2 + (Y-c)**2)**.5 < 0.0500001
    X_ss[i, mask] = 1.0
    if i % 5 == 0 or i < 5:
        fig = plt.figure()
        plt.imshow(X_ss[i])
        plt.show()

s = 1
data_on_grid_s = gaussian_filter(X_ss, sigma=(s, s, s))
i = 5
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(X_ss[i])
ax2.imshow(data_on_grid_s[i])
plt.show()

for i in range(21):
    if i % 5 == 0 or i < 5:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(X_ss[i])
        ax2.imshow(data_on_grid_s[i])
        plt.show()
