
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

sigma = 1
truncate = 4
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)

X, Y = np.meshgrid(x, y, indexing="ij")
X_ss = np.zeros((len(x), len(y)))
c = [.3]

mask = ((X-c)**2 + (Y-c)**2)**.5 < 0.100001
X_ss[mask] = 1.0

kernel_size = int(4 * sigma) + 1
kernel1D = cv2.getGaussianKernel(kernel_size, sigma)
psk = np.outer(kernel1D, kernel1D)
data_smooth = convolve2d(X_ss, psk, boundary='symm', mode='same')
# data_smooth = gaussian_filter(X_ss, sigma=sigma, truncate=truncate)

identity_kernel = np.zeros_like(psk)
identity_kernel[kernel_size//2, kernel_size//2] = 1.0
for w in [.1, 1, 2, 3, 4, 5]:
    unsharpening_mask = identity_kernel + (identity_kernel-psk/w)*w

    data_sharpened = convolve2d(
        data_smooth, unsharpening_mask, boundary='symm', mode='same')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(X_ss.T)
    ax2.imshow(data_smooth.T)
    ax3.imshow(data_sharpened.T)
    plt.show()
