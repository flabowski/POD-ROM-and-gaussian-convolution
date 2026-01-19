import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, wiener, convolve
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from smooth_POD_ROM.pre_processing import add_padding, remove_padding
from smooth_POD_ROM.pre_processing import to_frequency, to_space
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
import cv2

# edge detection, edge enhancement, unsharp masking, image sharpening


m, n = 71, 71
sigma = 0.5
truncate = 4

x, y = np.linspace(0, 1, m)[:, None], np.linspace(0, 1, n)[None]
image = x*y*0
image[x < 1-y+1e-6] = 1
image[x+1e-6 > 1-y] = -1


N = n
x = np.linspace(-(N//2), N//2, N, endpoint=True)
X, Y = np.meshgrid(x, x)
D = (X**2+Y**2)**.5
image = np.zeros_like(D)-1
image[D < 20] = 1

kernel_size = int(4 * sigma) + 1
kernel1D = cv2.getGaussianKernel(kernel_size, sigma)
kernel = np.outer(kernel1D, kernel1D)
# assert np.allclose(psk, kernel), "kernel not the same"
# t1 = datetime.now()
# data_smooth = gaussian_filter(data, sigma=sigma, truncate=truncate)
# t2 = datetime.now()
# data_smooth2 = convolve2d(data, psk, boundary='symm', mode='same')
# t3 = datetime.now()

kernel_padded = add_padding(kernel, image.shape, mode="constant")
image_padded = add_padding(image, kernel.shape, mode="symmetric")

kernel_f = to_frequency(kernel_padded, shift=True)
image_f = to_frequency(image_padded)

image_smooth_f = kernel_f*image_f
image_smooth = to_space(image_smooth_f)

data_smooth3 = remove_padding(image_smooth, image.shape)

# x = np.linspace(-100, 100, 2001, endpoint=True)
# y_orig = x.copy()
# y_orig[x < 0] = -1
# y_orig[x > 0] = 1
# y = gaussian_filter(image, sigma)


def get_kernel(s):
    kernel = -np.ones((s, s))
    kernel[s//2, s//2] = s**2
    if s == 3:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
    if s == 5:
        kernel = np.array([[0, 0, -1, 0, 0],
                           [0, -1, -1, -1, 0],
                           [-1, -1, 13, -1, -1],
                           [0, -1, -1, -1, 0],
                           [0, 0, -1, 0, 0]])
    return kernel


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image_padded.T, origin="lower")
ax2.imshow(image_smooth.T, origin="lower")
plt.show()


k1 = np.array([-1, 3, -1])
k2 = np.array([-1, -1, 5, -1, -1])
k3 = np.array([-1, -1, -1, 7, -1, -1, -1])
k4 = np.array([-1, -1, -1, -1, 9, -1, -1, -1, -1])
k5 = np.array([-1, -1, -1, -1, -1, 11, -1, -1, -1, -1, -1])

# vary kernel size
i = image_padded.shape[0]//2
fig, ax = plt.subplots()
ax.plot(image_padded[i, :], label="orig")
ax.plot(image_smooth[i, :], label="smooth")
# lbl = ["81", "41", "gauss", "linear81", "linear161", "circle", "hybrid"]
for j, s in enumerate([3, 5]):
    kernel = get_kernel(s)
    y2 = convolve2d(image_smooth, kernel, mode="same")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(image_padded.T, origin="lower", vmin=-2, vmax=2,
               interpolation="nearest")
    ax2.imshow(kernel.T, origin="lower", interpolation="nearest", vmax=0)
    ax3.imshow(image_smooth.T, origin="lower", vmin=-2, vmax=2,
               interpolation="nearest")
    ax4.imshow(y2.T, origin="lower", vmin=-2, vmax=2, interpolation="nearest")
    plt.show()
    ax.plot(y2[i, :], label=str(s))
ax.legend()
ax.set_xlim([50, 70])
ax.set_ylim([-2, 2])
plt.show()

# vary number of runs
fig, ax = plt.subplots()
ax.plot(image_padded[i, :], label="orig")
ax.plot(image_smooth[i, :], label="smooth")
y2 = image_smooth.copy()
for j in range(5):
    s = 3
    kernel = get_kernel(s)

    y2 = convolve2d(y2, kernel, mode="same")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(image_padded.T, origin="lower", vmin=-2, vmax=2,
               interpolation="nearest")
    ax2.imshow(kernel.T, origin="lower", interpolation="nearest", vmax=0)
    ax3.imshow(image_smooth.T, origin="lower", vmin=-2, vmax=2,
               interpolation="nearest")
    ax4.imshow(y2.T, origin="lower", vmin=-2, vmax=2, interpolation="nearest")
    plt.show()
    ax.plot(y2[i, :], label=str(j))
ax.legend()
ax.set_xlim([50, 70])
ax.set_ylim([-2, 2])
plt.show()
