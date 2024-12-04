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

# m, n = 71, 71
sigma = 0.5
# truncate = 4

# x, y = np.linspace(0, 1, m)[:, None], np.linspace(0, 1, n)[None]
# image = x*y*0
# image[x < 1-y+1e-6] = 1
# image[x+1e-6 > 1-y] = -1

# N = n
# x = np.linspace(-(N//2), N//2, N, endpoint=True)
# X, Y = np.meshgrid(x, x)
# D = (X**2+Y**2)**.5
# image = np.zeros_like(D)-1
# image[D < 20] = 1

kernel_size = int(4 * sigma) + 1
kernel1D = cv2.getGaussianKernel(kernel_size, sigma)
kernel = np.outer(kernel1D, kernel1D)
# # assert np.allclose(psk, kernel), "kernel not the same"
# # t1 = datetime.now()
# # data_smooth = gaussian_filter(data, sigma=sigma, truncate=truncate)
# # t2 = datetime.now()
# # data_smooth2 = convolve2d(data, psk, boundary='symm', mode='same')
# # t3 = datetime.now()

# kernel_padded = add_padding(kernel, image.shape, mode="constant")
# image_padded = add_padding(image, kernel.shape, mode="symmetric")

# kernel_f = to_frequency(kernel_padded, shift=True)
# image_f = to_frequency(image_padded)

# image_smooth_f = kernel_f*image_f
# image_smooth = to_space(image_smooth_f)

# data_smooth3 = remove_padding(image_smooth, image.shape)
image_padded = np.load(
    "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/damBreakROM/p_test.npy")
image_smooth = np.load(
    "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/damBreakROM/p_smooth.npy")
# x = np.linspace(-100, 100, 2001, endpoint=True)
# y_orig = x.copy()
# y_orig[x < 0] = -1
# y_orig[x > 0] = 1
# y = gaussian_filter(image, sigma)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image_padded.T, origin="lower")
ax2.imshow(image_smooth.T, origin="lower")
plt.show()


k1 = np.array([-1, 3, -1])
k2 = np.array([-1, -1, 5, -1, -1])
k3 = np.array([-1, -1, -1, 7, -1, -1, -1])
k4 = np.array([-1, -1, -1, -1, 9, -1, -1, -1, -1])
k5 = np.array([-1, -1, -1, -1, -1, 11, -1, -1, -1, -1, -1])

kernel0 = -np.outer(kernel1D, kernel1D)
n = kernel0.shape[0]//2
kernel0[n, n] = 0
kernel0[n, n] = 1-np.sum(kernel0)

kernel1 = np.array([[-1., -1., -1.],
                    [-1., 9., -1.],
                    [-1., -1., -1.]])
kernel2 = np.array([[-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., 25., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.]])
kernel3 = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., 49., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1.]])
kernel4 = np.array([[-1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., 81., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1., -1., -1.]])
kernel5 = np.array([[0,  0,  0,  0, -1,  0,  0,  0,  0],
                    [0,  0,  0, -1, -1, -1,  0,  0,  0],
                    [0,  0, -1, -1, -1, -1, -1,  0,  0],
                    [0, -1, -1, -1, -1, -1, -1, -1,  0],
                    [-1, -1, -1, -1, 41, -1, -1, -1, -1],
                    [0, -1, -1, -1, -1, -1, -1, -1,  0],
                    [0,  0, -1, -1, -1, -1, -1,  0,  0],
                    [0,  0,  0, -1, -1, -1,  0,  0,  0],
                    [0,  0,  0,  0, -1,  0,  0,  0,  0]])
kernel6 = np.array([[-0.00076345, -0.00183141, -0.00342153, -0.0049783, -0.00564116, -0.0049783, -0.00342153, -0.00183141, -0.00076345],
                    [-0.00183141, -0.00439334, -0.00820783, -0.01194233, -
                        0.01353243, -0.01194233, -0.00820783, -0.00439334, -0.00183141],
                    [-0.00342153, -0.00820783, -0.01533425, -0.0223112, -
                        0.0252819, -0.0223112, -0.01533425, -0.00820783, -0.00342153],
                    [-0.0049783, -0.01194233, -0.0223112, -0.03246261, -
                        0.03678495, -0.03246261, -0.0223112, -0.01194233, -0.0049783],
                    [-0.00564116, -0.01353243, -0.0252819, -0.03678495,
                        1.95831716, -0.03678495, -0.0252819, -0.01353243, -0.00564116],
                    [-0.0049783, -0.01194233, -0.0223112, -0.03246261, -
                        0.03678495, -0.03246261, -0.0223112, -0.01194233, -0.0049783],
                    [-0.00342153, -0.00820783, -0.01533425, -0.0223112, -
                        0.0252819, -0.0223112, -0.01533425, -0.00820783, -0.00342153],
                    [-0.00183141, -0.00439334, -0.00820783, -0.01194233, -
                        0.01353243, -0.01194233, -0.00820783, -0.00439334, -0.00183141],
                    [-0.00076345, -0.00183141, -0.00342153, -0.0049783, -0.00564116, -0.0049783, -0.00342153, -0.00183141, -0.00076345]])

kernel7 = np.array([[0,  0,  0,  0, -1,  0,  0,  0,  0],
                    [0,  0,  0, -1, -2, -1,  0,  0,  0],
                    [0,  0, -1, -2, -3, -2, -1,  0,  0],
                    [0, -1, -2, -3, -4, -3, -2, -1,  0],
                    [-1, -2, -3, -4, 81, -4, -3, -2, -1],
                    [0, -1, -2, -3, -4, -3, -2, -1,  0],
                    [0,  0, -1, -2, -3, -2, -1,  0,  0],
                    [0,  0,  0, -1, -2, -1,  0,  0,  0],
                    [0,  0,  0,  0, -1,  0,  0,  0,  0]])
kernel8 = np.array([[-1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.],
                    [-1.,  -2.,  -2.,  -2.,  -2.,  -2.,  -2.,  -2.,  -1.],
                    [-1.,  -2.,  -3.,  -3.,  -3.,  -3.,  -3.,  -2.,  -1.],
                    [-1.,  -2.,  -3.,  -4.,  -4.,  -4.,  -3.,  -2.,  -1.],
                    [-1.,  -2.,  -3.,  -4., 161.,  -4.,  -3.,  -2.,  -1.],
                    [-1.,  -2.,  -3.,  -4.,  -4.,  -4.,  -3.,  -2.,  -1.],
                    [-1.,  -2.,  -3.,  -3.,  -3.,  -3.,  -3.,  -2.,  -1.],
                    [-1.,  -2.,  -2.,  -2.,  -2.,  -2.,  -2.,  -2.,  -1.],
                    [-1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.]])
kernel9 = np.array([[0.,  0.,  0.,  0., -0.12310563,  0.,  0.,  0.,  0.],
                    [0.,  0., -0.51755435, -0.96082797, -
                        1.12310563, -0.96082797, -0.51755435,  0.,  0.],
                    [0., -0.51755435, -1.2946785, -1.88703765, -
                        2.12310563, -1.88703765, -1.2946785, -0.51755435,  0.],
                    [0., -0.96082797, -1.88703765, -2.70889206, -
                        3.12310563, -2.70889206, -1.88703765, -0.96082797,  0.],
                    [-0.12310563, -1.12310563, -2.12310563, -3.12310563,
                        69.90733198, -3.12310563, -2.12310563, -1.12310563, -0.12310563],
                    [0., -0.96082797, -1.88703765, -2.70889206, -
                        3.12310563, -2.70889206, -1.88703765, -0.96082797,  0.],
                    [0., -0.51755435, -1.2946785, -1.88703765, -
                        2.12310563, -1.88703765, -1.2946785, -0.51755435,  0.],
                    [0.,  0., -0.51755435, -0.96082797, -
                        1.12310563, -0.96082797, -0.51755435,  0.,  0.],
                    [0.,  0.,  0.,  0., -0.12310563,  0.,  0.,  0.,  0.]])

kernel10 = (kernel4+kernel6)/2
kernel11 = np.array([[0, -1., 0],
                    [-1., 5., -1.],
                    [0, -1., 0]])
kernel12 = np.array([[0, 0., 0],
                    [0., 1., 0.],
                    [0, 0., 0]])
# kernel12 = np.array([[0, 0, -1., 0, 0],
#                     [-1 -1., 5., -1.],
#                     [0, -1., 0]])
# p = 100
# for s in [3, 5, 7, 9, 11, 13]:
#     kernel = -np.ones((s, s))
#     kernel[s//2, s//2] = s**2
i = image_padded.shape[0]//2
fig, ax = plt.subplots()
ax.plot(image_padded[i, :], label="orig")
ax.plot(image_smooth[i, :], label="smooth")
lbl = ["", "9", "25", "49", "81", "41", "gauss",
       "linear81", "linear161", "circle", "hybrid", "std5", "std9", "I"]
for j, kernel in enumerate([kernel0, kernel1, kernel2, kernel3, kernel4, kernel5, kernel6,
                            kernel7, kernel8, kernel9, kernel10, kernel11, kernel12]):
    y2 = convolve2d(image_smooth, kernel, mode="same", boundary='symm')
    e = np.mean((y2-image_padded)**2)**.5
    print(np.max(kernel), np.sum(kernel), e, sep="\t")

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    ax1.imshow(image_padded.T, origin="lower", vmin=-2, vmax=2,
               interpolation="nearest")
    ax2.imshow(kernel.T, origin="lower", interpolation="nearest", vmax=0)
    ax3.imshow(image_smooth.T, origin="lower", vmin=-2, vmax=2,
               interpolation="nearest")
    ax4.imshow(y2.T, origin="lower", vmin=-2,
               vmax=2, interpolation="nearest")
    ax5.imshow((y2-image_padded).T, origin="lower", vmin=-2,
               vmax=2, interpolation="nearest")
    plt.title(e)
    plt.show()
    ax.plot(y2[i, :], label=lbl[j])
ax.legend()
plt.show()
