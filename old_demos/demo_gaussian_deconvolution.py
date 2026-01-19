import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, wiener
from scipy.ndimage import gaussian_filter
from smooth_POD_ROM.pre_processing import on_regular_grid, get_centers, _gauss_2d, add_padding, to_frequency
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents"
# TODO: use library for NUFFT?

# Define the Gaussian kernel size and sigma (standard deviation)
sigma = 7.0
truncate = 4
mode = "constant"  # reflect, constant
case = 2


def get_test_image(case):
    if case == 0:
        m, n = 7, 7
        x, y = np.linspace(0, 1, m)[:, None], np.linspace(0, 1, n)[None]
        image = x*(1-y)  # np.sin(x) * np.cos(y)  # data_on_grid
        image[x <= 1-y] = 1
        image[x > 1-y] = 0
    elif case == 1:
        image = misc.face(gray=True).astype(float)[::2, ::2][:-1, :-1]
    elif case == 2:
        image = np.load(pth+"/tmp/data_on_grid.npy")
    return image


image = get_test_image(case)
kernel = _gauss_2d(sigma, truncate=4, size=False)

# pad_x, pad_y = image.shape
# kernel_padded = np.pad(
#     kernel, [(pad_x//2, pad_x//2), (pad_y//2, pad_y//2)], mode='constant')

# pad_x, pad_y = kernel.shape
# image_padded = np.pad(
#     image, [(pad_x//2, pad_x//2), (pad_y//2, pad_y//2)], mode=mode)

# fft_original = fft2(image_padded)
# fft_kernel = fft2(ifftshift(kernel_padded))

kernel_padded = add_padding(kernel, image.shape, mode='constant')
image_padded = add_padding(image, kernel.shape, mode=mode)

fft_kernel = to_frequency(kernel_padded, shift=True)
fft_original = to_frequency(image_padded)


# convolution in the frequency domain
img_f = fft_original * fft_kernel
# Perform inverse FFT to get the convolved image in the spatial domain
img_s = ifft2(fft_original * fft_kernel).real


# -----------------------------------------------------------------------------
inv_kernel = 1 / fft_kernel
img_reconstructed_f = img_f / inv_kernel
img_reconstructed_s = ifft2(img_reconstructed_f).real
# -----------------------------------------------------------------------------
img_smooth_f2 = to_frequency(img_s)
img_deconvolved_f = img_smooth_f2 / fft_kernel
img_reconstructed_s2 = ifft2(img_deconvolved_f).real
# -----------------------------------------------------------------------------

inv_kernel_s = ifftshift(ifft2((inv_kernel)).real)
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
ax.imshow(inv_kernel_s.T, origin="lower", interpolation="nearest")
plt.show()
asd
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.imshow(image_padded.T, origin="lower", interpolation="nearest")
ax2.imshow(img_reconstructed_s.T, origin="lower", interpolation="nearest")
ax3.imshow(img_reconstructed_s2.T, origin="lower", interpolation="nearest")
plt.show()


if False:
    # psf_f = to_frequency(psf, data_on_grid.shape)
    # data_on_grid_f = to_frequency(data_on_grid, psf.shape)
    # data_on_grid_smooth_f = data_on_grid_f * psf_f
    # data_on_grid_s = to_space(data_on_grid_smooth_f)
    # #data_decon = deconvolve_exact(smoothROMprediction, psf, psf_f)
    # data_on_grid_smooth_s2 = smoothROMprediction
    # data_on_grid_smooth_f2 = fft2(data_on_grid_smooth_s2)
    # data_on_grid_f2 = data_on_grid_smooth_f2 / psf_f
    # data_on_grid_s2 = to_space(data_on_grid_f2)
    print(np.allclose(image, data_on_grid))
    print(np.allclose(kernel, psf))
    print(np.allclose(fft_kernel, psf_f))
    print(np.allclose(img_f, data_on_grid_smooth_f))
    print(np.allclose(img_s, data_on_grid_s))
    print(np.allclose(img_s, smoothROMprediction))  # True!

    print(np.allclose(img_s, data_on_grid_smooth_s2))
    print(np.allclose(data_on_grid_smooth_f2, data_on_grid_smooth_f))  # !

    print(np.allclose(data_decon, img_reconstructed_s))  # !

    a, b = data_on_grid_smooth_f2, data_on_grid_smooth_f
    a, b = np.log(np.abs(fftshift(a))), np.log(np.abs(fftshift(b)))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax1.imshow(a.T, origin="lower", interpolation="nearest")
    ax2.imshow(b.T, origin="lower", interpolation="nearest")
    ax3.imshow((a-b).T, origin="lower", interpolation="nearest")
    plt.show()


img_gaussian_filter = gaussian_filter(
    image, sigma, truncate=truncate, mode=mode)  # (d c b a | a b c d | d c b a)
fft_gaussian_filter = fft2(img_gaussian_filter)


plt.figure()
plt.subplot(121)
plt.imshow(img_reconstructed_s, cmap='gray', interpolation="nearest")
plt.title('img_reconstructed_s')
plt.axis('off')
plt.subplot(122)
plt.imshow(np.log(np.abs(fftshift(img_reconstructed_f))), cmap='gray',
           interpolation="nearest")
plt.title('img_reconstructed_f')
plt.axis('off')
plt.show()

print(image.shape)
print()
print(image_padded.shape)
print(kernel_padded.shape)
print(img_s.shape)
# print(img_gaussian_filter.shape)
print()
print(fft_original.shape)
print(fft_kernel.shape)
print(img_f.shape)
# print(fft_gaussian_filter.shape)

plt.figure()

plt.subplot(241)
plt.imshow(image_padded, cmap='gray', interpolation="nearest")
plt.title('Original Image')
plt.axis('off')

plt.subplot(242)
plt.imshow(kernel_padded, cmap='gray', interpolation="nearest")
plt.title('Gaussian Kernel')
plt.axis('off')

plt.subplot(243)
plt.imshow(img_s, cmap='gray', interpolation="nearest")
plt.title('Convolved Image')
plt.axis('off')

plt.subplot(244)
plt.imshow(img_gaussian_filter, cmap='gray', interpolation="nearest")
plt.title('gaussian_filter convolved Image')
plt.axis('off')

# Plot the original image in the frequency domain
plt.subplot(245)
plt.imshow(np.log(np.abs(fftshift(fft_original))), cmap='gray',
           vmin=-3, vmax=3, interpolation="nearest")
plt.title('Original Image')
plt.axis('off')
# Plot the Gaussian kernel in the frequency domain
plt.subplot(246)
plt.imshow(np.log(np.abs(fftshift(fft_kernel))), cmap='gray',
           vmin=-3, vmax=3, interpolation="nearest")
plt.title('Gaussian Kernel')
plt.axis('off')
# Plot the convolved image in the frequency domain
plt.subplot(247)
plt.imshow(np.log(np.abs(fftshift(img_f))), cmap='gray',
           vmin=-3, vmax=3, interpolation="nearest")
plt.title('Convolved Image')
plt.axis('off')
# Plot the convolved image in the frequency domain
plt.subplot(248)
plt.imshow(np.log(np.abs(fftshift(fft_gaussian_filter))), cmap='gray',
           vmin=-3, vmax=3, interpolation="nearest")
plt.title('gaussian_filter convolved Image')
plt.axis('off')

plt.tight_layout()
plt.show()

img_convolve2d = convolve2d(image, kernel, mode='same', boundary='symm')
# Compare the two convolved images to ensure they are the same
difference = np.abs(img_convolve2d - img_gaussian_filter)
max_difference = np.max(difference)
print(f"Maximum difference between the two convolved images: {max_difference}")
fft_convolve2d = (fft2(img_convolve2d, s=image_padded.shape))


# Display the original and convolved images
plt.figure(figsize=(16, 12))

plt.subplot(231)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(232)
plt.imshow(img_convolve2d, cmap='gray')
plt.title('Convolved (convolve2d)')
plt.axis('off')

plt.subplot(233)
plt.imshow(img_gaussian_filter, cmap='gray')
plt.title('Convolved (gaussian_filter)')
plt.axis('off')

# Display the original and smoothed images in the frequency domain
plt.subplot(234)
plt.imshow(np.log(np.abs(fft_original)), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(235)
plt.imshow(np.log(np.abs(fft_convolve2d)), cmap='gray')
plt.title('Convolved (convolve2d) Frequency Domain')
plt.axis('off')

plt.subplot(236)
plt.imshow(np.log(np.abs(fft_gaussian_filter)), cmap='gray')
plt.title('Convolved (gaussian_filter) Frequency Domain')
plt.axis('off')

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------------
def plot_original_vs_deconvolved(original, convolved, deconvolved):
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(original, cmap='gray', origin='lower')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(convolved, cmap='gray', origin='lower')
    plt.title('Convolved Image')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(deconvolved, cmap='gray', origin='lower')
    plt.title('Deconvolved')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    return


def deconvolution_gauss(data):
    k = _gauss_2d(sigma)
    s = (k.shape[0]-1) // 2
    k[s, s] = 0.0
    k[s, s] = np.sum(-k)
    return convolve2d(data, k, boundary='fill', mode='same')


original = image
convolved = img_convolve2d
deconvolved = wiener(image, mysize=kernel.shape[0])
plot_original_vs_deconvolved(original, convolved, deconvolved)
