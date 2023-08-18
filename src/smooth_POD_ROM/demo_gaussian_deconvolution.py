import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, wiener
from scipy.ndimage import gaussian_filter
from smooth_POD_ROM.pre_processing import on_regular_grid, get_centers, _gauss_2d
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
# TODO: use library for NUFFT?


m, n = 7, 7
x, y = np.linspace(0, 1, m)[:, None], np.linspace(0, 1, n)[None]
image = x*(1-y)  # np.sin(x) * np.cos(y)  # data_on_grid
image[x <= 1-y] = 1
image[x > 1-y] = 0

# image = misc.face(gray=True).astype(float)[::2, ::2]

# Define the Gaussian kernel size and sigma (standard deviation)
sigma = 1.0
truncate = 2
mode = "reflect"
# mode = "constant"

kernel = psf = _gauss_2d(sigma, truncate=truncate)
img_gaussian_filter = gaussian_filter(
    image, sigma, truncate=truncate, mode=mode)  # (d c b a | a b c d | d c b a)


# Compute the 2D FFT of the original and smoothed images
pad_x, pad_y = image.shape
kernel_padded = np.pad(
    kernel, [(pad_x//2, pad_x//2), (pad_y//2, pad_y//2)], mode='constant')
pad_x, pad_y = kernel.shape
# symmetric, constant
image_padded = np.pad(
    image, [(pad_x//2, pad_x//2), (pad_y//2, pad_y//2)], mode=mode)

fft_original = (fft2(image_padded))
fft_gaussian_filter = (fft2(img_gaussian_filter))
fft_kernel = (fft2(kernel_padded))
# convolution in the frequency domain
img_f = fft_original * fft_kernel
# Perform inverse FFT to get the convolved image in the spatial domain
img_s = ifftshift(ifft2(fft_original * fft_kernel).real)

img_reconstructed_f = img_f / fft_kernel
img_reconstructed_s = (ifft2(img_reconstructed_f).real)
# TODO: why do we not need ifftshit here?
# TODO: why are the results from gaussian_filter and img_f not the same??


plt.figure()
plt.subplot(121)
plt.imshow(img_reconstructed_s, cmap='gray', interpolation="nearest")
plt.title('img_reconstructed_s')
plt.axis('off')
plt.subplot(122)
plt.imshow(np.log(np.abs(fftshift(img_reconstructed_f))), cmap='gray',
           vmin=-3, vmax=3, interpolation="nearest")
plt.title('img_reconstructed_f')
plt.axis('off')
plt.show()

print(image.shape)
print()
print(image_padded.shape)
print(kernel_padded.shape)
print(img_s.shape)
print(img_gaussian_filter.shape)
print()
print(fft_original.shape)
print(fft_kernel.shape)
print(img_f.shape)
print(fft_gaussian_filter.shape)

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
