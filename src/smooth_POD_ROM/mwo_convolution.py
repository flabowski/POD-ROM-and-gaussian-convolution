import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

m, n = 7, 7
x, y = np.linspace(0, 1, m)[:, None], np.linspace(0, 1, n)[None]
image = x*y
image[x <= 1-y] = 1
image[x > 1-y] = 0

# Define the Gaussian kernel size and sigma (standard deviation)
sigma = 1.0
truncate = 2
kernel_size = 5
mode = "constant"

kernel = np.fromfunction(
    lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)
                                                   ** 2 + (y-(kernel_size-1)/2)**2) / (2*sigma**2)),
    (kernel_size, kernel_size)
)

img_gaussian_filter = gaussian_filter(
    image, sigma, truncate=truncate, mode=mode)

# Compute the 2D FFT of the original and smoothed images
pad_x, pad_y = image.shape
kernel_padded = np.pad(
    kernel, [(pad_x//2, pad_x//2), (pad_y//2, pad_y//2)], mode='constant')
pad_x, pad_y = kernel.shape
# symmetric, constant
image_padded = np.pad(
    image, [(pad_x//2, pad_x//2), (pad_y//2, pad_y//2)], mode=mode)

img_gaussian_filter = gaussian_filter(
    image_padded, sigma, truncate=truncate, mode=mode)

fft_original = (fft2(image_padded))
fft_gaussian_filter = (fft2(img_gaussian_filter))
fft_kernel = (fft2(kernel_padded))
# convolution in the frequency domain
img_f = fft_original * fft_kernel
# Perform inverse FFT to get the convolved image in the spatial domain
img_s = ifftshift(ifft2(fft_original * fft_kernel).real)

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

np.log(np.abs(fftshift(img_f))) - np.log(np.abs(fftshift(fft_gaussian_filter)))
