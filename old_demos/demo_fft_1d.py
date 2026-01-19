import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, wiener, butter, bessel
from scipy.ndimage import gaussian_filter
# from smooth_POD_ROM.pre_processing import on_regular_grid, get_centers, _gauss_2d, add_padding, to_frequency, to_space
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fft, ifft, fftfreq
from scipy.fftpack import fftfreq
from scipy import misc
import cv2


def to_frequency(padded, shift=False):
    if shift:
        return fft(ifftshift(padded))
    else:
        return fft(padded)


def to_space(subj):
    return ifft(subj).real


def bode(fft_result):
    # fft_result = fft(signal)
    N = len(fft_result)
    frequencies = fftfreq(N)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies[:N//2], 20 * np.log10(magnitude[:N//2]), "o--")
    plt.title('Bode Plot - Magnitude')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.ylim([-80, 80])

    plt.subplot(2, 1, 2)
    plt.semilogx(frequencies[:N//2], np.degrees(phase[:N//2]), "o--")
    plt.title('Bode Plot - Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True)
    plt.ylim([-80, 80])

    plt.tight_layout()
    plt.show()


N = 200
sigma = 2
truncate = 4
x = np.linspace(-1, 1, N)
signal = np.zeros_like(x)
signal[(-.1 < x) & (x < .1)] = 1

kernel_size = N
box = np.zeros(kernel_size,)
box[kernel_size//3:-kernel_size//3] = 1
gauss = cv2.getGaussianKernel(kernel_size, sigma).ravel()
b, a = butter(4, .5, 'low', analog=False)
# np.pad()
# box, gauss, bessel, butterworth, chebyshev, elliptic

kernel = gauss
kernel = -np.ones(N)
kernel[N//2] = N

kernel_f = to_frequency(kernel, shift=True)
signal_f = to_frequency(signal)

image_smooth_f = kernel_f*signal_f
image_smooth_s = to_space(image_smooth_f)

kernel_s = ifftshift(to_space(kernel_f))
kernel_inv_s = ifftshift(to_space(1/kernel_f))

bode(signal_f)
bode(image_smooth_f)
bode(kernel_f)
bode(1/kernel_f)

# fc = 10
# fs = 32
# T=2
# fs = N/T
# t = np.arange(-1, 1, 1/fs)


fig, (ax_s, ax_f) = plt.subplots(2, 1)
ax_s.plot(np.linspace(-1, 1, kernel_size), kernel, marker="o", ls="")
# ax_s.plot(np.linspace(-1, 1, kernel_size), kernel_inv_s, marker="o", ls="")
# ax_s.plot(x, kernel_s)
ax_s.plot(x, signal, marker="o", ls="")
# ax_s.plot(x, image_smooth_s)
ax_s.set_ylim([-2, 2])

ax_f.plot(fftfreq(kernel_f.size), np.abs(kernel_f), marker="o", ls="")
# ax_f.plot(fftfreq(kernel_f.size), np.abs(1/kernel_f), marker="o", ls="")
ax_f.plot(fftfreq(signal_f.size), signal_f, marker="o")
# ax_f.plot(fftfreq(image_smooth_f.size), image_smooth_f, marker="o")
ax_f.set_ylim([-20, 20])
