from unittest import TestCase, main
import numpy as np
from smooth_POD_ROM.pre_processing import on_regular_grid, get_centers, _gauss_2d
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener, convolve2d
from skimage import restoration
from skimage.restoration import richardson_lucy
import tvtk
import cv2
import matplotlib.pyplot as plt


sigma = 4
truncate = 2
x, y = np.linspace(0, 1, 100)[:, None], np.linspace(0, 1, 50)[None]
data = np.sin(x) * np.cos(y)  # data_on_grid
# data[data < .5] = 0
# data[data > .5] = 1
data[x < 1-y] = 1
data[x > 1-y] = 0

psf = _gauss_2d(sigma, truncate=truncate)
data_smooth = convolve2d(data, psf, boundary='fill', mode='full')
py, px = (psf.shape[0]-1)//2, (psf.shape[1]-1)//2  # padding in x and y


# fourier
data_frequency = np.fft.fftshift(np.fft.fft2(data))
smooth_data_frequency = np.fft.fftshift(np.fft.fft2(data_smooth))

rows, cols = data_smooth.shape
freq_rows = np.fft.fftshift(np.fft.fftfreq(rows))
freq_cols = np.fft.fftshift(np.fft.fftfreq(cols))
freq_rows, freq_cols = np.meshgrid(freq_rows, freq_cols)
extent = (freq_cols.min(), freq_cols.max(), freq_rows.min(), freq_rows.max())

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
cax1 = ax1.imshow(data, origin="lower", cmap='gray')
cax2 = ax2.imshow(data_smooth, origin="lower", cmap='gray')
cax1 = ax3.imshow(np.log(np.abs(data_frequency)), cmap='gray', extent=extent)
cax2 = ax4.imshow(np.log(np.abs(smooth_data_frequency)),
                  cmap='gray', extent=extent)
cbar1 = fig.colorbar(cax1, ax=ax3, label='Log-Amplitude')
cbar2 = fig.colorbar(cax2, ax=ax4, label='Log-Amplitude')
# for ax in (ax3, ax4):
#     ax.set_title('Frequenzbereich des Bildes')
#     ax.set_xlabel('Frequenz (Zeilenumfang)')
#     ax.set_ylabel('Frequenz (Spaltenumfang)')
plt.show()


asd
# richardson lucy deconvolution
error = np.zeros((50,))
for i in range(50):
    data_decon = richardson_lucy(data_smooth, psf, num_iter=i+1)
    e = data-data_decon[py:-py, px:-px]
    error[i] = np.mean(e**2)**.5
    # #################################################################
    if i+1 in [1, 2, 5, 10, 20, 50]:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, sharex=True, sharey=True)

        ax1.imshow(data.T, origin="lower")
        l, r = -px, data.T.shape[1]+px
        b, t = -py, data.T.shape[0]+py
        ax2.imshow(data_smooth.T, origin="lower", extent=(l, r, b, t))
        ax3.imshow(data_decon.T, origin="lower", extent=(l, r, b, t))
        ax4.imshow(e.T, origin="lower")
        ax1.set_xlim(l, r)
        ax1.set_ylim(b, t)
        plt.show()
plt.figure()
plt.plot(np.arange(50)+1, error, "b.")
plt.show()
