from unittest import TestCase, main
import numpy as np
from datetime import datetime
from smooth_POD_ROM.pre_processing import (on_regular_grid, get_centers,
                                           _gauss_2d, add_padding,
                                           remove_padding, to_frequency,
                                           to_space, convolve_f, gaussian_f)
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import wiener, convolve2d
from skimage import restoration
from skimage.restoration import richardson_lucy
import tvtk
import cv2
import matplotlib.pyplot as plt


class TestDataImport(TestCase):

    def test_padding(self):
        x, y = np.linspace(0, 1, 11)[:, None], np.linspace(0, 1, 5)[None]
        data = np.sin(x) * np.cos(y)
        data_padded = add_padding(data, (11+4, 5+4))
        data2 = remove_padding(data_padded, data.shape)
        assert np.allclose(data2, data)

    def test_trafo(self):
        x, y = np.linspace(0, 1, 11)[:, None], np.linspace(0, 1, 5)[None]
        data = np.sin(x) * np.cos(y)
        data_f = to_frequency(data)
        data2 = to_space(data_f)
        assert np.allclose(data2, data)

    def test_get_centers(self):
        pts = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]])
        tri = np.array([[0, 1, 2], [1, 2, 3]])
        c = get_centers(pts, tri)
        assert np.allclose(c, [[1., 2.], [2., 2.]])

    def test_on_regular_grid(self):
        pts = np.array([[-1, -1], [1, -1], [-1, 1]])
        data = [0, 2, 2]
        xx, yy, data_on_grid = on_regular_grid(pts, data, method='linear',
                                               fill_value=0.0)
        assert np.allclose(xx, [[-1, -1], [1, 1]])
        assert np.allclose(yy, [[-1, 1], [-1, 1]])
        assert np.allclose(data_on_grid, [[0., 2.], [2., 0.]])

    def test_gaussian_convolution(self):
        sigma = 4
        truncate = 3
        x, y = np.linspace(0, 1, 101)[:, None], np.linspace(0, 1, 51)[None]
        data = np.sin(x) * np.cos(y)  # data_on_grid
        data[data < .5] = 0
        data[data > .5] = 1

        psk = _gauss_2d(sigma, truncate=truncate)
        kernel_size = int(2*truncate * sigma) + 1
        kernel1D = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.outer(kernel1D, kernel1D)
        assert np.allclose(psk, kernel), "kernel not the same"

        t1 = datetime.now()

        data_smooth = gaussian_filter(data, sigma=sigma, truncate=truncate)

        t2 = datetime.now()

        data_smooth2 = convolve2d(data, psk, boundary='symm', mode='same')

        t3 = datetime.now()

        kernel_padded = add_padding(kernel, data.shape, mode="constant")
        image_padded = add_padding(data, kernel.shape, mode="symmetric")

        kernel_f = to_frequency(kernel_padded, shift=True)
        image_f = to_frequency(image_padded)

        image_smooth_f = kernel_f*image_f
        image_smooth = to_space(image_smooth_f)

        data_smooth3 = remove_padding(image_smooth, data.shape)

        t4 = datetime.now()
        print((t2-t1).total_seconds())
        print((t3-t2).total_seconds())
        print((t4-t3).total_seconds())
        assert np.allclose(
            data_smooth, data_smooth2), "convolve2d smoothing not the same"
        assert np.allclose(
            data_smooth, data_smooth3), "convolution in frequency domain not the same"
        # if False:
        #     fig, ((ax1, ax2, ax3), (ax12, ax22, ax32)) = plt.subplots(
        #         2, 3, sharex=True, sharey=True)
        #     ax1.imshow(data_smooth)
        #     ax2.imshow(data_smooth2)
        #     ax3.imshow(data_smooth3)
        #     ax12.imshow(data_smooth-data_smooth)
        #     ax22.imshow(data_smooth2-data_smooth)
        #     ax32.imshow(data_smooth3-data_smooth)
        #     plt.show()

    def test_rld(self):
        # richardson lucy deconvolution
        sigma = 4
        truncate = 2
        x, y = np.linspace(0, 1, 100)[:, None], np.linspace(0, 1, 50)[None]
        data = np.sin(x) * np.cos(y)  # data_on_grid
        data[data < .5] = 0
        data[data > .5] = 1

        psf = _gauss_2d(sigma, truncate=truncate)
        data_smooth2 = convolve2d(data, psf, boundary='fill', mode='full')
        py, px = (psf.shape[0]-1)//2, (psf.shape[1]-1)//2  # padding in x and y
        error = np.zeros((50,))
        for i in range(50):
            data_decon = richardson_lucy(data_smooth2, psf, num_iter=i+1)
            e = data-data_decon[py:-py, px:-px]
            error[i] = np.mean(e**2)**.5
            # #################################################################
        #     if i+1 in [1, 2, 5, 10, 20, 50]:
        #         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        #             2, 2, sharex=True, sharey=True)

        #         ax1.imshow(data.T, origin="lower")
        #         l, r = -px, data.T.shape[1]+px
        #         b, t = -py, data.T.shape[0]+py
        #         ax2.imshow(data_smooth2.T, origin="lower", extent=(l, r, b, t))
        #         ax3.imshow(data_decon.T, origin="lower", extent=(l, r, b, t))
        #         ax4.imshow(e.T, origin="lower")
        #         ax1.set_xlim(l, r)
        #         ax1.set_ylim(b, t)
        #         plt.show()
        # plt.figure()
        # plt.plot(np.arange(50)+1, error, "b.")
        # plt.show()

    def test_convolve_f(self):
        sigma = 0.05
        x = np.linspace(0, 1, 100)
        dx = x[1]
        y = np.zeros_like(x)
        y[40:60] = 1
        kernel_f = gaussian_f(x, sigma)
        y_smooth = convolve_f(y, kernel_f)
        y_smooth2 = gaussian_filter1d(
            y, sigma/dx, mode="wrap", cval=0, truncate=20)
        assert np.allclose(y_smooth, y_smooth2)
        # plt.plot(y)
        # plt.plot(y_smooth)
        # plt.plot(y_smooth2)


if __name__ == "__main__":
    main()
