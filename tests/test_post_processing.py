# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:26:32 2023

@author: florianma
"""
from unittest import TestCase, main
import numpy as np
from smooth_POD_ROM.post_processing import neares_neighbour
from datetime import datetime
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class TestDataImport(TestCase):
    def test_nn2D(self):
        p1 = [.1, .2, .3]
        p2 = [10., 20.]
        P1, P2 = np.meshgrid(p1, p2)
        pts = np.c_[P1.ravel(), P2.ravel()]
        x = np.array([[.1,   10.]])
        d = neares_neighbour(x, pts)
        assert d == 0.0, "1 point as row failed"
        x = np.array([.1,   10.])
        d = neares_neighbour(x, pts)
        assert d == 0.0, "1 point flattened failed"
        x = np.array([.15,   15.])
        d = neares_neighbour(x, pts)
        assert (d - 5.00025) < 1e-6, "1 point distance miscalculated"
        x = np.array([[.1,   10],
                      [.1,   10.]])
        d = neares_neighbour(x[:, None, :], pts)
        assert np.allclose(d, np.array([0., 0.])), "several points failed"

    def test_rld(self):
        # richardson lucy deconvolution
        sigma = 4
        truncate = 2
        x, y = np.linspace(0, 1, 100)[:, None], np.linspace(0, 1, 50)[None]
        data = np.sin(x) * np.cos(y)  # data_on_grid
        data[data < .5] = 0
        data[data > .5] = 1

        # psf = _gauss_2d(sigma, truncate=truncate)
        # data_smooth2 = convolve2d(data, psf, boundary='fill', mode='full')
        # TODO:
        # py, px = (psf.shape[0]-1)//2, (psf.shape[1]-1)//2  # padding in x and y
        # error = np.zeros((50,))
        # for i in range(50):
        #     data_decon = richardson_lucy(data_smooth2, psf, num_iter=i+1)
        #     e = data-data_decon[py:-py, px:-px]
        #     error[i] = np.mean(e**2)**.5
        #     # #################################################################
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


if __name__ == "__main__":
    main()
