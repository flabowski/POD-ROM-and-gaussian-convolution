from unittest import TestCase, main
import numpy as np
from smooth_POD_ROM.pre_processing import on_regular_grid, get_centers


class TestDataImport(TestCase):

    def test_get_centers(self):
        pts = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]])
        tri = np.array([[0, 1, 2], [1, 2, 3]])
        c = get_centers(pts, tri)
        assert np.allclose(c, [[1., 2.], [2., 2.]])

    def test_on_regular_grid(self):
        t = [-1, 0, 1]
        xx, yy = np.meshgrid(t, t)
        pts = np.c_[xx.ravel(), yy.ravel()]
        tri = np.array([[0, 1, 4], [0, 3, 4],
                        [1, 2, 5], [4, 5, 1],
                        [3, 4, 6], [4, 6, 7],
                        [4, 5, 7], [5, 7, 8],])
        data = [1.4, 1.6, 3.4, 3.6, 2.0, 2.0, 1, -1]
        xcenter, ycenter, data_on_grid = on_regular_grid(pts, tri, data,
                                                         method='linear')
        assert np.allclose(xcenter, [-0.5,  0.5])
        assert np.allclose(ycenter, [-0.5,  0.5])
        assert np.allclose(data_on_grid, [1.5, 2.0, 3.5, 0.0])


if __name__ == "__main__":
    main()
