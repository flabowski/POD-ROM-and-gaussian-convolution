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
        pts = np.array([[-1, -1], [1, -1], [-1, 1]])
        data = [0, 2, 2]
        xx, yy, data_on_grid = on_regular_grid(pts, data, method='linear',
                                               fill_value=0.0)
        assert np.allclose(xx, [[-1, -1], [1, 1]])
        assert np.allclose(yy, [[-1, 1], [-1, 1]])
        assert np.allclose(data_on_grid, [[0., 2.], [2., 0.]])


if __name__ == "__main__":
    main()
