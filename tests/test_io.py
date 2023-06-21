from unittest import TestCase, main
import smooth_POD_ROM as sROM
from smooth_POD_ROM.io import get_data, get_field
import os


class TestDataImport(TestCase):
    file = os.path.dirname(sROM.__file__) + \
        "/../../tests/data/damBreak_0010_0010_1_0.vtk"

    def test_get_data(self):
        points, triangles, point_data_dict = get_data(self.file)
        assert points.shape == (9281, 3), "expected 9281 points in test file"
        assert triangles.shape == (18144, 3), "expected 18144 triangles"
        for key in ['p_rgh', 'alpha.water', 'U']:
            assert key in point_data_dict.keys(), "could not load "+key

    def test_get_field(self):
        field_data = get_field(self.file, n='alpha.water')
        assert field_data.shape == (9281,), "expected 9281 values"


if __name__ == "__main__":
    main()
