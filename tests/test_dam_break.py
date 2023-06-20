import smooth_POD_ROM as sROM
from smooth_POD_ROM.dam_break_ROM import Get_SnapsParam, get_polys
from unittest import TestCase, main
from turbulucid import Case
import os
import pytest


class TestSPD(TestCase):
    file = os.path.dirname(sROM.__file__) + \
        "/../../tests/data/damBreak_0010_0010_1_0.vtk"
    test_snapshot = Case(file)

    def test_load(self):
        data_dir = os.path.dirname(sROM.__file__)+"/../../tests/data/"
        snapshots, parameters, cases = Get_SnapsParam(
            data_dir, 10, 100, 10,  50, 0,  10)

    def test_get_polys(self):
        get_polys(self.test_snapshot)

    # with pytest.raises(ZeroDivisionError):
    #     a = 1/0


if __name__ == "__main__":
    main()
