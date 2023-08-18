import smooth_POD_ROM as sROM
from unittest import TestCase, main
from smooth_POD_ROM.io import get_data, get_field
import os


f = "\\\\files.ad.ife.no/MatPro_files/oyvindj/runs/Florian/interface/workdir2/pull_speed_3.0_mmpm_50/"
file = f+"result.0001.vtk"
get_data(file)

# class TestCZ(TestCase):
#     file = os.path.dirname(sROM.__file__) + \
#         "/../../tests/data/damBreak_0010_0010_1_0.vtk"
#     # test_snapshot = Case(file)

#     # def test_load(self):
#     #     data_dir = os.path.dirname(sROM.__file__)+"/../../tests/data/"
#     #     snapshots, parameters, cases = Get_SnapsParam(
#     #         data_dir, 10, 100, 10,  50, 0,  10)

#     # def test_get_polys(self):
#     #     get_polys(self.test_snapshot)

#     # with pytest.raises(ZeroDivisionError):
#     #     a = 1/0


if __name__ == "__main__":
    main()
