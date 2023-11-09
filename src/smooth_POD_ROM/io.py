import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import datetime


def get_data(file):
    # f = "\\\\files.ad.ife.no/MatPro_files/oyvindj/runs/Florian/interface/workdir2/pull_speed_3.0_mmpm_50/"
    # f = "U:/POD-ROM-and-gaussian-convolution/tests/data/"
    # file = f+"result.0001.vtk"
    # file = f+"damBreak_0010_0010_1_0.vtk"
    assert os.path.isfile(file), file+" is not a file"
    if file.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file)
    else:
        # vtkPolyDataReader, vtkUnstructuredGridReader
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(file)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()

    # Check if the dataset is a vtkPolyData
    # if not isinstance(data, vtk.vtkPolyData):
    #     raise ValueError("The dataset is not a polydata.")
    # if not isinstance(data, vtk.vtkUnstructuredGrid):
    #     raise ValueError("The dataset is not a UnstructuredGrid.")

    # print("Dataset Information:")

    # Get the points as a NumPy array
    # print(f"Number of Points: {data.GetNumberOfPoints()}")
    # print(f"Number of Cells: {data.GetNumberOfCells()}")
    points_vtk = data.GetPoints()
    points = vtk_to_numpy(points_vtk.GetData())

    if isinstance(data, vtk.vtkPolyData):
        # Get the polygons (cells) as a NumPy array
        polygons_vtk = data.GetPolys()
        cell_array = vtk_to_numpy(polygons_vtk.GetData()).reshape(-1, 4)
        assert np.all(cell_array[:, 0] == 3), "expected triangles."
        triangles = cell_array[:, 1:]

    if isinstance(data, vtk.vtkUnstructuredGrid):
        n = data.GetNumberOfCells()
        cells = data.GetCells()
        cell_array = vtk_to_numpy(cells.GetData()).reshape(n, -1)
        triangles = cell_array[:, 1:]

    point_data = data.GetPointData()
    point_data_dict = {}
    # print(f"Point Data Arrays: {point_data.GetNumberOfArrays()}")
    if point_data:
        for i in range(point_data.GetNumberOfArrays()):
            field_name = point_data.GetArrayName(i)
            # print(field_name)
            pd = point_data.GetArray(field_name)
            point_data_dict[field_name] = vtk_to_numpy(pd)
    point_data.GetArray("Vel (planar)")
    # Speed Planar -> Speed__Planar
    # Vel (planar) -> Vel_planar
    # DarcyTerm
    # CELL_DATA: ElementId, DomainId

    cell_data = data.GetCellData()
    cell_data_dict = {}
    # print(f"Cell Data Arrays: {cell_data.GetNumberOfArrays()}")
    if cell_data:
        for i in range(cell_data.GetNumberOfArrays()):
            field_name = cell_data.GetArrayName(i)
            # print(field_name)
            cd = cell_data.GetArray(field_name)
            cell_data_dict[field_name] = vtk_to_numpy(cd)
    return points, triangles, point_data_dict, cell_data_dict


def get_field(file, n='alpha.water'):
    assert os.path.isfile(file), file+" is not a valid file name."
    if file.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file)
    else:
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(file)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()
    if n == "points":
        array = data.GetPoints().GetData()
    else:
        point_data = data.GetPointData()
        array = point_data.GetArray(n)
    return vtk_to_numpy(array)


if __name__ == "__main__":
    file = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/VTK_Legacy_NEW/damBreak_1000_0100_1_30.vtk"
    # Dataset Information:
    # Number of Points: 9281
    # Number of Cells: 18144
    # Point Data Arrays: 4
    # p
    # p_rgh
    # alpha.water
    # U
    # Cell Data Arrays: 5
    # cellID
    # p
    # p_rgh
    # alpha.water
    # U
    file = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/workdir2/pull_speed_0_mmpm_50/result.0000.vtk"
    # Dataset Information:
    # Number of Points: 1092
    # Point Data Arrays: 12
    # Temperature
    # Peak_Temp
    # Temp_Rate
    # Frac_Solid
    # Solid_Time
    # Heat_Flux
    # Enthalpy
    # Pressure
    # Speed
    # Fluid_Xvel
    # Fluid_Yvel
    # Fluid_Zvel
    # Cell Data Arrays: 0
    file = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk" + \
        "/Documents/data/damBreak_results/damBreak_500_75_1_292/internal.vtu"
    # Dataset Information:
    # Number of Points: 18562
    # Number of Cells: 9072
    # Point Data Arrays: 4
    # alpha.water
    # p
    # p_rgh
    # U
    # Cell Data Arrays: 4
    # alpha.water
    # p
    # p_rgh
    # U
