import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def get_data(file):

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()

    # Check if the dataset is a vtkPolyData
    if not isinstance(polydata, vtk.vtkPolyData):
        raise ValueError("The dataset is not a polydata.")

    # Print some information about the dataset
    print("Dataset Information:")
    print(f"Number of Points: {polydata.GetNumberOfPoints()}")
    print(f"Number of Cells: {polydata.GetNumberOfCells()}")
    print(f"Point Data Arrays: {polydata.GetPointData().GetNumberOfArrays()}")
    print(f"Cell Data Arrays: {polydata.GetCellData().GetNumberOfArrays()}")

    # Get the points as a NumPy array
    points_vtk = polydata.GetPoints()
    points = vtk_to_numpy(points_vtk.GetData())

    # Get the polygons (cells) as a NumPy array
    polygons_vtk = polydata.GetPolys()
    cell_array = vtk_to_numpy(polygons_vtk.GetData()).reshape(-1, 4)
    assert np.all(cell_array[:, 0] == 3), "expected triangles."
    triangles = cell_array[:, 1:]

    # Access point data and retrieve field names
    point_data = polydata.GetPointData()
    field_names = []
    if point_data:
        for i in range(point_data.GetNumberOfArrays()):
            field_names.append(point_data.GetArrayName(i))

    # Create a dictionary to store the point data
    point_data_dict = {}

    # Iterate through the field names and store the data in the dictionary
    for field_name in field_names:
        array = point_data.GetArray(field_name)
        num_tuples = array.GetNumberOfTuples()
        num_components = array.GetNumberOfComponents()
        field_data = []
        for i in range(num_tuples):
            value = array.GetTuple(i)
            if num_components == 1:
                field_data.append(value[0])
            else:
                field_data.append(list(value))
        point_data_dict[field_name] = field_data

    # no idea what this is. maybe BC's?
    # cell_data = polydata.GetCellData()
    # cell_id_array = cell_data.GetArray("cellID")
    # cell_id_np = vtk_to_numpy(cell_id_array)

    return points, triangles, point_data_dict


def get_field(file, n='alpha.water'):
    assert os.path.isfile(file), file+" is not a valid file name."
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()
    point_data = polydata.GetPointData()

    array = point_data.GetArray(n)
    num_tuples = array.GetNumberOfTuples()
    num_components = array.GetNumberOfComponents()
    field_data = []
    for i in range(num_tuples):
        value = array.GetTuple(i)
        if num_components == 1:
            field_data.append(value[0])
        else:
            field_data.append(list(value))
    return np.array(field_data)


if __name__ == "__main__":
    pass
