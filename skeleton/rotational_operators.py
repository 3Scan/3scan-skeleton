import os

import numpy as np


"""
has functions of operators to flip and rotate a cube in 12 possible ways
as in reference paper
A Parallel 3D 12-_subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
cube is of equal dimensions in x, y and z in this program
first dimension(0) is z, second(1) is y, third(2) is x
np.rot90 is not used to Rotate an array by 90 degrees in the counter-clockwise direction.
The first two dimensions are rotated; therefore, the array must be atleast 2-D. nor
scipy.ndimage.interpolation.rotate is used because axes definitions are different in scipy
Instead a rot_3D_90 is written in this program to work to our needs
"""


def _column(cube_array, nth_column):
    """
    Returns array of ith _column of the cube_array
    Parameters
    ----------
    cube_array : numpy array
        2D or 3D numpy array

    nth_column : integer
        nth column of the cube_array

    Returns
    -------
    array formed from nth column of the cube_array

    """
    return np.array([row[nth_column] for row in cube_array])


def rot_3D_90(cube_array, rot_axis='z', k=0):
    """
    Returns a 3D array after rotating 90 degrees anticlockwise k times around rot_axis
    Parameters
    ----------
    cube_array : numpy array
        3D numpy array

    rot_axis : string
        can be z, y or x specifies which axis should the array be rotated around, by default 'z'

    k : integer
        indicates how many times to rotate, by default '0' (doesn't rotate)

    Returns
    -------
    rot_cube_array : array
        roated cube_array of the cube_array

    """
    assert 1 not in np.unique(cube_array.shape)
    assert cube_array.ndim == 3, "number of dimensions must be 3, it is %i " % cube_array.ndim
    k = k % 4  # modulus of k, since rotating 5 times is same as rotating once (360 degrees rotation)
    if rot_axis == 'z':
        if k == 0:  # doesn't rotate
            return cube_array
        elif k == 1:  # rotate 90 degrees around z
            return cube_array[::-1, :, :].swapaxes(0, 1)
        elif k == 2:  # rotate 180 degrees around z
            return cube_array[::-1, :, :]
        elif k == 3:  # rotate 270 degrees around z
            return cube_array.swapaxes(0, 1)[::-1, :, :]
    elif rot_axis == 'x':
        if k == 0:  # doesn't rotate
            return cube_array
        elif k == 1:  # rotate 90 degrees around x
            return cube_array[:, :, ::-1].swapaxes(1, 2)
        elif k == 2:  # rotate 180 degrees around x
            return cube_array[:, :, ::-1][:, ::-1, :]
        elif k == 3:  # rotate 270 degrees around x
            return cube_array.swapaxes(1, 2)[:, :, ::-1]
    elif rot_axis == 'y':
        if k == 0:  # doesn't rotate
            rot_cube_array = cube_array
        elif k == 1:  # rotate 90 degrees around y
            ithSlice = [cube_array[i] for i in range(3)]
            rot_slices = [np.column_stack((_column(ithSlice[2], i), _column(ithSlice[1], i), _column(ithSlice[0], i)))
                          for i in range(3)]
            rot_cube_array = np.array((rot_slices[0], rot_slices[1], rot_slices[2]))
        elif k == 2:  # rotate 270 degrees around y
            rot_cube_array = cube_array[::-1, :, :][:, :, ::-1]
        elif k == 3:  # rotate 270 degrees around y
            ithSlice = [cube_array[i] for i in range(3)]
            rot_slices = [np.column_stack((_column(ithSlice[0], i), _column(ithSlice[1], i), _column(ithSlice[2], i)))
                          for i in range(3)]
            rot_cube_array = np.array((rot_slices[0], rot_slices[1], rot_slices[2]))
        return rot_cube_array


def get_directions_list(cube_array):
    """
    Returns a list of rotated 3D arrays to change pixel to one of 12 directions
    in UN, UE, US, UW, NE, NW, ND, ES, ED, SW, SD, and WD
    Parameters
    ----------
    cube_array : numpy array
        3D numpy array

    Returns
    -------
    list

    """
    assert cube_array.ndim == 3, "number of dimensions must be 3, it is %i " % cube_array.ndim
    # mask outs border voxels in US
    first_subiteration = cube_array.copy(order='C')
    # mask outs border voxels in NE
    second_subiteration = rot_3D_90(rot_3D_90(cube_array, 'y', 2), 'x', 3).copy(order='C')
    # mask outs border voxels in WD
    third_subiteration = rot_3D_90(rot_3D_90(cube_array, 'x', 1), 'z', 1).copy(order='C')
    # mask outs border voxels in ES
    fourth_subiteration = rot_3D_90(cube_array, 'x', 3).copy(order='C')
    # mask outs border voxels in UW
    fifth_subiteration = rot_3D_90(cube_array, 'y', 3).copy(order='C')
    # mask outs border voxels in ND
    sixth_subiteration = rot_3D_90(rot_3D_90(rot_3D_90(cube_array, 'x', 3), 'z', 1), 'y', 1).copy(order='C')
    # mask outs border voxels in SW
    seventh_subiteration = rot_3D_90(cube_array, 'x', 1).copy(order='C')
    # mask outs border voxels in UN
    eighth_subiteration = rot_3D_90(cube_array, 'y', 2).copy(order='C')
    # mask outs border voxels in ED
    ninth_subiteration = rot_3D_90(rot_3D_90(cube_array, 'x', 3), 'z', 1).copy(order='C')
    # mask outs border voxels in NW
    tenth_subiteration = rot_3D_90(rot_3D_90(cube_array, 'y', 2), 'x', 1).copy(order='C')
    # mask outs border voxels in UE
    eleventh_subiteration = rot_3D_90(cube_array, 'y', 1).copy(order='C')
    # mask outs border voxels in SD
    twelveth_subiteration = rot_3D_90(cube_array, 'x', 2).copy(order='C')

    # List of 12 rotated configuration arrays
    DIRECTIONS_LIST = [first_subiteration, second_subiteration, third_subiteration, fourth_subiteration,
                       fifth_subiteration, sixth_subiteration, seventh_subiteration, eighth_subiteration,
                       ninth_subiteration, tenth_subiteration, eleventh_subiteration, twelveth_subiteration]
    return DIRECTIONS_LIST


REFERENCE_ARRAY = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                           [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                           [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)
DIRECTIONS_LIST = get_directions_list(REFERENCE_ARRAY)
# List of 12 functions corresponding to transformations in 12 directions
# Path of pre-generated lookuparray.npy
LOOKUPARRAY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookuparray.npy')
