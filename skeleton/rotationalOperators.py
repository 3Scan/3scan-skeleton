import os

import numpy as np


"""
has functions of operators to flip and rotate a cube in 12 possible ways
as in reference paper
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
cube is of equal dimensions in x, y and z in this program
first dimension(0) is z, second(1) is y, third(2) is x
np.rot90 is not used to Rotate an array by 90 degrees in the counter-clockwise direction.
The first two dimensions are rotated; therefore, the array must be atleast 2-D. nor
scipy.ndimage.interpolation.rotate is used because axes definitions are different in scipy
Instead a rot3D90 is written in this program to work to our needs
"""


def column(cubeArray, i):
    """
    Returns array of ith column of the cubeArray
    Parameters
    ----------
    cubeArray : numpy array
        2D or 3D numpy array

    i : integer
        ith column of the cubeArray

    Returns
    -------
    array formed from ith column of the cubeArray

    """
    assert cubeArray.ndim > 1, "number of dimensions must be greater than one, it is %i " % cubeArray.ndim
    return np.array([row[i] for row in cubeArray])


def flipLrInX(cubeArray):
    """
    Returns an array flipped left to right
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    flip a 3D cube array in X with respect to element at origin, center of the cube

    """
    assert cubeArray.ndim == 3, "number of dimensions must be 3, it is %i " % cubeArray.ndim
    cubeArrayFlippedLrInX = np.copy(cubeArray)
    cubeArrayFlippedLrInX[:] = cubeArray[:, :, ::-1]
    return cubeArrayFlippedLrInX


def flipUdInY(cubeArray):
    """
    Returns an array flipped up to down
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    flip a 3D cube array in Y with respect to element at origin, center of the cube

    """
    assert cubeArray.ndim == 3, "number of dimensions must be 3, it is %i " % cubeArray.ndim
    cubeArrayFlippedUdInY = np.copy(cubeArray)
    cubeArrayFlippedUdInY[:] = cubeArray[:, ::-1, :]
    return cubeArrayFlippedUdInY


def flipFbInZ(cubeArray):
    """
    Returns an array flipped front to back
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    flip a 3D cube array in Z with respect to element at origin, center of the cube

    """
    assert cubeArray.ndim == 3, "number of dimensions must be 3, it is %i " % cubeArray.ndim
    cubeArrayFlippedFbInZ = np.copy(cubeArray)
    cubeArrayFlippedFbInZ[:] = cubeArray[::-1, :, :]
    return cubeArrayFlippedFbInZ


def rot3D90(cubeArray, rotAxis='z', k=0):
    """
    Returns a 3D array after rotating 90 degrees anticlockwise k times around rotAxis
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    rotAxis : string
        can be z, y or x specifies which axis should the array be rotated around, by default 'z'

    k : integer
        indicates how many times to rotate, by default '0' (doesn't rotate)

    Returns
    -------
    rotcubeArray : array
        roated cubeArray of the cubeArray

    """
    assert cubeArray.ndim == 3, "number of dimensions must be 3, it is %i " % cubeArray.ndim
    k = k % 4  # modulus of k, since rotating 5 times is same as rotating once (360 degrees rotation)
    if rotAxis == 'z':
        if k == 0:  # doesn't rotate
            return cubeArray
        elif k == 1:  # rotate 90 degrees around z
            return flipFbInZ(cubeArray).swapaxes(0, 1)
        elif k == 2:  # rotate 180 degrees around z
            return flipFbInZ(cubeArray)
        elif k == 3:  # rotate 270 degrees around z
            return flipFbInZ(cubeArray.swapaxes(0, 1))
    elif rotAxis == 'x':
        if k == 0:  # doesn't rotate
            return cubeArray
        elif k == 1:  # rotate 90 degrees around x
            return flipLrInX(cubeArray).swapaxes(1, 2)
        elif k == 2:  # rotate 180 degrees around x
            return flipUdInY(flipLrInX(cubeArray))
        elif k == 3:  # rotate 270 degrees around x
            return flipLrInX(cubeArray.swapaxes(1, 2))
    elif rotAxis == 'y':
        if k == 0:  # doesn't rotate
            rotcubeArray = cubeArray
        elif k == 1:  # rotate 90 degrees around y
            ithSlice = [cubeArray[i] for i in range(3)]
            rotSlices = [np.column_stack((column(ithSlice[2], i), column(ithSlice[1], i), column(ithSlice[0], i)))
                         for i in range(3)]
            rotcubeArray = np.array((rotSlices[0], rotSlices[1], rotSlices[2]))
        elif k == 2:  # rotate 270 degrees around y
            rotcubeArray = flipLrInX(flipFbInZ(cubeArray))
        elif k == 3:  # rotate 270 degrees around y
            ithSlice = [cubeArray[i] for i in range(3)]
            rotSlices = [np.column_stack((column(ithSlice[0], i), column(ithSlice[1], i), column(ithSlice[2], i)))
                         for i in range(3)]
            rotcubeArray = np.array((rotSlices[0], rotSlices[1], rotSlices[2]))
        return rotcubeArray


def getDirectionsList(cubeArray):
    """
    Returns a list of rotated 3D arrays to change pixel to one of 12 directions
    in UN, UE, US, UW, NE, NW, ND, ES, ED, SW, SD, and WD
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    list

    """
    assert cubeArray.ndim == 3, "number of dimensions must be 3, it is %i " % cubeArray.ndim
    # mask outs border voxels in US
    firstSubiteration = cubeArray.copy(order='C')
    # mask outs border voxels in NE
    secondSubiteration = rot3D90(rot3D90(cubeArray, 'y', 2), 'x', 3).copy(order='C')
    # mask outs border voxels in WD
    thirdSubiteration = rot3D90(rot3D90(cubeArray, 'x', 1), 'z', 1).copy(order='C')
    # mask outs border voxels in ES
    fourthSubiteration = rot3D90(cubeArray, 'x', 3).copy(order='C')
    # mask outs border voxels in UW
    fifthSubiteration = rot3D90(cubeArray, 'y', 3).copy(order='C')
    # mask outs border voxels in ND
    sixthSubiteration = rot3D90(rot3D90(rot3D90(cubeArray, 'x', 3), 'z', 1), 'y', 1).copy(order='C')
    # mask outs border voxels in SW
    seventhSubiteration = rot3D90(cubeArray, 'x', 1).copy(order='C')
    # mask outs border voxels in UN
    eighthSubiteration = rot3D90(cubeArray, 'y', 2).copy(order='C')
    # mask outs border voxels in ED
    ninthSubiteration = rot3D90(rot3D90(cubeArray, 'x', 3), 'z', 1).copy(order='C')
    # mask outs border voxels in NW
    tenthSubiteration = rot3D90(rot3D90(cubeArray, 'y', 2), 'x', 1).copy(order='C')
    # mask outs border voxels in UE
    eleventhSubiteration = rot3D90(cubeArray, 'y', 1).copy(order='C')
    # mask outs border voxels in SD
    twelvethSubiteration = rot3D90(cubeArray, 'x', 2).copy(order='C')

    # List of 12 rotated configuration arrays
    DIRECTIONS_LIST = [firstSubiteration, secondSubiteration, thirdSubiteration, fourthSubiteration,
                       fifthSubiteration, sixthSubiteration, seventhSubiteration, eighthSubiteration,
                       ninthSubiteration, tenthSubiteration, eleventhSubiteration, twelvethSubiteration]
    return DIRECTIONS_LIST


REFERENCE_ARRAY = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                           [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                           [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)
DIRECTIONS_LIST = getDirectionsList(REFERENCE_ARRAY)
# List of 12 functions corresponding to transformations in 12 directions
# Path of pre-generated lookuparray.npy
LOOKUPARRAY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookuparray.npy')
