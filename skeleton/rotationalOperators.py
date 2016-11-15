import os

import numpy as np


"""
has functions of operators to flip and rotate a cube in 12 possible ways
as in reference paper
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
cube is of equal dimensions in x, y and z in this program
"""
REFERENCE_ARRAY = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                           [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                           [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)


def column(matrix, i):
    """
    Returns ith column of the matrix in a list
    Parameters
    ----------
    matrix : numpy array
        2D or 3D numpy array

    i : integer
        ith column of the matrix

    Returns
    -------
    list of ith column of the matrix

    """
    return [row[i] for row in matrix]


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
    assert cubeArray.ndim == 3
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
    assert cubeArray.ndim == 3
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
    assert cubeArray.ndim == 3
    cubeArrayFlippedFbInZ = np.copy(cubeArray)
    cubeArrayFlippedFbInZ[:] = cubeArray[::-1, :, :]
    return cubeArrayFlippedFbInZ


def rot3D90(cubeArray=REFERENCE_ARRAY, rotAxis='z', k=0):
    """
    Returns a 3D array after rotating 90 degrees anticlockwise k times around rotAxis
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array, by default REFERENCE_ARRAY

    rotAxis : string
        can be z, y or x specifies which axis should the array be rotated around, by default 'z'

    k : integer
        indicates how many times to rotate, by default '0' (doesn't rotate)

    Returns
    -------
    rotMatrix : array
        roated Matrix of the cubeArray

    """
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
            rotMatrix = cubeArray
        elif k == 1:  # rotate 90 degrees around y
            ithSlice = [cubeArray[i] for i in range(3)]
            rotSlices = []
            for i in range(3):
                a = np.array(column(ithSlice[2], i))
                b = np.array(column(ithSlice[1], i))
                c = np.array(column(ithSlice[0], i))
                rotSlices[i] = np.column_stack((a, b, c))
            rotMatrix = np.array((rotSlices[0], rotSlices[1], rotSlices[2]))
        elif k == 2:  # rotate 270 degrees around y
            rotMatrix = flipLrInX(flipFbInZ(cubeArray))
        elif k == 3:  # rotate 270 degrees around y
            ithSlice = [cubeArray[i] for i in range(3)]
            rotSlices = []
            for i in range(3):
                a = np.array(column(ithSlice[0], i))
                b = np.array(column(ithSlice[1], i))
                c = np.array(column(ithSlice[2], i))
                rotSlices[i] = np.column_stack((a, b, c))
            rotMatrix = np.array((rotSlices[0], rotSlices[1], rotSlices[2]))
        return rotMatrix

firstSubiteration = REFERENCE_ARRAY.copy(order='C')  # mask outs border voxels in US
secondSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'y', 2), 'x', 3).copy(order='C')  # mask outs border voxels in NE
thirdSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'x', 1), 'z', 1).copy(order='C')  # mask outs border voxels in WD
fourthSubiteration = rot3D90(REFERENCE_ARRAY, 'x', 3).copy(order='C')  # mask outs border voxels in ES
fifthSubiteration = rot3D90(REFERENCE_ARRAY, 'y', 3).copy(order='C')  # mask outs border voxels in UW
sixthSubiteration = rot3D90(rot3D90(rot3D90(REFERENCE_ARRAY, 'x', 3), 'z', 1), 'y', 1).copy(order='C')  # mask outs border voxels in ND
seventhSubiteration = rot3D90(REFERENCE_ARRAY, 'x', 1).copy(order='C')  # mask outs border voxels in SW
eighthSubiteration = rot3D90(REFERENCE_ARRAY, 'y', 2).copy(order='C')  # mask outs border voxels in UN
ninthSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'x', 3), 'z', 1).copy(order='C')  # mask outs border voxels in ED
tenthSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'y', 2), 'x', 1).copy(order='C')  # mask outs border voxels in NW
eleventhSubiteration = rot3D90(REFERENCE_ARRAY, 'y', 1).copy(order='C')  # mask outs border voxels in UE
twelvethSubiteration = rot3D90(REFERENCE_ARRAY, 'x', 2).copy(order='C')  # mask outs border voxels in SD

# List of 12 directions
DIRECTIONLIST = [firstSubiteration, secondSubiteration, thirdSubiteration, fourthSubiteration,
                 fifthSubiteration, sixthSubiteration, seventhSubiteration, eighthSubiteration,
                 ninthSubiteration, tenthSubiteration, eleventhSubiteration, twelvethSubiteration]

# Path of pre-generated lookuparray.npy
LOOKUPARRAYPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookuparray.npy')
