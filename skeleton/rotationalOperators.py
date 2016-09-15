import os

import numpy as np


"""
has functions of operators to flip and rotate a cube in 12 possible ways
as in reference paper
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
cube is of equal dimensions in x, y and z in this program
"""
REFERENCEARRAY = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
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
    cubeArrayFlippedFbInZ = np.copy(cubeArray)
    cubeArrayFlippedFbInZ[:] = cubeArray[::-1, :, :]
    return cubeArrayFlippedFbInZ


def _rot3D90(cubeArray=REFERENCEARRAY, rotAxis='z', k=0):
    """
    Returns a 3D array after rotating 90 degrees anticlockwise k times around rotAxis
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array, by default REFERENCEARRAY

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
            return cubeArray
        elif k == 1:  # rotate 90 degrees around y
            slice0 = cubeArray[0]
            slice1 = cubeArray[1]
            slice2 = cubeArray[2]
            a = np.array(column(slice2, 0))
            b = np.array(column(slice1, 0))
            c = np.array(column(slice0, 0))
            a1 = np.array(column(slice2, 1))
            b1 = np.array(column(slice1, 1))
            c1 = np.array(column(slice0, 1))
            a2 = np.array(column(slice2, 2))
            b2 = np.array(column(slice1, 2))
            c2 = np.array(column(slice0, 2))
            rotSlice0 = np.column_stack((a, b, c))
            rotSlice1 = np.column_stack((a1, b1, c1))
            rotSlice2 = np.column_stack((a2, b2, c2))
            rotMatrix = np.array((rotSlice0, rotSlice1, rotSlice2))
            return rotMatrix
        elif k == 2:  # rotate 270 degrees around y
            return flipLrInX(flipFbInZ(cubeArray))
        elif k == 3:  # rotate 270 degrees around y
            slice0 = cubeArray[0]
            slice1 = cubeArray[1]
            slice2 = cubeArray[2]
            a = np.array(column(slice0, 2))
            b = np.array(column(slice1, 2))
            c = np.array(column(slice2, 2))
            a1 = np.array(column(slice0, 1))
            b1 = np.array(column(slice1, 1))
            c1 = np.array(column(slice2, 1))
            a2 = np.array(column(slice0, 0))
            b2 = np.array(column(slice1, 0))
            c2 = np.array(column(slice2, 0))
            rotSlice0 = np.column_stack((a, b, c))
            rotSlice1 = np.column_stack((a1, b1, c1))
            rotSlice2 = np.column_stack((a2, b2, c2))
            rotMatrix = np.array((rotSlice0, rotSlice1, rotSlice2))
            return rotMatrix

firstSubiteration = REFERENCEARRAY.copy(order='C')  # mask outs border voxels in US
secondSubiteration = _rot3D90(_rot3D90(REFERENCEARRAY, 'y', 2), 'x', 3).copy(order='C')  # mask outs border voxels in NE
thirdSubiteration = _rot3D90(_rot3D90(REFERENCEARRAY, 'x', 1), 'z', 1).copy(order='C')  # mask outs border voxels in WD
fourthSubiteration = _rot3D90(REFERENCEARRAY, 'x', 3).copy(order='C')  # mask outs border voxels in ES
fifthSubiteration = _rot3D90(REFERENCEARRAY, 'y', 3).copy(order='C')  # mask outs border voxels in UW
sixthSubiteration = _rot3D90(_rot3D90(_rot3D90(REFERENCEARRAY, 'x', 3), 'z', 1), 'y', 1).copy(order='C')  # mask outs border voxels in ND
seventhSubiteration = _rot3D90(REFERENCEARRAY, 'x', 1).copy(order='C')  # mask outs border voxels in SW
eighthSubiteration = _rot3D90(REFERENCEARRAY, 'y', 2).copy(order='C')  # mask outs border voxels in UN
ninthSubiteration = _rot3D90(_rot3D90(REFERENCEARRAY, 'x', 3), 'z', 1).copy(order='C')  # mask outs border voxels in ED
tenthSubiteration = _rot3D90(_rot3D90(REFERENCEARRAY, 'y', 2), 'x', 1).copy(order='C')  # mask outs border voxels in NW
eleventhSubiteration = _rot3D90(REFERENCEARRAY, 'y', 1).copy(order='C')  # mask outs border voxels in UE
twelvethSubiteration = _rot3D90(REFERENCEARRAY, 'x', 2).copy(order='C')  # mask outs border voxels in SD

# List of 12 directions
DIRECTIONLIST = [firstSubiteration, secondSubiteration, thirdSubiteration, fourthSubiteration,
                 fifthSubiteration, sixthSubiteration, seventhSubiteration, eighthSubiteration,
                 ninthSubiteration, tenthSubiteration, eleventhSubiteration, twelvethSubiteration]

# Path of pre-generated lookuparray.npy
LOOKUPARRAYPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookuparray.npy')
