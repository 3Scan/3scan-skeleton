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


def firstSubIter(cubeArray):
    """
    Returns a list of array elements after no transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in Up South (US)
    """
    assert 1 not in np.unique(cubeArray.shape)
    listedArray = list(np.reshape(cubeArray, 27))
    del(listedArray[13])
    return listedArray


def secondSubIter(cubeArray):
    """
    Returns a list of array elements after 1st transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in North East (NE)
    """
    assert 1 not in np.unique(cubeArray.shape)
    firstTransition = rot3D90(rot3D90(cubeArray, 'y', 2), 'x', 3)
    listedArray = list(np.reshape(firstTransition, 27))
    del(listedArray[13])
    return listedArray


def thirdSubIter(cubeArray):
    """
    Returns a list of array elements after 2nd transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in West Down (WD)
    """
    assert 1 not in np.unique(cubeArray.shape)
    secondTransition = rot3D90(rot3D90(cubeArray, 'x', 1), 'z', 1)
    listedArray = list(np.reshape(secondTransition, 27))
    del(listedArray[13])
    return listedArray


def fourthSubIter(cubeArray):
    """
    Returns a list of array elements after 3rd transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in East South (ES)
    """
    assert 1 not in np.unique(cubeArray.shape)
    thirdTransition = rot3D90(cubeArray, 'x', 3)
    listedArray = list(np.reshape(thirdTransition, 27))
    del(listedArray[13])
    return listedArray


def fifthSubIter(cubeArray):
    """
    Returns a list of array elements after 4th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in Up West (UW)
    """
    assert 1 not in np.unique(cubeArray.shape)
    fourthTransition = rot3D90(cubeArray, 'y', 3)
    listedArray = list(np.reshape(fourthTransition, 27))
    del(listedArray[13])
    return listedArray


def sixthSubIter(cubeArray):
    """
    Returns a list of array elements after 5th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in North Down (ND)
    """
    assert 1 not in np.unique(cubeArray.shape)
    fifthTransition = rot3D90(rot3D90(rot3D90(cubeArray, 'x', 3), 'z', 1), 'y', 1)
    listedArray = list(np.reshape(fifthTransition, 27))
    del(listedArray[13])
    return listedArray


def seventhSubIter(cubeArray):
    """
    Returns a list of array elements after 6th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in South West (SW)
    """
    assert 1 not in np.unique(cubeArray.shape)
    sixthTransition = rot3D90(cubeArray, 'x', 1)
    listedArray = list(np.reshape(sixthTransition, 27))
    del(listedArray[13])
    return listedArray


def eighthSubIter(cubeArray):
    """
    Returns a list of array elements after 7th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in Up North (UN)
    """
    assert 1 not in np.unique(cubeArray.shape)
    seventhTransition = rot3D90(cubeArray, 'y', 2)
    listedArray = list(np.reshape(seventhTransition, 27))
    del(listedArray[13])
    return listedArray


def ninthSubIter(cubeArray):
    """
    Returns a list of array elements after 8th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in East Down (ED)
    """
    assert 1 not in np.unique(cubeArray.shape)
    eighthTransition = rot3D90(rot3D90(cubeArray, 'x', 3), 'z', 1)
    listedArray = list(np.reshape(eighthTransition, 27))
    del(listedArray[13])
    return listedArray


def tenthSubIter(cubeArray):
    """
    Returns a list of array elements after 9th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in North West (NW)
    """

    assert 1 not in np.unique(cubeArray.shape)
    ninthTransition = rot3D90(rot3D90(cubeArray, 'y', 2), 'x', 1)
    listedArray = list(np.reshape(ninthTransition, 27))
    del(listedArray[13])
    return listedArray


def eleventhSubIter(cubeArray):
    """
    Returns a list of array elements after 10th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in Up East (UE)
    """
    assert 1 not in np.unique(cubeArray.shape)
    tenthTransition = rot3D90(cubeArray, 'y', 1)
    listedArray = list(np.reshape(tenthTransition, 27))
    del(listedArray[13])
    return listedArray


def twelvethSubIter(cubeArray):
    """
    Returns a list of array elements after 11th transformation
    Parameters
    ----------
    cubeArray : numpy array
        3D numpy array

    Returns
    -------
    listedArray: list
        list of 26 elements
    returns a list of array elements after removing the element at origin
    after transforming array in South Down (SD)
    """
    assert 1 not in np.unique(cubeArray.shape)
    eleventhTransition = rot3D90(cubeArray, 'x', 2)
    listedArray = list(np.reshape(eleventhTransition, 27))
    del(listedArray[13])
    return listedArray


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
    rotcubeArray : array
        roated cubeArray of the cubeArray

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

# mask outs border voxels in US
firstSubiteration = REFERENCE_ARRAY.copy(order='C')
# mask outs border voxels in NE
secondSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'y', 2), 'x', 3).copy(order='C')
# mask outs border voxels in WD
thirdSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'x', 1), 'z', 1).copy(order='C')
# mask outs border voxels in ES
fourthSubiteration = rot3D90(REFERENCE_ARRAY, 'x', 3).copy(order='C')
# mask outs border voxels in UW
fifthSubiteration = rot3D90(REFERENCE_ARRAY, 'y', 3).copy(order='C')
# mask outs border voxels in ND
sixthSubiteration = rot3D90(rot3D90(rot3D90(REFERENCE_ARRAY, 'x', 3), 'z', 1), 'y', 1).copy(order='C')
# mask outs border voxels in SW
seventhSubiteration = rot3D90(REFERENCE_ARRAY, 'x', 1).copy(order='C')
# mask outs border voxels in UN
eighthSubiteration = rot3D90(REFERENCE_ARRAY, 'y', 2).copy(order='C')
# mask outs border voxels in ED
ninthSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'x', 3), 'z', 1).copy(order='C')
# mask outs border voxels in NW
tenthSubiteration = rot3D90(rot3D90(REFERENCE_ARRAY, 'y', 2), 'x', 1).copy(order='C')
# mask outs border voxels in UE
eleventhSubiteration = rot3D90(REFERENCE_ARRAY, 'y', 1).copy(order='C')
# mask outs border voxels in SD
twelvethSubiteration = rot3D90(REFERENCE_ARRAY, 'x', 2).copy(order='C')

# List of 12 rotated configuration arrays
DIRECTIONS_LIST = [firstSubiteration, secondSubiteration, thirdSubiteration, fourthSubiteration,
                   fifthSubiteration, sixthSubiteration, seventhSubiteration, eighthSubiteration,
                   ninthSubiteration, tenthSubiteration, eleventhSubiteration, twelvethSubiteration]

# List of 12 functions corresponding to transformations in 12 directions
TRANSFORMATIONS_LIST = [firstSubIter, secondSubIter, thirdSubIter, fourthSubIter,
                        fifthSubIter, sixthSubIter, seventhSubIter, eighthSubIter,
                        ninthSubIter, tenthSubIter, eleventhSubIter, twelvethSubIter]

# Path of pre-generated lookuparray.npy
LOOKUPARRAY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookuparray.npy')
