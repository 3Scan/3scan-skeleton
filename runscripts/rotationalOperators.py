import numpy as np


"""
has functions of operators to flip and rotate a cube in all possible ways

"""
referenceArray = np.array([[[2 ** 0, 2 ** 1, 2 ** 2], [2 ** 3, 2 ** 4, 2 ** 5], [2 ** 6, 2 ** 7, 2 ** 8]],
                          [[2 ** 9, 2 ** 10, 2 ** 11], [2 ** 12, 0, 2 ** 13], [2 ** 14, 2 ** 15, 2 ** 16]],
                          [[2 ** 17, 2 ** 18, 2 ** 19], [2 ** 20, 2 ** 21, 2 ** 22], [2 ** 23, 2 ** 24, 2 ** 25]]], dtype=np.uint64)


def column(matrix, i):
    return [row[i] for row in matrix]


def flipLrInx(cubeArray):
    cubeArrayFlippedLrInx = np.copy(cubeArray)
    cubeArrayFlippedLrInx[:] = cubeArray[:, :, ::-1]
    return cubeArrayFlippedLrInx


def flipUdIny(cubeArray):
    cubeArrayFlippedLrIny = np.copy(cubeArray)
    cubeArrayFlippedLrIny[:] = cubeArray[:, ::-1, :]
    return cubeArrayFlippedLrIny


def flipFbInz(cubeArray):
    cubeArrayFlippedLrInz = np.copy(cubeArray)
    cubeArrayFlippedLrInz[:] = cubeArray[::-1, :, :]
    return cubeArrayFlippedLrInz

referenceArray = flipLrInx(flipUdIny(flipFbInz(referenceArray)))


def _rot3D90(cubeArray=referenceArray, rotAxis='z', k=0):
    m = np.array(cubeArray)
    k = k % 4
    if rotAxis == 'z':
        if k == 0:
            return m
        elif k == 1:
            return flipFbInz(m).swapaxes(0, 1)
        elif k == 2:
            return flipFbInz(m)
        else:
            # k == 3
            return flipFbInz(m.swapaxes(0, 1))
    elif rotAxis == 'x':
        if k == 0:
            return m
        elif k == 1:
            return flipLrInx(m).swapaxes(1, 2)
        elif k == 2:
            return flipUdIny(flipLrInx(m))
        else:
            # k == 3
            return flipLrInx(m.swapaxes(1, 2))
    elif rotAxis == 'y':
        if k == 0:
            return m
        elif k == 1:
            slice0 = m[0]
            slice1 = m[1]
            slice2 = m[2]
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
        elif k == 2:
            return flipLrInx(flipFbInz(m))
        else:
            slice0 = m[0]
            slice1 = m[1]
            slice2 = m[2]
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


def firstSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in up or south
    """
    assert np.ndim(validateMatrix) == 3
    listedMatrix = list(np.reshape(validateMatrix, 27))
    del(listedMatrix[13])
    return listedMatrix


def secondSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in north or east
    """
    assert np.ndim(validateMatrix) == 3
    firstTransition = _rot3D90(_rot3D90(validateMatrix, 'y', 2), 'x', 3)
    listedMatrix = list(np.reshape(firstTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def thirdSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in west or down
    """
    assert np.ndim(validateMatrix) == 3
    secondTransition = _rot3D90(_rot3D90(validateMatrix, 'x', 1), 'z', 1)
    listedMatrix = list(np.reshape(secondTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def fourthSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in east or south
    """
    assert np.ndim(validateMatrix) == 3
    thirdTransition = _rot3D90(validateMatrix, 'x', 3)
    listedMatrix = list(np.reshape(thirdTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def fifthSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in up or west
    """
    assert np.ndim(validateMatrix) == 3
    fourthTransition = _rot3D90(validateMatrix, 'y', 3)
    listedMatrix = list(np.reshape(fourthTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def sixthSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in north or down
    """
    assert np.ndim(validateMatrix) == 3
    fifthTransition = _rot3D90(_rot3D90(_rot3D90(validateMatrix, 'x', 3), 'z', 1), 'y', 1)
    listedMatrix = list(np.reshape(fifthTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def seventhSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in south or west
    """
    assert np.ndim(validateMatrix) == 3
    sixthTransition = _rot3D90(validateMatrix, 'x', 1)
    listedMatrix = list(np.reshape(sixthTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def eighthSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in up or north
    """
    assert np.ndim(validateMatrix) == 3
    seventhTransition = _rot3D90(validateMatrix, 'y', 2)
    listedMatrix = list(np.reshape(seventhTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def ninthSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in east or down
    """
    assert np.ndim(validateMatrix) == 3
    eighthTransition = _rot3D90(_rot3D90(validateMatrix, 'x', 3), 'z', 1)
    listedMatrix = list(np.reshape(eighthTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def tenthSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in north or west
    """
    assert np.ndim(validateMatrix) == 3
    ninthTransition = _rot3D90(_rot3D90(validateMatrix, 'y', 2), 'x', 1)
    listedMatrix = list(np.reshape(ninthTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def eleventhSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in up or east
    """
    assert np.ndim(validateMatrix) == 3
    tenthTransition = _rot3D90(validateMatrix, 'y', 1)
    listedMatrix = list(np.reshape(tenthTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


def twelvethSubIter(validateMatrix):
    """
    returns a decision if there is a valid borders voxels
    that can be removed in south or down
    """
    assert np.ndim(validateMatrix) == 3
    eleventhTransition = _rot3D90(validateMatrix, 'x', 2)
    listedMatrix = list(np.reshape(eleventhTransition, 27))
    del(listedMatrix[13])
    return listedMatrix


firstSubiteration = referenceArray
secondSubiteration = _rot3D90(_rot3D90(referenceArray, 'y', 2), 'x', 3)
thirdSubiteration = _rot3D90(_rot3D90(referenceArray, 'x', 1), 'z', 1)
fourthSubiteration = _rot3D90(referenceArray, 'x', 3)
fifthSubiteration = _rot3D90(referenceArray, 'y', 3)
sixthSubiteration = _rot3D90(_rot3D90(_rot3D90(referenceArray, 'x', 3), 'z', 1), 'y', 1)
seventhSubiteration = _rot3D90(referenceArray, 'x', 1)
eighthSubiteration = _rot3D90(referenceArray, 'y', 2)
ninthSubiteration = _rot3D90(_rot3D90(referenceArray, 'x', 3), 'z', 1)
tenthSubiteration = _rot3D90(_rot3D90(referenceArray, 'y', 2), 'x', 1)
eleventhSubiteration = _rot3D90(referenceArray, 'y', 1)
twelvethSubiteration = _rot3D90(referenceArray, 'x', 2)

directionList = [firstSubiteration, secondSubiteration, thirdSubiteration, fourthSubiteration,
                 fifthSubiteration, sixthSubiteration, seventhSubiteration, eighthSubiteration,
                 ninthSubiteration, tenthSubiteration, eleventhSubiteration, twelvethSubiteration]
