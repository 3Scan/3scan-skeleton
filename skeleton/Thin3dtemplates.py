import numpy as np

from pyeda.inter import exprvar, expr

"""
   Note: scipy.ndimage.rotate or numpy rotate don't
   work the same way as _rot3D90
"""


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


def _rot3D90(cubeArray, rotAxis='z', k=0):
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


def _getTempsDelexpression(listedMatrix):
    """

       initially check if the voxel is a curve end point
       if it is do not delete it.. Else proceed to compare
       voxel with different possible templates

    """

    """
       each template is implemented as a boolean expression of symbols
       where each symbol represents each one of the 26 neighbors

    """
    str1 = ''.join(str(e) for e in listedMatrix)
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = map(exprvar, str1)

    direction1 = (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & p & (d | e | f | m | n | u | v | w | g | h | i | o | q | x | y | z)) | \
                 (~(a) & ~(b) & ~(c) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & v & (r | s | t | j | k | l | m | n | u | w | o | p | q | x | y | z)) | \
                 (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & y & (m | n | u | w | o | q | x | z)) | \
                 (~(a) & ~(b) & ~(c) & ~(k) & ~(e) & ~(d & j) & ~ (l & f) & p & v) | \
                 (~(a) & ~(b) & ~(k) & ~(e) & c & v & p & ~(j & d) & (l ^ f)) | \
                 (a & v & p & ~(b) & ~(c) & ~(k) & ~(e) & ~(l & f) & (j ^ d)) | \
                 (~(a) & ~(b) & ~(k) & ~(e) & n & v & p & ~(j & d)) | \
                 (~(b) & ~(c) & ~(k) & ~(e) & m & v & p & ~(l & f)) | \
                 (~(b) & ~(k) & ~(e) & a & n & v & p & (j ^ d)) | \
                 (~(b) & ~(k) & ~(e) & c & m & v & p & (l ^ f)) | \
                 (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(d) & ~(e) & ~(g) & ~(h) & q & y) | \
                 (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(e) & ~(f) & ~(h) & ~(i) & o & y) | \
                 (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(r) & ~(s) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & w & y) | \
                 (~(a) & ~(b) & ~(c) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & ~(k) & ~(l) & ~(s) & ~(t) & u & y)
    return expr(direction1, simplify=True)


def firstSubIter(validateMatrix):
    """

    returns a decision if there is a valid borders voxels
    that can be removed in up or south

    """
    assert np.ndim(validateMatrix) == 3
    listedMatrix = list(np.reshape(validateMatrix, 27))
    del(listedMatrix[13])
    # val1 = _getTempsDelexpression(listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
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
    # val1 = _getTempsDelexpression(listedMatrix)
    # str1 = ''.join(str(e) for e in listedMatrix)
    return listedMatrix


if __name__ == '__main__':
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = map(exprvar, 'abcdefghijklmnopqrstuvwxyz')
    origin = exprvar('origin')
    validateMatrix = np.array([[[a, b, c], [d, e, f], [g, h, i]], [[j, k, l], [m, origin, n], [o, p, q]], [[r, s, t], [u, v, w], [x, y, z]]])
    usDeletiondirection, str1 = firstSubIter(validateMatrix)
    neDeletiondirection, str2 = secondSubIter(validateMatrix)
    wdDeletiondirection, str3 = thirdSubIter(validateMatrix)
    esDeletiondirection, str4 = fourthSubIter(validateMatrix)
    uwDeletiondirection, str5 = fifthSubIter(validateMatrix)
    ndDeletiondirection, str6 = sixthSubIter(validateMatrix)
    swDeletiondirection, str7 = seventhSubIter(validateMatrix)
    unDeletiondirection, str8 = eighthSubIter(validateMatrix)
    edDeletiondirection, str9 = ninthSubIter(validateMatrix)
    nwDeletiondirection, str10 = tenthSubIter(validateMatrix)
    ueDeletiondirection, str11 = eleventhSubIter(validateMatrix)
    sdDeletiondirection, str12 = twelvethSubIter(validateMatrix)
