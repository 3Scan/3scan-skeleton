import numpy as np

from skeleton.rotational_operators import get_directions_list

"""
Lookuptable is 3Scan's idea of pre-generating a look up array of length (2 ** 26)
that has all possible configurations of binary strings of length 26 representing
26 voxels around a voxel at origin in a cube as indices and values at these
indices saying if the voxel can be removed or not (as it belongs to the boundary
not the skeleton) as in reference paper
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
"""


def firstTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not j) & (not k) & (not l) & (not r) & (not s) & (not t) & p &
              (d | e | f | m | n | u | v | w | g | h | i | o | q | x | y | z))
    return result


def secondTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not d) & (not e) & (not f) & (not g) & (not h) & (not i) & v &
              (r | s | t | j | k | l | m | n | u | w | o | p | q | x | y | z))
    return result


def thirdTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not j) & (not k) & (not l) & (not r) & (not s) & (not t) & (not d) &
              (not e) & (not f) & (not g) & (not h) & (not i) & y & (m | n | u | w | o | q | x | z))
    return result


def fourthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not k) & (not e) & (not (d & j)) & (not (l & f)) & p & v)
    return result


def fifthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not k) & (not e) & c & v & p & (not (j & d)) & (l ^ f))
    return result


def sixthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = (a & v & p & (not b) & (not c) & (not k) & (not e) & (not (l & f)) & (j ^ d))
    return result


def seventhTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not k) & (not e) & n & v & p & (not (j & d)))
    return result


def eighthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not b) & (not c) & (not k) & (not e) & m & v & p & (not (l & f)))
    return result


def ninthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not b) & (not k) & (not e) & a & n & v & p & (j ^ d))
    return result


def tenthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not b) & (not k) & (not e) & c & m & v & p & (l ^ f))
    return result


def eleventhTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not j) & (not k) & (not l) & (not r) & (not s) & (not t) & (not d) &
              (not e) & (not g) & (not h) & q & y)
    return result


def twelvethTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not j) & (not k) & (not l) & (not r) & (not s) & (not t) & (not e) &
              (not f) & (not h) & (not i) & o & y)
    return result


def thirteenthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not j) & (not k) & (not r) & (not s) & (not d) & (not e) & (not f) &
              (not g) & (not h) & (not i) & w & y)
    return result


def fourteenthTemplate(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    result = ((not a) & (not b) & (not c) & (not d) & (not e) & (not f) & (not g) & (not h) & (not i) &
              (not k) & (not l) & (not s) & (not t) & u & y)
    return result


def getVoxelDeletionFlag(neighborValues, direction):
    """
    Returns a flag saying voxel should be deleted or not

    Parameters
    ----------
    neighborValues : list
        list of first order neighborhood (26 voxels) of a nonzero value at origin

    direction : array
       transformation array describing rotation of cube to remove boundary voxels in a different direction

    Returns
    -------
    shouldVoxelBeDeleted : boolean
        0 => should not be deleted
        1 => should be deleted

    """
    assert len(neighborValues) == 27
    # reshape neighborValues to a 3 x 3 x 3 cube
    neighborMatrix = np.reshape(neighborValues, (3, 3, 3))
    # transform neighborValues to direction
    neighborValues = get_directions_list(neighborMatrix)[direction]
    neighborValues = list(np.reshape(neighborValues, 27))
    del(neighborValues[13])
    # assign 26 voxels in a 2nd ordered neighborhood of a 3D voxels as 26 alphabet variables
    neighborValues = tuple(neighborValues)
    # insert aplhabetical variables into equations of templates for deleting the boundary voxel
    shouldVoxelBeDeleted = (firstTemplate(*neighborValues) |
                            secondTemplate(*neighborValues) |
                            thirdTemplate(*neighborValues) |
                            fourthTemplate(*neighborValues) |
                            fifthTemplate(*neighborValues) |
                            sixthTemplate(*neighborValues) |
                            seventhTemplate(*neighborValues) |
                            eighthTemplate(*neighborValues) |
                            ninthTemplate(*neighborValues) |
                            tenthTemplate(*neighborValues) |
                            eleventhTemplate(*neighborValues) |
                            twelvethTemplate(*neighborValues) |
                            thirteenthTemplate(*neighborValues) |
                            fourteenthTemplate(*neighborValues))
    return shouldVoxelBeDeleted


def generateLookupArray(stop=2**26, direction=0):
    """
    Returns lookuparray

    Parameters
    ----------
    stop : int
    integer describing the length of array

    direction : int
       describing nth rotation of cube to remove boundary voxels in a different direction

    Returns
    -------
    lookuparray : array
        value at an index of the array = 0 => should not be deleted
        value at an index of the array = 1 => should be deleted

    Notes
    ------
    This program is run once, and the array is saved as lookuparray.npy in
    the same folder in main function. It doesn't have to be run again unless if templates are changed

    """
    lookUparray = np.zeros(stop, dtype=bool)
    for item in range(0, stop):
        print("nth iteration", item)
        # convert the decimal number to a binary string
        neighborValues = [(item >> digit) & 0x01 for digit in range(26)]
        # if it's a single non zero voxel in the 26 neighbors
        if np.sum(neighborValues) == 1:
            lookUparray[item] = 0
        else:
            # voxel at origin/center of the cube should be nonzero, so insert
            neighborValues.insert(13, 1)
            lookUparray[item] = getVoxelDeletionFlag(neighborValues, direction)
    return lookUparray

if __name__ == '__main__':
    # generating and saving all the 12 lookuparrays
    for index in range(12):
        lookUparray = generateLookupArray(2 ** 26, index)
        np.save("lookuparray%i.npy" % (index + 1), lookUparray)
