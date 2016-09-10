import numpy as np

from skeleton.rotationalOperators import DIRECTIONLIST

"""
Lookuptable is 3Scan's idea of pre-generating a look up array of length (2 ** 26)
that has all possible configurations of binary strings of length 26 representing
26 voxels around a voxel at origin in a cube as indices and values at these
indices saying if the voxel can be removed or not (as it belongs to the boundary
not the skeleton) as in reference paper
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
"""


def _getVoxelDeletionFlag(neighborValues, direction):
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
    # 3 x 3 x 3 cube
    neighborMatrix = np.reshape(neighborValues, (3, 3, 3))
    neighborMatrix = direction(neighborMatrix)
    neighborValues = np.ravel(neighborMatrix).tolist()
    del neighborValues[13]
    # assign 26 voxels in a 2nd ordered neighborhood of a 3D voxels as 26 alphabet variables
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = tuple(neighborValues)
    # insert aplhabetical variables into equations of templates for deleting the boundary voxel
    shouldVoxelBeDeleted = (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & p & (d | e | f | m | n | u | v | w | g | h | i | o | q | x | y | z)) | \
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
    return shouldVoxelBeDeleted


def generateLookupArray(stop, direction):
    """
    Returns lookuparray

    Parameters
    ----------
    stop : int
    integer describing the length of array

    direction : array
       transformation array describing rotation of cube to remove boundary voxels in a different direction

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
        # convert the decimal number to a binary string
        neighborValues = [(item >> digit) & 0x01 for digit in range(26)]
        # voxel at origin/center of the cube should be nonzero, so insert
        neighborValues.insert(13, 1)
        # if it's a single non zero voxel in the cube
        if np.sum(neighborValues) == 1:
            lookUparray[item] = 0
        else:
            lookUparray[item] = _getVoxelDeletionFlag(neighborValues, direction)
    return lookUparray

if __name__ == '__main__':
    # generating and saving all the 12 lookuparrays
    for index, direction in enumerate(DIRECTIONLIST):
        lookUparray = generateLookupArray(2 ** 26, direction)
        np.save("lookuparray%i.npy" % (index + 1), lookUparray)
