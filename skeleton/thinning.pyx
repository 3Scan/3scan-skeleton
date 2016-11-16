import numpy as np
cimport cython  # NOQA
from rotationalOperators import DIRECTIONS_LIST, LOOKUPARRAY_PATH
"""
cython convolve to speed up thinning
"""
lookUparray = np.load(LOOKUPARRAY_PATH)

def cy_convolve(unsigned long long int[:, :, :] binaryArr, unsigned long long int[:, :, :] kernel, Py_ssize_t[:, ::1]  points):
    """
    Returns convolved output only at points
    Parameters
    ----------
    binaryArr : Numpy array
        3D binary numpy array

    kernel : Numpy array
        3D array template of type uint64

    points : array
        array of 3D coordinates to find convolution at
    Returns
    -------
    responses : Numpy array
        3D convolved numpy array only at "points"
    """
    cdef Py_ssize_t i, j, y, z, x, n
    cdef Py_ssize_t ks = kernel.shape[0]
    cdef Py_ssize_t npoints = points.shape[0]
    cdef unsigned long long int[::1] responses = np.zeros(npoints, dtype='u8')
    for n in range(npoints):
        x = points[n, 2]
        y = points[n, 1]
        z = points[n, 0]
        for k in range(ks):
            for i in range(ks):
                for j in range(ks):
                    responses[n] += binaryArr[z + k - 1, y + i - 1, x + j - 1] * kernel[k, i, j]

    return np.asarray(responses, order='C')


def cy_getThinned3D(unsigned long long int[:, :, :] arr):
    """
    Return thinned output
    Parameters
    ----------
    binaryArr : Numpy array
        2D or 3D binary numpy array

    Returns
    -------
    Numpy array
        2D or 3D binary thinned numpy array of the same shape
    """
    assert np.max(arr) in [0, 1], "arr must be boolean"
    cdef Py_ssize_t numPixelsremoved = 1
    cdef Py_ssize_t x, y, z
    # Loop until array doesn't change equivalent to you cant remove any pixels 
    # => numPixelsremoved = 0
    while numPixelsremoved > 0:
        pixBefore = np.sum(arr)
        # loop through all 12 subiterations
        for i in range(12):
            nonzeroCoordinates = np.asarray(np.transpose(np.nonzero(arr)), order='C')
            # convolve to find config number and convolve only at points in the array "nonzeroCoordinates"
            convImage = cy_convolve(arr, kernel=DIRECTIONS_LIST[i], points=nonzeroCoordinates)
            removableIndices = (index for value, index in zip(convImage, nonzeroCoordinates) if lookUparray[value] == 1)
            for x, y, z in removableIndices:
                arr[x, y, z] = 0
        numPixelsremoved = pixBefore - np.sum(arr)
    return np.asarray(arr)