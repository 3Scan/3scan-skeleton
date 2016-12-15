import time

import numpy as np
cimport cython  # NOQA

from skeleton.rotational_operators import DIRECTIONS_LIST, LOOKUPARRAY_PATH
"""
cython convolve to speed up thinning
"""
LOOKUPARRAY = np.load(LOOKUPARRAY_PATH)

SELEMENT = np.array([[[False, False, False], [False,  True, False], [False, False, False]],
                     [[False,  True, False], [True,  False,  True], [False,  True, False]],
                     [[False, False, False], [False,  True, False], [False, False, False]]], dtype=np.uint64)


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
    arrShape = np.asarray(binaryArr).shape
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


def cy_get_thinned3D(unsigned long long int[:, :, :] arr):
    """
    Return thinned output
    Parameters
    ----------
    binaryArr : Numpy array
        3D binary numpy array

    Returns
    -------
    Numpy array
        3D binary thinned numpy array of the same shape
    """
    numPixelsremoved = 0
    cdef Py_ssize_t iterCount = 0 
    cdef Py_ssize_t x, y, z
    # Loop until array doesn't change equivalent to you cant remove any pixels => numPixelsremoved = 0
    while iterCount == 0 or numPixelsRemoved > 0:
        iterTime = time.time()
        # loop through all 12 subiterations
        nonzeroCoordinates = np.asarray(np.transpose(np.nonzero(arr)), order='C')
        borderPointArr = cy_convolve(arr, kernel=SELEMENT, points=nonzeroCoordinates)
        borderPointArrCoordinates =  np.asarray([index for value, index in zip(borderPointArr, nonzeroCoordinates) if value != 6], order='C')
        numPixelsRemoved = 0
        for i in range(12):
            convImage = cy_convolve(arr, kernel=DIRECTIONS_LIST[i], points=borderPointArrCoordinates)
            for value, (x, y, z) in zip(convImage, borderPointArrCoordinates):
                if LOOKUPARRAY[value]: 
                    arr[x, y, z] = 0
                    numPixelsRemoved += 1
        iterCount += 1
        print("Finished iteration %i, %0.2f s, removed %i pixels" % (iterCount, time.time() - iterTime, numPixelsRemoved))
    return np.asarray(arr)