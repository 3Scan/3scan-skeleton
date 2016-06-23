import numpy as np
cimport cython  # NOQA

import os
lookUparray = np.load(os.path.join(os.path.dirname(__file__), 'lookuparray.npy'))

@cython.boundscheck(False)
def cy_convolve(unsigned long long int[:, :, :] im, unsigned long long int[:, :, :] kernel, Py_ssize_t[:, ::1] points):
    cdef Py_ssize_t i, j, y, z, x, n, ks = kernel.shape[0]
    cdef Py_ssize_t npoints = points.shape[0]
    cdef unsigned long long int[::1] responses = np.zeros(npoints, dtype='u8')
    for n in range(npoints):
        x = points[n, 2]
        y = points[n, 1]
        z = points[n, 0]
        for k in range(ks):
            for i in range(ks):
                for j in range(ks):
                    responses[n] += im[z + k - 1, y + i - 1, x + j - 1] * kernel[k, i, j]

    return np.asarray(responses)


@cython.boundscheck(False)
def cy_getThinned3D(unsigned long long int[:, :, :] image, unsigned long long int[:, :, :, ::1] directionList):
    """
    function to skeletonize a 3D binary image with object in brighter contrast than background.
    In other words, 1 = object, 0 = background
    """
    assert np.max(image) in [0, 1], "image must be boolean"
    cdef Py_ssize_t numPixelsremoved = 1
    cdef Py_ssize_t x, y, z
    cdef unsigned long long int[:, :, :] orig = image
    while numPixelsremoved > 0:
        pixBefore = np.sum(image)
        for i in range(12):
            points = np.asarray(np.transpose(np.nonzero(image)), order='C')
            convImage = cy_convolve(image, kernel=directionList[i], points=points)
            indices = [index for value, index in zip(convImage.tolist(), points.tolist()) if lookUparray[value] == 1]
            for x, y, z in indices:
                image[x, y, z] = 0
            image = np.multiply(image, orig)
        numPixelsremoved = pixBefore - np.sum(image)
    return np.asarray(image)