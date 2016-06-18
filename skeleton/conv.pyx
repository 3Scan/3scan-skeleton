import numpy as np
cimport cython  # NOQA


@cython.boundscheck(False)
def cy_convolve(unsigned long int[:, :, :] im, unsigned long int[:, :, :] kernel, Py_ssize_t[:, ::1] points):
    cdef Py_ssize_t i, j, y, z, x, n, ks = kernel.shape[0]
    cdef Py_ssize_t npoints = points.shape[0]
    cdef double[::1] responses = np.zeros(npoints, dtype='f8')

    for n in range(npoints):
        x = points[n, 1]
        y = points[n, 0]
        z = points[n, 2]
        for i in range(ks):
            for j in range(ks):
                for k in range(ks):
                    responses[n] += im[y + i, x + j, z + k] * kernel[i, j, k]

    return np.asarray(responses)
