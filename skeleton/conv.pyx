import numpy as np
cimport cython  # NOQA


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
