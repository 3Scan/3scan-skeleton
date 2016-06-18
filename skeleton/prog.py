import numpy as np
from conv import cy_convolve


def py_convolve(im, kernel, points):
    ks = kernel.shape[0] // 2
    data = np.pad(im, ks, mode='constant', constant_values=0)
    return cy_convolve(data, kernel, points)

if __name__ == '__main__':
    penguin = np.array([[[1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]], [[0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 1, 0]]], dtype=np.uint64).copy(order='C')
    kernel = np.ones((3, 3, 3), dtype=np.uint64)
    r = py_convolve(penguin, kernel=kernel, points=np.array([[0, 0, 0]]).copy(order='C'))
