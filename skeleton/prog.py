import numpy as np
from conv import cy_convolve
from scipy.ndimage.filters import convolve
from runscripts.rotationalOperators import flipLrInx, flipUdIny, flipFbInz


def py_convolve(im, kernel, points):
    ks = kernel.shape[0] // 2
    data = np.pad(im, ks, mode='constant', constant_values=0)
    return cy_convolve(data, kernel, points)

if __name__ == '__main__':
    penguin = np.array([[[1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]], [[0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 1, 0]]], dtype=np.uint64).copy(order='C')
    kernel = np.array([[[0, 1, 1], [1, 0, 0], [0, 1, 0]], [[1, 0, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [1, 1, 0], [0, 0, 1]]], dtype=np.uint64).copy(order='C')
    kernelflipped = flipLrInx(flipUdIny(flipFbInz(kernel)))
    truth = py_convolve(penguin, kernel=kernelflipped, points=np.array(np.transpose(np.nonzero(penguin))).copy(order='C'))
    truth = truth.tolist()
    convImage = convolve(penguin, kernel, mode='constant')
    groundtruth = [item for item in (convImage * penguin).ravel().tolist() if item != 0]
    assert truth == groundtruth
