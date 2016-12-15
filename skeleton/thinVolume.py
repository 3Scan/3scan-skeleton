import time

import numpy as np
import pyximport; pyximport.install() # NOQA
from skimage.morphology import skeletonize

from skeleton.thinning import cy_get_thinned3D # NOQA

"""
Thinning algorithm as described in
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
z is the nth image of the stack in 3D array and is the first dimension in this program
"""


def get_thinned(binaryArr):
    """
    Return thinned output
    Parameters
    ----------
    binaryArr : Numpy array
        2D or 3D binary numpy array

    Returns
    -------
    result : boolean Numpy array
        2D or 3D binary thinned numpy array of the same shape
    """
    assert np.max(binaryArr) in [0, 1], "input must always be a binary array"
    voxCount = np.sum(binaryArr)
    if binaryArr.sum() == 0:
        return binaryArr.astype(bool)
    elif len(binaryArr.shape) == 2:
        return skeletonize(binaryArr).astype(bool)
    else:
        start_skeleton = time.time()
        zOrig, yOrig, xOrig = np.shape(binaryArr)
        orig = np.lib.pad(binaryArr, 1, 'constant')
        result = cy_get_thinned3D(np.uint64(orig))
        print("thinned %i number of pixels in %0.2f seconds" % (voxCount, time.time() - start_skeleton))
        return result[1:zOrig + 1, 1: yOrig + 1, 1: xOrig + 1].astype(bool)


if __name__ == '__main__':
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = get_thinned(sample)
