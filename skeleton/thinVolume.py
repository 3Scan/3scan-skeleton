import numpy as np
import os
import pyximport; pyximport.install() # NOQA
import time

from skeleton.thinning import cy_getThinned3D # NOQA
from skeleton.rotationalOperators import directionList

from skimage.morphology import skeletonize

"""
   reference paper
   http://web.inf.u-szeged.hu/ipcg/publications/papers/PalagyiKuba_GMIP1999.pdf
"""


"""
each of the 12 iterations corresponds to eac


h of the following
directions - us, ne, wd, es, uw, nd, sw, un, ed, nw, ue, sd
imported from template expressions
evaluated in advance using pyeda
https://pyeda.readthedocs.org/en/latest/expr.html
"""
lookUpArrayPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookuparray.npy')


def getThinned(image):
    """
    function to skeletonize a 3D binary image whose z slices are the
    first dimension as [z, y, x] with object in brighter contrast than background.
    In other words, 1 = object, 0 = background
    """
    if len(image.shape) == 2:
        return skeletonize(image)
    else:
        assert np.max(image) in [0, 1]
        start_skeleton = time.time()
        zOrig, yOrig, xOrig = np.shape(image)
        orig = np.pad(np.uint64(image), 1, mode='constant', constant_values=0).copy(order='C')
        data = cy_getThinned3D(orig, directionList, lookUpArrayPath)
        print("done %i number of pixels in %0.2f seconds" % (np.sum(orig), time.time() - start_skeleton))
        result = data[1:zOrig + 1, 1: yOrig + 1, 1: xOrig + 1]
    return result


def main():
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = getThinned(sample)
    print("resultSkeleton", resultSkel)


if __name__ == '__main__':
    main()
