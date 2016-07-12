import numpy as np
import pyximport; pyximport.install() # NOQA
from skeleton.thinning import cy_getThinned3D # NOQA
import time
from runscripts.rotationalOperators import directionList

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


def getThinned3D(image):
    """
    function to skeletonize a 3D binary image whose z slices are the
    first dimension as [z, y, x] with object in brighter contrast than background.
    In other words, 1 = object, 0 = background
    """
    assert np.max(image) in [0, 1]
    start_skeleton = time.time()
    zOrig, yOrig, xOrig = np.shape(image)
    orig = np.pad(np.uint64(image), 1, mode='constant', constant_values=0).copy(order='C')
    data = cy_getThinned3D(orig, directionList)
    print("done %i number of pixels in %0.2f seconds" % (np.sum(orig), time.time() - start_skeleton))
    result = data[1:zOrig + 1, 1: yOrig + 1, 1: xOrig + 1]
    return result


def main():
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = getThinned3D(sample)
    print("resultSkeleton", resultSkel)


if __name__ == '__main__':
    main()
