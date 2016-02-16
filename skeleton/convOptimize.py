import numpy as np
import time
from scipy.ndimage.filters import convolve
from skeleton.rotationalOperators import directionList

"""
   reference paper
   http://web.inf.u-szeged.hu/ipcg/publications/papers/PalagyiKuba_GMIP1999.pdf
"""

import os
lookUparray = np.load(os.path.join(os.path.dirname(__file__), 'lookuparray.npy'))

"""
each of the 12 iterations corresponds to each of the following
directions - us, ne, wd, es, uw, nd, sw, un, ed, nw, ue, sd
imported from template expressions
evaluated in advance using pyeda
https://pyeda.readthedocs.org/en/latest/expr.html
"""


def getSkeletonize3D(image):
    """
    function to skeletonize a 3D binary image with object in brighter contrast than background.
    In other words, 1 = object, 0 = background
    """
    assert np.max(image) in [0, 1]
    zOrig, yOrig, xOrig = np.shape(image)
    padImage = np.lib.pad(image, 1, 'constant', constant_values=0)
    start_skeleton = time.time()
    pass_no = 0
    numPixelsremoved = 0
    while pass_no == 0 or numPixelsremoved > 0:
        for i in range(0, 12):
            convImage = convolve(np.uint64(padImage), directionList[i], mode='constant', cval=0)
            pixBefore = padImage.sum()
            padImage[lookUparray[convImage[:]] == 1] = 0
            numPixelsremoved = pixBefore - padImage.sum()
            # print("number of pixels removed in the {} direction is {}". format(i, numPixelsremoved))
            numPixelsremoved += numPixelsremoved
            # print("number of pixels removed in pass {} is {}".format(pass_no, numpixel_removed))
        pass_no += 1
    print("done %i number of pixels in %0.2f seconds" % (np.sum(image), time.time() - start_skeleton))
    return np.uint8(padImage[1:zOrig + 1, 1:yOrig + 1, 1:xOrig + 1])


def main():
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = getSkeletonize3D(sample)
    print("resultSkel", resultSkel)


if __name__ == '__main__':
    main()
