import numpy as np
# import time
# from scipy import ndimage
from scipy.ndimage.filters import convolve
from skeleton.rotationalOperators import directionList

"""
   reference paper
   http://web.inf.u-szeged.hu/ipcg/publications/papers/PalagyiKuba_GMIP1999.pdf
   same program as convOptimize.py assertions removed
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
    zOrig, yOrig, xOrig = np.shape(image)
    padImage = np.lib.pad(image, 1, 'constant', constant_values=0)
    numPixelsremoved = 1
    while numPixelsremoved > 0:
        pixBefore = padImage.sum()
        for i in range(0, 12):
            convImage = convolve(np.uint64(padImage), directionList[i], mode='constant', cval=0)
            convImage[padImage == 0] = 0
            padImage[lookUparray[convImage[:]] == 1] = 0
        numPixelsremoved = pixBefore - padImage.sum()
    return padImage[1:zOrig + 1, 1:yOrig + 1, 1:xOrig + 1]


def main():
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = getSkeletonize3D(sample)
    print("resultSkel", resultSkel)


if __name__ == '__main__':
    main()
