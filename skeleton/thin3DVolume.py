import numpy as np
import time
from scipy import ndimage
from scipy.ndimage.filters import convolve
from runscripts.rotationalOperators import directionList

"""
   reference paper
   http://web.inf.u-szeged.hu/ipcg/publications/papers/PalagyiKuba_GMIP1999.pdf
"""

import os
lookUparray = np.load(os.path.join(os.path.dirname(__file__), 'lookuparray.npy'))

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
    function to skeletonize a 3D binary image with object in brighter contrast than background.
    In other words, 1 = object, 0 = background
    """
    assert np.max(image) in [0, 1]
    start_skeleton = time.time()
    numPixelsremoved = 1
    iterCount = 0
    label_img1, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    while numPixelsremoved > 0:
        iterTime = time.time()
        pixBefore = image.sum()
        for i in range(0, 12):
            convImage = convolve(np.uint64(image), directionList[i], mode='constant')
            convImage[image == 0] = 0
            image[lookUparray[convImage[:]] == 1] = 0
        numPixelsremoved = pixBefore - image.sum()
        print("Finished iteration {}, {} s, removed {} pixels".format(iterCount, time.time() - iterTime, numPixelsremoved))
        iterCount += 1
    print("done %i number of pixels in %0.2f seconds (%i iterations)" % (np.sum(image), time.time() - start_skeleton, iterCount))
    label_img2, countObjectsSkel = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert countObjects == countObjectsSkel
    return image


def main():
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = getThinned3D(sample)
    print("resultSkeleton", resultSkel)


if __name__ == '__main__':
    main()
