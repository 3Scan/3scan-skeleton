import numpy as np
import time
from scipy import ndimage
from scipy.ndimage.filters import convolve
from skeleton.rotationalOperators import directionList

"""
   reference paper
   http://web.inf.u-szeged.hu/ipcg/publications/papers/PalagyiKuba_GMIP1999.pdf
"""

import os
lookUparray = np.load(os.path.join(os.path.dirname(__file__), 'lookuparray.npy'))
sElement = ndimage.generate_binary_structure(3, 1)


def _convolveImage(arr, flippedKernel):
    arr = np.ascontiguousarray(arr, dtype=np.uint64)
    result = convolve(arr, flippedKernel, mode='constant', cval=0)
    result[arr == 0] = 0
    return result


"""
each of the 12 iterations corresponds to each of the following
directions - us, ne, wd, es, uw, nd, sw, un, ed, nw, ue, sd
imported from template expressions
evaluated in advance using pyeda
https://pyeda.readthedocs.org/en/latest/expr.html
"""


def _skeletonPass(image):
    """
        each pass consists of 12 serial subiterations and finding the
        boundaries of the padded image/array
    """
    numPixelsremovedList = [] * 12
    for i in range(0, 12):
        convImage = _convolveImage(image, directionList[i])
        totalPixels, image = _applySubiter(image, convImage)
        # print("number of pixels removed in the {} direction is {}". format(i, totalPixels))
        numPixelsremovedList.append(totalPixels)
    numPixelsremoved = sum(numPixelsremovedList)
    return numPixelsremoved, image


def _applySubiter(image, convImage):
    """
       each subiteration paralleley reduces the border voxels in 12 directions
       going through each voxel and marking if it can be deleted or not in a
       different image named temp_del and finally multiply it with the original
       image to delete the voxels so marked
    """
    temp_del = np.zeros_like(image)
    temp_del[:] = lookUparray[convImage[:]]
    numpixel_removed = np.einsum('ijk->', image * temp_del, dtype=int)
    image[temp_del == 1] = 0
    return numpixel_removed, image


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
    numpixel_removed = 0
    while pass_no == 0 or numpixel_removed > 0:
        numpixel_removed, padImage = _skeletonPass(padImage)
        # print("number of pixels removed in pass {} is {}".format(pass_no, numpixel_removed))
        pass_no += 1
    print("done %i number of pixels in %f seconds" % (np.sum(image), time.time() - start_skeleton))
    return padImage[1:zOrig + 1, 1:yOrig + 1, 1:xOrig + 1]

if __name__ == '__main__':
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = getSkeletonize3D(sample)
    print("resultSkel", resultSkel)
