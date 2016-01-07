from scipy.misc import imread
import numpy as np
from time import localtime, strftime
import scipy
import time
import os

from convOptimize import getSkeletonize3D

"""
    takes in 2D image slices from the root directory
    converts them to binary and returns the thinnned 3d volume
    as a boolean array
"""

from skimage.filters import threshold_otsu


def convertToBinary(image, convert):
    """
       threshod an image and display, a global threshold image is in binary_global
       if convert is True,
       object is in brighter contrast and viceversa
    """
    global_thresh = threshold_otsu(image)
    if convert:
        binary_global = image > global_thresh
    else:
        binary_global = image < global_thresh
    return np.uint8(binary_global), global_thresh


if __name__ == '__main__':
    startt = time.time()
    count = 0
    root = "/home/pranathi/Downloads/twodimageslices"
    # list and sort all the files in the given greyscale
    # input root directory
    # which is to be skeletonized
    listOffiles = os.listdir(root)
    if sorted(listOffiles) == listOffiles:
        listOffiles = listOffiles
    else:
        listOffiles.sort()
    # carefully select all the files with the extension of
    # the two dimensional slice images needed and count the
    # number of slices to allocate memory for the three
    # dimensional volume
    listOfJpgs = []
    for file in listOffiles:
        if file.endswith(".jpg"):
            listOfJpgs.append(file)
            count = count + 1
    i = imread((os.path.join(root, file)))
    m, n = np.shape(i)
    inputIm = np.zeros((count, m, n), dtype=np.uint8)
    count1 = 0
    print("x, y, z dimensions are %i %i %i  " % (m, n, count))
    dimensions = (m, n, count)
    for file in listOfJpgs:
        inputIm[count1][:][:] = imread((os.path.join(root, file)))
        count1 += 1
    # smoothedIm = scipy.ndimage.filters.gaussian_filter(inputIm, [7, 0.7, 0.7])
    thresholdedIm, globalThreshold = convertToBinary(inputIm, False)
    print("skeletonizing started at")
    print(strftime("%a, %d %b %Y %H:%M:%S ", localtime()))
    print("threshold of the 3d volume is", globalThreshold)
    np.save('/home/pranathi/Downloads/mouseBrainGreyscale.npy', inputIm)
    np.save('/home/pranathi/Downloads/mouseBrainBinary.npy', thresholdedIm)
    skeletonIm = getSkeletonize3D(thresholdedIm)
    np.save('/home/pranathi/Downloads/mouseBrainSkeleton.npy', skeletonIm)
    for i in range(skeletonIm.shape[0]):
        scipy.misc.imsave('/home/pranathi/Downloads/twodskeletonslices/skeletonIm%i.png' % i, skeletonIm[i] * 255)
    print("skeletonizing ended at")
    print(strftime("%a, %d %b %Y %H:%M:%S", localtime()))
    print("\ttime taken to skeletonize the {} sized image is {}.".format(dimensions, (time.time() - startt)))
    from scipy import ndimage
    label_img1, countObjects = ndimage.measurements.label(thresholdedIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label_img2, countObjectsSkel = ndimage.measurements.label(skeletonIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert countObjects == countObjectsSkel
