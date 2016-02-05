from scipy.misc import imread, imsave
import numpy as np
import time
from time import localtime, strftime
from scipy import ndimage
import os

from convOptimize import getSkeletonize3D

"""
    takes in 2D image slices from the root directory
    converts them to binary and returns the thinnned 3d volume
    as a boolean array, image slices as pngs in the root directory
    under name twodkskeletonslices
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


def getDiceSimilarityCOefficient(inputIm, thresholdedIm):
    """
        Takes an original image and its segmentation result
        gives the efficiency of a segmentation with dice similariy statistic
        defined as 2(A & B)/ A + B where A, B are the original and segmentation
        result, & intersection ("non zero voxels in common"), A + B indicates sum of non
        zero voxels in A and B
    """
    numerator = np.sum(np.logical_and(thresholdedIm, inputIm))
    denominator = len(np.transpose(np.nonzero(inputIm))) + len(np.transpose(np.nonzero(thresholdedIm)))
    dsc = numerator / denominator
    return dsc

if __name__ == '__main__':
    startt = time.time()
    count = 0
    root = input("please enter a root directory where your 2D slices are----")
    formatOfFiles = input("please enter the format of 2D files---")
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
        if file.endswith(formatOfFiles):
            listOfJpgs.append(file)
            count = count + 1
    i = imread((os.path.join(root, listOffiles[0])))
    m, n = np.shape(i)
    inputIm = np.zeros((count, m, n), dtype=np.uint8)
    count1 = 0
    print("x, y, z dimensions are %i %i %i  " % (m, n, count))
    dimensions = (m, n, count)
    for file in listOfJpgs:
        inputIm[count1][:][:] = imread((os.path.join(root, file)))
        count1 += 1
    inputIm = ndimage.interpolation.zoom(inputIm, zoom=[1, 0.6, 0.7], order=0)
    inputIm = ndimage.filters.gaussian_filter(inputIm, sigma=1)
    thresholdedIm, globalThreshold = convertToBinary(inputIm, False)
    print("skeletonizing started at")
    print(strftime("%a, %d %b %Y %H:%M:%S ", localtime()))
    print("threshold of the 3d volume is", globalThreshold)
    os.mkdir(root + 'twodkskeletonslices')
    np.save(root + 'twodkskeletonslices/' + 'Greyscale.npy', inputIm)
    np.save(root + 'twodkskeletonslices/' + 'Binary.npy', thresholdedIm)
    skeletonIm = getSkeletonize3D(thresholdedIm)
    np.save(root + 'Skeleton.npy', skeletonIm)
    for i in range(skeletonIm.shape[0]):
        imsave(root + 'twodkskeletonslices/' + 'skeletonIm%i.png' % i, skeletonIm[i] * 255)
    print("skeletonizing ended at")
    print(strftime("%a, %d %b %Y %H:%M:%S", localtime()))
    print("\ttime taken to skeletonize the {} sized image is {}.".format(dimensions, (time.time() - startt)))
    label_img1, countObjects = ndimage.measurements.label(thresholdedIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label_img2, countObjectsSkel = ndimage.measurements.label(skeletonIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert countObjects == countObjectsSkel
