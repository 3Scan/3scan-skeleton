from scipy.misc import imread
import numpy as np

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
    from time import gmtime, strftime
    startt = time.time()
    count = 0
    root = "/Users/3scan_editing/Desktop/bstack_tif"
    for file in os.listdir(root):
        if file.endswith(".tif"):
            count = count + 1
    i = imread((os.path.join(root, file)))
    # m, n = np.shape(i)
    m = 4662 - 4150
    n = 4898 - 4386
    subVolumeKidney = np.zeros((count, m, n), dtype=np.uint8)
    count1 = 0
    print("x, y, z dimensions are %i %i %i  " % (m, n, count))
    dimensions = (m, n, count)
    globalThresholdsList = []
    for file in os.listdir(root):
        if file.endswith(".tif"):
            i = imread((os.path.join(root, file)))
            subVolumeKidney[count1][:][:] = i
            count1 += 1
    subVolumeKidney[subVolumeKidney == 255] = 1
    subVolumeKidney.astype(bool)
    import scipy
    print("skeletonizing started at")
    print(strftime("%a, %d %b %Y %H:%M:%S +4000", gmtime()))
    np.save('/Users/3scan_editing/records/scratch/KidneySubVolume.npy', subVolumeKidney)
    skeletonImKidney = getSkeletonize3D(subVolumeKidney)
    for i in range(skeletonImKidney.shape[0]):
        scipy.misc.imsave('/Users/3scan_editing/records/resultsKidney/skeletonImKidney%i.jpg' % i, skeletonImKidney[i] * 255)
    np.save('/Users/3scan_editing/records/scratch/skeleton3dKidney.npy', skeletonImKidney)
    print("skeletonizing ended at")
    print(strftime("%a, %d %b %Y %H:%M:%S +4000", gmtime()))
    print("\ttime taken to skeletonize the {} sized image is {}.".format(dimensions, (time.time() - startt)))
    from scipy import ndimage
    label_img1, countObjects = ndimage.measurements.label(subVolumeKidney, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label_img2, countObjectsSkel = ndimage.measurements.label(skeletonImKidney, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert countObjects == countObjectsSkel
