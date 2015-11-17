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
    root = "/Volumes/Seagate Backup Plus Drive/z_27.22000_31.61000_offset_1_3_margins_7000_7000_0_0_block_4/bstack"
    for file in os.listdir(root):
        if file.endswith(".png"):
            count = count + 1
    i = imread((os.path.join(root, file)))
    m, n = np.shape(i)
    count = 429
    m = 3734 - 3222
    n = 4552 - 4040
    inputIm = np.zeros((count, n, m), dtype=np.uint8)
    count1 = 0
    print("x, y, z dimensions are %i %i %i  " % (m, n, count))
    for file in os.listdir(root):
        if file.endswith(".png"):
            i = imread((os.path.join(root, file)))
            inputIm[count1][:][:] = i[4040:4552, 3222:3734]
            if count1 > 50:
                break
            count1 += 1
    # thresholdedIm, globalThreshold = convertToBinary(inputIm, False)
    print("skeletonizing started at")
    print(strftime("%a, %d %b %Y %H:%M:%S +4000", gmtime()))
    # subVolume[subVolume == 255] = 1
    np.save('/Users/3scan_editing/records/scratch/newBrainSubVolume.npy', inputIm)
    skeletonIm = getSkeletonize3D(thresholdedIm)
    np.save('/Users/3scan_editing/records/scratch/skeleton3NewBrain.npy', skeletonIm)
    print("skeletonizing ended at")
    print(strftime("%a, %d %b %Y %H:%M:%S +4000", gmtime()))
    print("\ttime taken to skeletonize the {} sized image is {}.".format(dimensions, (time.time() - startt)))
    from scipy import ndimage
    label_img1, countObjects = ndimage.measurements.label(thresholdedIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label_img2, countObjectsSkel = ndimage.measurements.label(skeletonIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert countObjects == countObjectsSkel
