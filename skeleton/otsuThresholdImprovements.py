import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import convolve
from scipy.misc import imread
from skimage.filters import threshold_otsu


"""
    assumes bright as foreground voxels
    otsu assumes a bimodal threshold, this results in bad segmentation in unimodal histograms, hence
    two improvements are proposed to improve the segmentation
"""


def avgimprovisedOtsu(subVolumeCopy):
    """
        remove voxels less than average and compute otsu for rest of the histogram
    """
    histData, bins = np.histogram(subVolumeCopy.flatten(), 256, [0, 256])
    sumation = np.sum(histData * bins[0:-1])
    weight = np.sum(histData)
    avg = sumation / weight
    subVolumeCopy[subVolumeCopy < int(avg)] = 0
    l1 = [y for y in subVolumeCopy.ravel() if y != 0]
    t = threshold_otsu(np.array(l1))
    print("avg value improvised otsu threshold", t)
    o = subVolumeCopy > t
    return o, t


def localMaximaImprovisedOtsu(subSubvolume):
    """
        remove voxels not at local maxima and compute otsu for rest of the histogram
    """
    subSubvolumeint = subSubvolume.astype(int)
    maskx = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.int64)
    masky = np.array([[[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                     [[0, -1, 0], [0, 8, 0], [0, -1, 0]],
                     [[0, -1, 0], [0, -1, 0], [0, -1, 0]]], dtype=np.int64)
    maskz = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]],
                     [[0, 0, 0], [-1, 8, -1], [0, 0, 0]],
                     [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]], dtype=np.int64)
    convolvex = convolve(subSubvolumeint, maskx)
    convolvey = convolve(subSubvolumeint, masky)
    convolvez = convolve(subSubvolumeint, maskz)
    maxi = np.maximum(convolvex, convolvey)
    maxf = np.maximum(maxi, convolvez)
    maxfi = np.ones(maxi.shape, dtype=bool)
    maxfi[maxf < 0] = 0
    subSubvolumenew = subSubvolume * maxfi
    l1 = [y for y in subSubvolumenew.ravel() if y != 0]
    newThresh = threshold_otsu(np.array(l1))
    print("local maxima improvised threshold", newThresh)
    o = subSubvolume > newThresh
    return o, newThresh


def otsuImprovements(i, j, k):
    subList = listOfJpgs[i - 9: i + 10]
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    count = 0
    for fileName in subList:
        # print(fileName)
        subVolume[count][:][:] = imread(fileName)
        count += 1
    subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
    subSubvolume = 255 - subSubvolume
    oldThresh = threshold_otsu(subSubvolume)
    maxip = np.amax(subSubvolume > oldThresh, 0)
    maxip[0, 0] = 0
    thresholdedIm, newThresh = localMaximaImprovisedOtsu(subSubvolume)
    maxip3 = np.amax(thresholdedIm, 0)
    o, t = avgimprovisedOtsu(subSubvolume)
    maxip4 = np.amax(o, 0)
    plt.subplot(2, 2, 1)
    plt.imshow(np.amax(subSubvolume, 0), cmap='gray')
    plt.title("grey scale mip")
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(maxip, cmap='gray')
    plt.title("only otsu threshold MIP, threshold is %i" % oldThresh)
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(maxip3, cmap='gray')
    plt.title("improvised otsu(1) MIP, threshold is %i" % newThresh)
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(maxip4, cmap='gray')
    plt.title("improvised otsu(2) MIP, threshold is %i" % t)
    plt.colorbar()


if __name__ == '__main__':
    root = input("please enter a path to your grey scale image files you want to threshold with otsu and compare with the improvised versions of otsu")
    formatOfFiles = input("please enter the format (extension png or jpg) for your image files")
    listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfJpgs.sort()
    coordinates = input("please enter a subvolume region you want to threshold and improvise using otsu's, integral coordinates in z followed by y and x")
    i, j, k = [int(item) for item in coordinates.split(' ')]
    otsuImprovements(i, j, k)
