import os
import time
import itertools
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage.filters import threshold_otsu
# from skeleton.convOptimize import getSkeletonize3D
# from skeleton.unitwidthcurveskeleton import getShortestPathskeleton


def convert(tupValList):
    i = tupValList[0][0]
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    subList = listOfJpgs[i - 9: i + 10]
    count = 0
    for fileName in subList:
        subVolume[count][:][:] = imread(fileName)
        count += 1
    nonZeropercentage = []
    for index, i, j, k in enumerate(tupValList):
        subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
        subSubvolume = 255 - subSubvolume
        interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
        t = threshold_otsu(interpolatedIm)
        interpolatedIm = interpolatedIm > t
        nonZeropercentage[index] = np.sum(interpolatedIm) / totalSize
        # thinned = getSkeletonize3D(interpolatedIm)
        # skeleton = getShortestPathskeleton(thinned)
        # np.save("/media/pranathi/User Data/subsubVolumeSkeletons/skeleton_{}_{}_{}.npy".format(i + 160, j, k), skeleton)
        # np.save("/media/pranathi/User Data/subsubVolumeThresholds/threshold_{}_{}_{}.npy".format(i + 160, j, k), interpolatedIm)
    return nonZeropercentage


if __name__ == '__main__':
    totalSize = 135 * 135 * 135
    nonZeropercentages = []
    mask = np.load('/media/pranathi/DATA/maskDownsampled.npy')
    iskpy = iskpx = 0; iskpz = 0
    root = '/media/pranathi/DATA/ii-5016-15-ms-brain_1920/filt/'
    formatOfFiles = 'png'
    listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfJpgs.sort()
    ilist = list(range(9, 799 - 10, 19))
    klist = [k for k in range(67, 8026 - 68, iskpx)]
    it = list(itertools.product(range(9, 799 - 10, iskpz), range(67, 17480 - 68, iskpy), klist))
    listElements = list(map(list, it))
    validit = [element for element, elementList in zip(it, listElements) if mask[(int(elementList[0] / 20), int(elementList[1] / 140), int(elementList[2] / 140))]]
    # poolLists = []
    del listElements; del klist; del it; del validit
    startt = time.time()
    pool = Pool(processes=multiprocessing.cpu_count()) - 1
    for i in ilist:
        l = [element for element in validit if element[0] == i]
        nonZeropercentages.append(pool.map(convert, l))
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
