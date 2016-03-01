import os
import time
import itertools
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage.filters import threshold_otsu
from skeleton.convOptimize import getSkeletonize3D
from skeleton.unitwidthcurveskeleton import getShortestPathskeleton


def convert(tupValList):
    i = tupValList[0][0]
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    subList = listOfJpgs[i - 9: i + 10]
    count = 0
    for fileName in subList:
        subVolume[count][:][:] = imread(fileName)
        count += 1
    for i, j, k in tupValList:
        subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
        subSubvolume = 255 - subSubvolume
        interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
        t = threshold_otsu(interpolatedIm)
        interpolatedIm = interpolatedIm > t
        thinned = getSkeletonize3D(interpolatedIm)
        skeleton = getShortestPathskeleton(thinned)
        np.save("/media/pranathi/User Data/subsubVolumeSkeletons/skeleton_{}_{}_{}.npy".format(i + 160, j, k), skeleton)
        np.save("/media/pranathi/User Data/subsubVolumeThresholds/threshold_{}_{}_{}.npy".format(i + 160, j, k), interpolatedIm)


if __name__ == '__main__':
    mask = np.load('/media/pranathi/User Data/maskDownsampled.npy')
    iskpy = iskpx = 280; iskpz = int(0.5 + 560 * 0.7 / 5.0)
    root = '/media/pranathi/User Data/ii-5016-15-ms-brain_1920/filt/'
    formatOfFiles = 'png'
    listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfJpgs.sort()
    ilist = list(range(9, 799 - 10, iskpz))
    klist = [k for k in range(67, 8026 - 68, iskpx) if k < 1600 or (k > 2350 and k < 4800) or (k > 5520 and k < 8020)]
    it = list(itertools.product(range(9, 799 - 10, iskpz), range(67, 17480 - 68, iskpy), klist))
    listElements = list(map(list, it))
    validit = [element for element, elementList in zip(it, listElements) if mask[(int(elementList[0] / 20), int(elementList[1] / 140), int(elementList[2] / 140))]]
    poolLists = []
    for i in ilist:
        poolLists.append([element for element in validit if element[0] == i])
    del listElements; del klist; del it; del validit
    startt = time.time()
    pool = Pool(processes=multiprocessing.cpu_count())
    pool.map(convert, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
