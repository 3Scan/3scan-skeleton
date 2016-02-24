import os
import time
import itertools
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage.filters import threshold_otsu
from skeleton.convOptimize import getSkeletonize3D
from skeleton.unitwidthcurveskeleton import getShortestPathskeleton

mask = np.load('/media/pranathi/DATA/maskDownsampled.npy')
isum = 0; iskpy = iskpx = 560; iskpz = int(0.5 + iskpx * 0.7 / 5.0)
root = '/media/pranathi/DATA/ii-5016-15-ms-brain_1920/filt/'
formatOfFiles = 'png'
f = open("/media/pranathi/DATA/StatsSubVolume.txt", 'w')
strLists = []
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()

klist = [k for k in range(67, 8026 - 68, iskpx) if k < 1600 or (k > 2350 and k < 4800) or (k > 5520 and k < 8020)]
it = list(itertools.product(range(9, 799 - 10, iskpz), range(67, 17480 - 68, iskpy), klist))
listElements = list(map(list, it))
validit = [element for element, elementList in zip(it, listElements) if mask[(int(elementList[0] / 20), int(elementList[1] / 140), int(elementList[2] / 140))]]


def convert(tupVal):
    i = tupVal[0]; j = tupVal[1]; k = tupVal[2];
    # subVolume = subVolumes[i]
    subVolume = np.zeros((298, 512, 512), dtype=np.uint8)
    subList = listOfJpgs[i - 9: i + 10]
    count = 0
    for fileName in subList:
        subVolume[count][:][:] = imread(fileName)
        count += 1
    subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
    subSubvolume = 255 - subSubvolume
    interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
    t = threshold_otsu(interpolatedIm)
    thresh = interpolatedIm > t
    thinned = getSkeletonize3D(thresh)
    skeleton = getShortestPathskeleton(thinned)
    np.save("/media/pranathi/DATA/subsubVolumeSkeletons/skeleton_{}_{}_{}.npy".format(i, j, k), skeleton)
    np.save("/media/pranathi/DATA/subsubVolumeThresholds/threshold_{}_{}_{}.npy".format(i, j, k), thresh)


def collect(i):
    print(i)
    global subVolumes
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    subList = listOfJpgs[i - 9: i + 10]
    count = 0
    for fileName in subList:
        print(fileName)
        subVolume[count][:][:] = imread(fileName)
        count += 1
    subVolumes.append(subVolume)

if __name__ == '__main__':
    startt = time.time()
    ilist = list(range(9, 799 - 10, iskpz))
    pool = Pool(processes=8)
    pool.map(collect, ilist)
    pool.map(convert, validit)
    print("time taken is %0.2f seconds", time.time() - startt)


