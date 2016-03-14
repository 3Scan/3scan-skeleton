import os
import time
import itertools
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage.filters import threshold_otsu
from collections import Counter
from skeleton.convOptimizeRV import getSkeletonize3D
from skeleton.unitwidthcurveskeletonRV import getShortestPathskeleton
import copy
from skeleton.segmentLengths import getSegmentsAndLengths


def convert(tupValList):
    # tupValList = list of centers of subsubvolumes
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
        subSubvolumecopy = copy.deepcopy(subSubvolume)
        histData, bins = np.histogram(subSubvolumecopy.flatten(), 256, [0, 256])
        sumation = np.sum(histData * bins[0:-1])
        weight = np.sum(histData)
        avg = sumation / weight
        subSubvolumecopy[subSubvolumecopy < int(avg)] = 0
        l1 = [y for y in subSubvolumecopy.ravel() if y != 0]
        t = threshold_otsu(np.array(l1))
        interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
        interpolatedIm = interpolatedIm > (0.95 * t)
        thinned = getSkeletonize3D(interpolatedIm)
        skeleton = getShortestPathskeleton(thinned)
        d1, d2, d3, t, typeGraphdict = getSegmentsAndLengths(skeleton)
        # d, di = getRadiusByPointsOnCenterline(shskel, boundaryIm)
        # dictR = {your_key: d[your_key] for your_key in d1.keys()}
        d = [str(t) + "\n", str(sum(d2.values())) + "\n", str(sum(d3.values())) + "\n", str(np.sum(interpolatedIm) / totalSize) + "\n"]
        f = open("/home/pranathi/subsubVolumestats/stats_{}_{}_{}.txt".format(i + 160, j, k), 'w')
        f.writelines(d)
        f.close()
        np.save("/home/pranathi/subsubVolumeskeletons/skeleton_{}_{}_{}.npy".format(i + 160, j, k), skeleton)
        np.save("/home/pranathi/subsubVolumethresholds/threshold_{}_{}_{}.npy".format(i + 160, j, k), interpolatedIm)


if __name__ == '__main__':
    totalSize = 135 * 135 * 135
    centers = []
    maskBrain = np.load('/media/pranathi/User Data/maskDownsampled10.npy')
    maskArtVein = np.load('/media/pranathi/User Data/maskArtVein.npy')
    iskpy = iskpx = 280; iskpz = int(0.5 + 560 * 0.7 / 5.0)
    root = '/media/pranathi/User Data/ii-5016-15-ms-brain_1920/filt/'
    formatOfFiles = 'png'
    listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfJpgs.sort()
    ilist = list(range(64, 799 - 10, iskpz))
    klist = [k for k in range(67, 8026 - 68, iskpx) if k < 1600 or (k > 2350 and k < 4800) or (k > 5520 and k < 8020)]
    it = list(itertools.product(ilist, range(67, 17480 - 68, iskpy), klist))
    listElements = list(map(list, it))
    subsubVolShape = (135, 135, 135)
    cornersOfCube = list(map(list, itertools.product((-1, 1), repeat=3)))
    cornersOfCube = [list((element[0] * 9, element[1] * 67, element[2] * 67)) for element in cornersOfCube]
    validit = [element for element, elementList in zip(it, listElements) for i in cornersOfCube if maskBrain[(int((elementList[0] + i[0]) / 10), int((elementList[1] + i[1]) / 7), int((elementList[2] + i[2]) / 7))] and maskArtVein[(int((elementList[0] + i[0]) / 10), int((elementList[1] + i[1]) / 7), int((elementList[2] + i[2]) / 7))] != 1]
    validSamp = list(set(validit))
    c = Counter(validit)
    validCenters = [element for element in c if c[element] == 8]
    for i, j, k in validCenters:
        jm = int(j / 7); km = int(k / 7)
        imf = int((i - 9) / 10)
        iml = int((i + 9) / 10)
        maskSub = maskArtVein[imf: iml + 1, jm - 9:jm + 10, km - 9: km + 10]
        if np.sum(maskSub) != 1:
            centers.append(tuple((i, j, k)))
    del listElements; del klist; del it; del validit; del validSamp
    startt = time.time()
    pool = Pool(processes=multiprocessing.cpu_count())
    poolLists = []
    for i in ilist:
        poolLists.append([element for element in centers if element[0] == i])
    pool.map(convert, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
