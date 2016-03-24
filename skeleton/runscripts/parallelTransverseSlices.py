import os
import time
import itertools
from multiprocessing import Pool
import numpy as np
from scipy.misc import imread
from collections import Counter


def convert(tupValList):
    # tupValList = list of centers of subsubvolumes
    i = tupValList[0][0]
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    subSubvolume = np.zeros((19, 135, 135), dtype=np.uint8)
    subList = listOfJpgs[i - 9: i + 10]
    count = 0
    for fileName in subList:
        subVolume[count][:][:] = imread(fileName)
        count += 1
    for i, j, k in tupValList:
        subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
        subSubvolume = 255 - subSubvolume
        np.save("/home/pranathi/subsubVolumegreyscale/greyscale_{}_{}_{}.npy".format(i + 160, j, k), subSubvolume)


def outOfPixBounds(nearByCoordinate, aShape):
    onbound = 1
    for index, maxVal in enumerate(aShape):
        isAtBoundary = nearByCoordinate[index] >= maxVal or nearByCoordinate[index] < 0
        if isAtBoundary:
            onbound = 0
            break
        else:
            continue
    return onbound


if __name__ == '__main__':
    totalSize = 2460375.0
    centers = []
    maskBrain = np.load('/home/pranathi/maskDownsampled10.npy')
    maskArtVein = np.load('/home/pranathi/maskArtVein.npy')
    iskpy = iskpx = 280; iskpz = 10
    root = '/home/pranathi/ii-5016-15-ms-brain_1920/filt/'
    formatOfFiles = 'png'
    listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfJpgs.sort()
    ilist = list(range(60, 799 - 10, iskpz))
    klist = [3147]
    it = list(itertools.product(ilist, range(67, 17480 - 68, iskpy), klist))
    listElements = list(map(list, it))
    subsubVolShape = (135, 135, 135)
    cornersOfCube = list(map(list, itertools.product((-1, 1), repeat=3)))
    aShape = (2497, 1147)
    cornersOfCube = [list((element[0] * 9, element[1] * 130, element[2] * 130)) for element in cornersOfCube]
    validit = [element for element, elementList in zip(it, listElements) for i in cornersOfCube if outOfPixBounds((int((elementList[1] + i[1]) / 7), int((elementList[2] + i[2]) / 7)), aShape) and maskBrain[(int((elementList[0] + i[0]) / 10), int((elementList[1] + i[1]) / 7), int((elementList[2] + i[2]) / 7))] and maskArtVein[(int((elementList[0] + i[0]) / 10), int((elementList[1] + i[1]) / 7), int((elementList[2] + i[2]) / 7))] != 1]
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
    del listElements; del klist; del it; del validit; del validSamp; del maskBrain; del maskArtVein;
    startt = time.time()
    pool = Pool(12)
    poolLists = []
    for i in ilist:
        poolLists.append([element for element in centers if element[0] == i])
    pool.map(convert, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
