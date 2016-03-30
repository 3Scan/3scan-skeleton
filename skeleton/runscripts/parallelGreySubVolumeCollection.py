import os
import time
import itertools
import multiprocessing
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
        np.save("/home/pranathi/subsubVolumegreyscaleNew_28/greyscale_{}_{}_{}.npy".format(i + 160, j, k), subSubvolume)


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
    # maskArtVein = np.load('/home/pranathi/maskArtVein.npy')
    # assert np.unique(maskArtVein).tolist() in [0, 1]
    iskpx = 900; iskpz = 10; iskpy = 71
    root = '/home/pranathi/ii-5016-15-ms-brain_1920/filt/'
    formatOfFiles = 'png'
    listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfJpgs.sort()
    ilist = list(range(60, 799 - 10, iskpz))
    # klist = [3667, 6367, 7267]
    klist = [2767]
    # klist = [k for k in range(67, 8026 - 68, iskpx) if (k > 2420 and k < 4000) or (k > 5500 and k < 8000)]
    it = list(itertools.product(ilist, range(67, 17480 - 68, iskpy), klist))
    listElements = list(map(list, it))
    subsubVolShape = (135, 135, 135)
    cornersOfCube = list(map(list, itertools.product((-1, 1), repeat=3)))
    aShape = (2497, 1147)
    cornersOfCube = [list((element[0] * 9, element[1] * 130, element[2] * 130)) for element in cornersOfCube]
    validit = [element for element, elementList in zip(it, listElements) for i in cornersOfCube if outOfPixBounds((int((elementList[1] + i[1] + 6) / 7.0), int((elementList[2] + i[2] + 6) / 7.0)), aShape) and maskBrain[(int((elementList[0] + i[0] + 9) / 10.0), int((elementList[1] + i[1] + 9) / 7.0), int((elementList[2] + i[2] + 6) / 7.0))]]
    c = Counter(validit)
    validCenters = [element for element in c if c[element] == 8]
    # for i, j, k in validCenters:
    #     jm = int((j + 6) / 7.0); km = int((k + 6) / 7.0)
    #     imf = int((i - 9 - 9) / 10.0)
    #     iml = int((i + 9 - 9) / 10.0)
    #     maskSub = maskArtVein[imf: iml + 1, jm - 9:jm + 10, km - 9: km + 10]
    #     if np.sum(maskSub) == 0:
    #         centers.append(tuple((i, j, k)))
    del listElements; del klist; del it; del validit; del maskBrain; del maskArtVein;
    startt = time.time()
    numProcessors = multiprocessing.cpu_count()
    Nilist = len(ilist)
    iilist = []
    valid = 0
    for k in range(0, Nilist, 10):
        iilist.append(ilist[k: k + 10])
    for index, i in enumerate(iilist):
        if index > 7:
            break
        poolLists = []
        for zplane in i:
            poolLists.append([element for element in validCenters if element[0] == zplane])
        for elem in poolLists:
            valid += len(elem)
        print(index, len(poolLists))
        pool = Pool(numProcessors)
        if poolLists == [[], [], []] or poolLists == []:
            break
        pool.map(convert, poolLists)
        pool.close()
        pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
    transverseSlice = imread("transverseSlice2767.png")
    for coords in validCenters:
        transverseSlice[coords[0] - 1: coords[0] + 2, int((coords[1] + 6) / 7) - 1: int((coords[1] + 6) / 7) + 2] = 1
