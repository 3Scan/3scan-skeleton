import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skeleton.runscipts.thin3DVolumeRV import getThinned3D
from skeleton.runscripts.unitwidthcurveskeletonRV import getShortestPathskeleton
from skeleton.runscripts.segmentLengthsRV import getSegmentsAndLengths


def convert(tupValList):
    for npy in listOfNpys:
        subSubvolume = np.load(npy)
        histData, bins = np.histogram(subSubvolume.flatten(), 256, [0, 256])
        sumation = np.sum(histData * bins[0:-1])
        weight = np.sum(histData)
        avg = sumation / weight
        subSubvolume[subSubvolume < int(avg)] = 0
        l1 = [y for y in subSubvolume.ravel() if y != 0]
        t = threshold_otsu(np.array(l1))
        interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
        interpolatedIm = interpolatedIm > (0.85 * t)
        skeleton = getShortestPathskeleton(getThinned3D(np.load(npy)))
        path = (npy.replace('threshold', 'stat')).replace('npy', 'txt')
        d1, d2, d3, cycles = getSegmentsAndLengths(skeleton)
        d = [str(d1) + "\n", str(d2) + "\n", str(d3) + "\n", str(cycles), str(np.sum(interpolatedIm) / totalSize) + "\n"]
        f = open(path, 'w')
        f.writelines(d)
        f.close()
        np.save(npy.replace('greyscale', 'skeleton'), skeleton)
        np.save(npy.replace('greyscale', 'threshold'), interpolatedIm)


if __name__ == '__main__':
    totalSize = 2460375.0
    root = '/home/pranathi/subsubVolumegreyscale/'
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort()
    numProcessors = multiprocessing.cpu_count()
    chunkSize = int(len(listOfNpys) / numProcessors)
    poolLists = []
    for i in range(0, len(listOfNpys), chunkSize + 1):
        print(i)
        poolLists.append(listOfNpys[i: i + chunkSize + 1])
    startt = time.time()
    pool = Pool(processes=numProcessors)
    pool.map(convert, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
