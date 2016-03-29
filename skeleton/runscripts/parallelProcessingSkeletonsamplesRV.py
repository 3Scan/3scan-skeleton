import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.filters import threshold_otsu
from skeleton.runscripts.thin3DVolumeRV import getSkeleton3D
from skeleton.runscripts.segmentLengthsRV import getSegmentsAndLengths


def convert(tupValList):
    for npy in tupValList:
        subSubvolume = np.load(npy)
        histData, bins = np.histogram(subSubvolume.flatten(), 256, [0, 256])
        vsum = 0
        for i in range(0, 255):
            vsum += histData[i]
            if vsum < tsum:
                t = i
        subSubvolume[subSubvolume < t] = 0
        l1 = np.array([y for y in subSubvolume.ravel() if y != 0])
        t = threshold_otsu(l1)
        interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
        interpolatedIm = interpolatedIm > (0.85 * t)
        i = np.ascontiguousarray(interpolatedIm, dtype=np.uint16)
        erode_im = fftconvolve(i, selem, mode='same')
        erode_im[interpolatedIm == 0] = 0
        percentVasc = np.sum(interpolatedIm) / totalSize
        threshPath = npy.replace('greyscale', 'threshold')
        if np.sum(erode_im >= 8000) == 0 and percentVasc < 0.1 and os.path.exists(threshPath) == 0:
            np.save(threshPath, interpolatedIm)
            skeleton = getSkeleton3D(interpolatedIm)
            np.save(npy.replace('greyscale', 'skeleton'), skeleton)
            path = (npy.replace('greyscale', 'stat')).replace('npy', 'txt')
            f = open(path, 'w')
            d1, d2, d3, cycles = getSegmentsAndLengths(skeleton)
            d = [str(percentVasc) + "\n", str(d1) + "\n", str(d2) + "\n", str(d3) + "\n", str(cycles) + "\n"]
            f.writelines(d)
            f.close()


if __name__ == '__main__':
    totalSize = 2460375.0
    selem = np.zeros((31, 31, 31), dtype=np.uint16)
    xs, ys, zs = np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
    r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    selem[(r < 1)] = 1
    tsum = 0.8 * 19 * 135 * 135
    root = '/home/pranathi/subsubVolumegreyscale/'
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort()
    numProcessors = multiprocessing.cpu_count()
    chunkSize = int(len(listOfNpys) / numProcessors)
    poolLists = []
    for i in range(0, len(listOfNpys), chunkSize + 1):
        poolLists.append(listOfNpys[i: i + chunkSize + 1])
    startt = time.time()
    pool = Pool(processes=numProcessors)
    pool.map(convert, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
