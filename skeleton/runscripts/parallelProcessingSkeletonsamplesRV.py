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
            if vsum < tsum:
                vsum += histData[i]
                t = i
        subSubvolume[subSubvolume < t] = 0
        l1 = np.array([y for y in subSubvolume.ravel() if y != 0])
        threshold = threshold_otsu(l1)
        interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
        interpolatedIm = interpolatedIm > (0.85 * threshold)
        i = np.ascontiguousarray(interpolatedIm, dtype=np.uint16)
        erode_im = fftconvolve(i, selem, mode='same')
        percentVasc = np.sum(interpolatedIm) / totalSize
        threshPath = npy.replace('greyscale', 'threshold')
        if np.max(erode_im) <= 8000 and percentVasc <= 0.1:
            np.save(threshPath, interpolatedIm)
        else:
            t1 = t
            for i in range(t + 1, 255):
                if vsum < t1sum:
                    vsum += histData[i]
                    t1 = i
            subSubvolume[subSubvolume < t1] = 0
            l1 = np.array([y for y in subSubvolume.ravel() if y != 0])
            threshold = threshold_otsu(l1)
            interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
            interpolatedIm = interpolatedIm > (0.85 * threshold)
            i = np.ascontiguousarray(interpolatedIm, dtype=np.uint16)
            erode_im = fftconvolve(i, selem2, mode='same')
            percentVasc = np.sum(interpolatedIm) / totalSize
            if np.max(erode_im) <= 6500 and percentVasc <= 0.07:
                np.save(threshPath, interpolatedIm)
            else:
                t2 = t1
                for i in range(t1 + 1, 255):
                    if vsum < t2sum:
                        vsum += histData[i]
                        t2 = i
                subSubvolume[subSubvolume < t2] = 0
                l1 = np.array([y for y in subSubvolume.ravel() if y != 0])
                threshold = threshold_otsu(l1)
                interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
                interpolatedIm = interpolatedIm > (0.85 * threshold)
                i = np.ascontiguousarray(interpolatedIm, dtype=np.uint16)
                erode_im = fftconvolve(i, selem3, mode='same')
                percentVasc = np.sum(interpolatedIm) / totalSize
                if np.max(erode_im) <= 5000 and percentVasc <= 0.04:
                    np.save(threshPath, interpolatedIm)


if __name__ == '__main__':
    totalSize = 2460375.0
    tsum = 0.8 * 19 * 135 * 135
    t1sum = 0.9 * 19 * 135 * 135
    t2sum = 0.95 * 19 * 135 * 135
    selem = np.zeros((31, 31, 31), dtype=np.uint16)
    xs, ys, zs = np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
    r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    selem[(r < 1)] = 1
    selem2 = np.zeros((29, 29, 29), dtype=np.uint16)
    xs, ys, zs = np.mgrid[-1:1:29j, -1:1:29j, -1:1:29j]
    r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    selem2[(r < 1)] = 1
    selem3 = np.zeros((27, 27, 27), dtype=np.uint16)
    xs, ys, zs = np.mgrid[-1:1:27j, -1:1:27j, -1:1:27j]
    r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    selem3[(r < 1)] = 1
    root = '/home/pranathi/subsubVolumegreyscaleNew_28/'
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
