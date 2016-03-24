import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.ndimage.filters import convolve


def convert(tupValList):
    for npy in tupValList:
        interpolatedIm = np.load(npy)
        r = convolve(np.uint64(interpolatedIm), selem, mode='constant', cval=0)
        percentVasc = np.sum(interpolatedIm) / totalSize
        if np.sum(r >= 8000) != 0 and percentVasc < 0.1:
            np.save(npy.replace('threshold', 'thresholdextract'), interpolatedIm)


if __name__ == '__main__':
    totalSize = 2460375.0
    selem = np.zeros((31, 31, 31), dtype=np.uint64)
    xs, ys, zs = np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
    r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    selem[(r < 1)] = 1
    root = '/home/pranathi/subsubVolumethreshold/'
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
