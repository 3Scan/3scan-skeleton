import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
from skeleton.runscripts.thin3DVolumeRV import getSkeleton3D
from skeleton.runscripts.segmentLengthsRV import getSegmentStats


def convert(tupValList):
    for npy in tupValList:
        interpolatedIm = np.load(npy)
        skeleton = getSkeleton3D(interpolatedIm)
        np.save(npy.replace('threshold', 'skeleton'), skeleton)
        path = (npy.replace('threshold', 'stat')).replace('npy', 'txt')
        f = open(path, 'w')
        d1, d2, d3, cycles = getSegmentStats(skeleton)
        percentVasc = np.sum(interpolatedIm) / totalSize
        d = [str(percentVasc) + "\n", str(d1) + "\n", str(d2) + "\n", str(d3) + "\n", str(cycles) + "\n"]
        f.writelines(d)
        f.close()


if __name__ == '__main__':
    totalSize = 2460375.0
    root = '/home/pranathi/subsubVolumethresholdNew_28/'
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
