import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
from skeleton.runscripts.thin3DVolumeRV import getSkeleton3D


def convert(tupValList):
    for npy in tupValList:
        interpolatedIm = np.load(npy)
        skeleton = getSkeleton3D(interpolatedIm)
        np.save(npy.replace('skeleton', 'thresholdextract'), skeleton)


if __name__ == '__main__':
    root = '/home/pranathi/subsubVolumethresholdextract/'
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
