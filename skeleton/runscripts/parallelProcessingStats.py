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
        npy = npy.replace('media', 'home')
        npy = npy.replace('User Data/', '')
        pathSkel = npy.replace('threshold', 'skeleton')
        skeleton = getSkeleton3D(interpolatedIm)
        np.save(pathSkel, skeleton)
        path = (npy.replace('threshold', 'stat')).replace('npy', 'txt')
        f = open(path, 'w')
        avgBranchingIndex, numSegments, totalLength, totalTortuosity, cyclesTot = getSegmentStats(skeleton)
        percentVasc = np.sum(interpolatedIm) / totalSize
        d = [str(percentVasc) + "\n", str(avgBranchingIndex) + "\n", str(numSegments) + "\n", str(totalLength) + "\n", str(totalTortuosity) + "\n" + str(cyclesTot) + "\n"]
        f.writelines(d)
        f.close()


if __name__ == '__main__':
    totalSize = 2460375.0
    root = '/media/pranathi/User Data/subsubVolumethresholdNew_28/'
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort()
    listOfNpys.remove(root + 'threshold_460_10007_7087.npy')
    listOfNpys.remove(root + 'threshold_630_8871_7087.npy')
    listN = []
    for npy in listOfNpys:
        npyN = npy.replace('media', 'home')
        npyN = npyN.replace('User Data/', '')
        pathSkel = npyN.replace('threshold', 'skeleton')
        if os.path.exists(pathSkel) == 0:
            listN.append(npy)
    numProcessors = multiprocessing.cpu_count()
    chunkSize = int(len(listN) / numProcessors)
    poolLists = []
    for i in range(0, len(listN), chunkSize + 1):
        poolLists.append(listN[i: i + chunkSize + 1])
    startt = time.time()
    pool = Pool(processes=numProcessors)
    pool.map(convert, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
    # stat_350_7167_2767
