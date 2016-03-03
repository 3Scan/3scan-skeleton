import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
# from skeleton.xlsxwrite import xlsxWrite
# from skeleton.segmentLengths import getSegmentsAndLengths
# from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline, _getBouondariesOfimage
from skeleton.convOptimize import getSkeletonize3D
from skeleton.unitwidthcurveskeleton import getShortestPathskeleton


def excelWriteParallel(listOfNpys):
    for npy in listOfNpys:
        npySkeleton = npy.replace('threshold', 'skeleton')
        # boundaryIm = _getBouondariesOfimage(np.load(npy))
        shskel = getShortestPathskeleton(getSkeletonize3D(np.load(npy)))
        np.save(npySkeleton, shskel)
        # path = (npy.replace('threshold', 'statsExcel')).replace('npy', 'xlsx')
        # d1, d2, d3, t = getSegmentsAndLengths(shskel)
        # d, di = getRadiusByPointsOnCenterline(shskel, boundaryIm)
        # dictR = {your_key: d[your_key] for your_key in d1.keys()}
        # listOfDicts = [dictR, d1, d2, d3]
        # xlsxWrite(listOfDicts, path)


if __name__ == '__main__':
    root = '/media/pranathi/KINGSTON/subsubVolumethresholds/'
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort()
    numProcessors = multiprocessing.cpu_count() - 2
    chunkSize = int(len(listOfNpys) / numProcessors)
    poolLists = []
    for i in range(0, len(listOfNpys), chunkSize + 1):
        print(i)
        poolLists.append(listOfNpys[i: i + chunkSize + 1])
    startt = time.time()
    pool = Pool(processes=numProcessors)
    pool.map(excelWriteParallel, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
