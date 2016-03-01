import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
from skeleton.segmentStatsDisjointGraph import xlsxWrite
from skeleton.segmentLengths import getSegmentsAndLengths
from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline, _getBouondariesOfimage


def excelWriteParallel(listOfNpys):
    for npy in listOfNpys:
        shskel = np.load(npy)
        npyThresh = npy.replace('skeleton', 'threshold')
        boundaryIm = _getBouondariesOfimage(np.load(npyThresh))
        path = (npy.replace('skeleton', 'statsExcel')).replace('npy', 'xlsx')
        d1, d2, d3, t = getSegmentsAndLengths(shskel)
        d, di = getRadiusByPointsOnCenterline(shskel, boundaryIm)
        dictR = {your_key: d[your_key] for your_key in d1.keys()}
        listOfDicts = [dictR, d1, d2, d3]
        xlsxWrite(listOfDicts, path)


if __name__ == '__main__':
    root = '/home/pranathi/subsubVolumeskeletons/'
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort()
    numProcessors = multiprocessing.cpu_count()
    chunkSize = len(listOfNpys) / numProcessors
    poolLists = []
    for i in range(0, chunkSize):
        poolLists.append(listOfNpys[i: i + chunkSize])
    startt = time.time()
    pool = Pool(processes=numProcessors)
    pool.map(excelWriteParallel, poolLists)
    pool.close()
    pool.join()
    print("time taken is %0.2f seconds" % (time.time() - startt))
