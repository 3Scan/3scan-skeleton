import os
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
from skeleton.runscripts.branchAnglesRV import getBranchAngles


def convert(tupValList):
    for npy in tupValList:
        path = (npy.replace('skeleton', 'stat')).replace('npy', 'txt')
        f = open(path, 'w')
        dist, T1, T2 = getBranchAngles(np.load(npy))
        d = [str(dist) + "\n", str(T1) + "\n", str(T2) + "\n"]
        f.writelines(d)
        f.close()


if __name__ == '__main__':
    totalSize = 2460375.0
    root = '/home/pranathi/subsubVolumeskeletonNew_28/'
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
    # stat_350_7167_2767
