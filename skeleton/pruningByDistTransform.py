import time
import numpy as np
from scipy import ndimage
import networkx as nx
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.segmentStats import _getDistanceBetweenPointsInpath


"""
    code to prune the spurious edges on skeleton
    References:
    https://en.wikipedia.org/wiki/Pruning_(morphology)
    http://www.mathworks.com/matlabcentral/answers/88284-remove-the-spurious-edge-of-skeleton?requestedDomain=www.mathworks.com
"""


def _getChessboardDist(path, cycle=0):
    """
       finds distance between points in a given path
       if it is a cycle then distance from last coordinate
       to first coordinate in the path is added to the
       distance list (since a cycle contains this edge as
       the last edge in its path)
    """
    distList = []
    if cycle:
        for index, item in enumerate(path):
            x1, y1, z1 = item
            if index + 1 < len(path):
                item2 = path[index + 1]
            elif index + 1 == len(path):
                item2 = path[0]
            x2, y2, z2 = item2
            dist = max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
            distList.append(dist)
    else:
        for index, item in enumerate(path[:-1]):
            x1, y1, z1 = item
            item2 = path[index + 1]
            x2, y2, z2 = item2
            dist = max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
            distList.append(dist)
    return sum(distList)


def getPrunedSkeleton(skel):
    start_prune = time.time()
    label_img1, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
    networkxGraph = getNetworkxGraphFromarray(skel)
    networkxGraph = removeCliqueEdges(networkxGraph)
    ndd = nx.degree(networkxGraph)
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    print("number of end points are", len(listEndIndices))
    listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    listIndices = list(map(tuple, np.transpose(np.array(np.where(skel != 0)))))
    print("number of branch points are", len(listBranchIndices))
    skelD = np.copy(skel)
    skelD1 = np.copy(skel)
    count = 0
    for endPoint in listEndIndices:
        D = np.zeros(skel.shape)
        D1 = np.zeros(skel.shape)
        listOfBranchDists = []
        listOfBranchDists2 = []
        for nonzero in listIndices:
            if nx.has_path(networkxGraph, endPoint, nonzero) and endPoint != nonzero:
                simplePath = nx.shortest_path(networkxGraph, endPoint, nonzero)
                dist = _getDistanceBetweenPointsInpath(simplePath)
                dist2 = _getChessboardDist(simplePath)
                D1[nonzero] = dist2
                D[nonzero] = dist
                if nonzero in listBranchIndices:
                    listOfBranchDists.append(dist)
                    listOfBranchDists2.append(dist2)
        if listOfBranchDists != []:
            count += 1
            skelD[D < min(listOfBranchDists)] = 0
            skelD1[D1 < min(listOfBranchDists2)] = 0
    label_img2, countObjectsPruned = ndimage.measurements.label(skelD, structure=np.ones((3, 3, 3), dtype=np.uint8))
    print("time taken to prune is %0.3f seconds" % (time.time() - start_prune))
    # assert countObjects == countObjectsPruned, "Number of disconnected objects in pruned skeleton {} is greater than input objects {}".format(countObjectsPruned, countObjects)
    print(countObjects, countObjectsPruned)
    return skelD, skelD1


if __name__ == '__main__':
    filePath = input("please enter a root directory where your 3D input is---")
    skel = np.load(filePath)
