import time
import numpy as np
from scipy import ndimage
import networkx as nx
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges
import itertools
# from skeleton.thin3DVolume import getThinned3D
# from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton

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


start_prune = time.time()
filePath = input("please enter a root directory where your 3D input is---")
# skel = getShortestPathSkeleton(getThinned3D(np.load(filePath)))
skel = np.load(filePath)
label_img1, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
networkxGraph = getNetworkxGraphFromarray(skel)
networkxGraph = removeCliqueEdges(networkxGraph)
ndd = nx.degree(networkxGraph)
listEndIndices = [k for (k, v) in ndd.items() if v == 1]
listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
branchendpoints = listBranchIndices + listEndIndices
Dmask = np.zeros_like(skel)
maskPresent = np.zeros_like(skel)
D = np.zeros(skel.shape)
count = 0
for sourceOnTree, item in itertools.product(listEndIndices, listBranchIndices):
    listOfBranchDists = []
    for simplePath in nx.all_simple_paths(networkxGraph, source=sourceOnTree, target=item):
        if sum([1 for point in simplePath if point in branchendpoints]) == 2:
            curveLength = np.sqrt(np.sum(np.square(np.array(sourceOnTree) - np.array(item))))
            D[item] = curveLength
            listOfBranchDists.append(curveLength)
    if listOfBranchDists != []:
        count += 1
        Dmask[D < min(listOfBranchDists)] = 1
        print(count, Dmask.sum())
skelD = skel - Dmask
label_img2, countObjectsPruned = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
print("time taken is %0.3f seconds" % (time.time() - start_prune))
assert countObjects == countObjectsPruned, "Number of disconnected objects in pruned skeleton {} is greater than input objects {}".format(countObjectsPruned, countObjects)


