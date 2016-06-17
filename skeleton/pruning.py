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


def getPrunedSkeleton(skel):
    print(np.unique(skel))
    start_prune = time.time()
    label_img1, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
    networkxGraph = getNetworkxGraphFromarray(skel)
    networkxGraph = removeCliqueEdges(networkxGraph)
    ndd = nx.degree(networkxGraph)
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    listIndices = list(np.transpose(np.array(np.where(skel != 0))))
    skelD = np.copy(skel)
    for endPoint in listEndIndices:
        listOfBranchDists = []
        D = np.zeros(skel.shape)
        for item in listIndices:
            tupItem = tuple(item)
            tupEnd = tuple(endPoint)
            if nx.has_path(networkxGraph, tupEnd, tupItem):
                simplePath = next(nx.all_simple_paths(networkxGraph, source=tupEnd, target=tupItem))
                dist = _getDistanceBetweenPointsInpath(simplePath)
                D[tupItem] = dist
                if tupItem in listBranchIndices:
                    listOfBranchDists.append(dist)
        if listOfBranchDists != []:
            skelD[D < min(listOfBranchDists)] = 0

    label_img2, countObjectsPruned = ndimage.measurements.label(skelD, structure=np.ones((3, 3, 3), dtype=np.uint8))
    print("time taken is %0.3f seconds" % (time.time() - start_prune))
    assert countObjects == countObjectsPruned, "Number of disconnected objects in pruned skeleton {} is greater than input objects {}".format(countObjectsPruned, countObjects)
    return skelD


if __name__ == '__main__':
    filePath = input("please enter a root directory where your 3D input is---")
    skel = np.load(filePath)
