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
    start_prune = time.time()
    label_img1, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
    networkxGraph = getNetworkxGraphFromarray(skel)
    networkxGraph = removeCliqueEdges(networkxGraph)
    ndd = nx.degree(networkxGraph)
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    print("number of end points are", len(listEndIndices))
    listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    listIndices = networkxGraph.nodes()
    skelD = np.zeros_like(skel)
    for item in listIndices:
        skelD[item] = 1
    print("number of branch points are", len(listBranchIndices))
    for endPoint in listEndIndices:
        D = np.zeros(skel.shape)
        listOfBranchDists = []
        for nonzero in listIndices:
            if nx.has_path(networkxGraph, endPoint, nonzero) and endPoint != nonzero:
                simplePath = nx.shortest_path(networkxGraph, endPoint, nonzero)
                dist = _getDistanceBetweenPointsInpath(simplePath)
                D[nonzero] = dist
                if nonzero in listBranchIndices:
                    listOfBranchDists.append(dist)
        if listOfBranchDists != []:
            skelD[D < min(listOfBranchDists)] = 0

    label_img2, countObjectsPruned = ndimage.measurements.label(skelD, structure=np.ones((3, 3, 3), dtype=np.uint8))
    print("time taken to prune is %0.3f seconds" % (time.time() - start_prune))
    # assert countObjects == countObjectsPruned, "Number of disconnected objects in pruned skeleton {} is greater than input objects {}".format(countObjectsPruned, countObjects)
    print(countObjects, countObjectsPruned)
    return skelD


if __name__ == '__main__':
    filePath = input("please enter a root directory where your 3D input is---")
    skel = np.load(filePath)
    skelPruned = getPrunedSkeleton(skel)
