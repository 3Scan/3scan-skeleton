import time
import numpy as np
from scipy import ndimage
import networkx as nx
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges

"""
    code to prune the spurious edges on skeleton
    References:
    https://en.wikipedia.org/wiki/Pruning_(morphology)
    http://www.mathworks.com/matlabcentral/answers/88284-remove-the-spurious-edge-of-skeleton?requestedDomain=www.mathworks.com
"""


def getPrunedSkeleton(skel, cutoff=9):
    start_prune = time.time()
    label_img1, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
    networkxGraph = getNetworkxGraphFromarray(skel)
    networkxGraph = removeCliqueEdges(networkxGraph)
    ndd = nx.degree(networkxGraph)
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    skelD = np.copy(skel)
    branchendpoints = listEndIndices + listBranchIndices
    for endPoint in listEndIndices:
        for item in listBranchIndices:
            tupItem = tuple(item)
            tupEnd = tuple(endPoint)
            if nx.has_path(networkxGraph, tupEnd, tupItem):
                simplePaths = list(nx.all_simple_paths(networkxGraph, source=tupEnd, target=tupItem, cutoff=cutoff))
                for simplePath in simplePaths:
                    if sum([1 for point in simplePath if point in branchendpoints]) == 2:
                        for pointsSmallBranches in simplePath[:-1]:
                            skelD[pointsSmallBranches] = 0
    label_img2, countObjectsPruned = ndimage.measurements.label(skelD, structure=np.ones((3, 3, 3), dtype=np.uint8))
    print("time taken is %0.3f seconds" % (time.time() - start_prune))
    print(countObjects, countObjectsPruned)
    # assert countObjects == countObjectsPruned, "Number of disconnected objects in pruned skeleton {} is greater than input objects {}".format(countObjectsPruned, countObjects)
    return skelD


if __name__ == '__main__':
    filePath = input("please enter a root directory where your 3D input is---")
    skel = np.load(filePath)
