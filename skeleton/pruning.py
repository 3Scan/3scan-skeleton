import time
import numpy as np
from scipy import ndimage
import networkx as nx
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges
import itertools

"""
    code to prune the spurious edges on skeleton
    References:
    https://en.wikipedia.org/wiki/Pruning_(morphology)
    http://www.mathworks.com/matlabcentral/answers/88284-remove-the-spurious-edge-of-skeleton?requestedDomain=www.mathworks.com
"""


def getPrunedSkeleton(skel, cutoff=9):
    """
    take 3D ND-array of skeletonized data and prune branches shorter than length 'cutoff'
    """
    # NOTE MJP: we should not be using the full 3D array when we can avoid it, since a sparse
    #    representation will be more efficient
    start_prune = time.time()
    label_img1, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
    del label_img1
    networkxGraph = removeCliqueEdges(getNetworkxGraphFromarray(skel))
    ndd = nx.degree(networkxGraph)
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    skelD = np.copy(skel)
    del skel
    bre = list(itertools.product(listEndIndices, listBranchIndices))
    totalSteps = len(bre)
    for index, (endPoint, branchPoint) in enumerate(bre):
        if nx.has_path(networkxGraph, endPoint, branchPoint):  # is it on the same subgraph
            simplePaths = [simplePath for simplePath in nx.all_simple_paths(networkxGraph, source=endPoint, target=branchPoint, cutoff=cutoff) if countBranchPointsOnSimplePath(simplePath, listBranchIndices)]
            removeNodesOnPath(simplePaths, skelD)
        progress = int((100 * index) / totalSteps)
        print("pruning in progress {}% \r".format(progress), end="", flush=True)
    label_img2, countObjectsPruned = ndimage.measurements.label(skelD, structure=np.ones((3, 3, 3), dtype=np.uint8))
    del label_img2
    print("time taken to prune is %0.3f seconds" % (time.time() - start_prune))
    print(countObjects, countObjectsPruned)
    # assert countObjects == countObjectsPruned, "Number of disconnected objects in pruned skeleton {} is greater than input objects {}".format(countObjectsPruned, countObjects)
    return skelD


def countBranchPointsOnSimplePath(simplePath, listBranchIndices):
    """
    given a list of nodes as a simplePath, return how many branch points there are on that path
    """
    return sum([1 for point in simplePath if point in listBranchIndices])


def removeNodesOnPath(simplePaths, skelD):
    """
    remove the nodes on simplePath from 3D volume skelD
    """
    for simplePath in simplePaths:
        for pointsSmallBranches in simplePath[:-1]:
            skelD[pointsSmallBranches] = 0
    return skelD  # we are doing removal in place, so technically dont need to return it


if __name__ == '__main__':
    filePath = input("please enter a root directory where your 3D input is---")
    skel = np.load(filePath)
