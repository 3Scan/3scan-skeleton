import itertools
import networkx as nx
import time


def _countBranchPointsOnSimplePath(simplePath, listBranchIndices):
    """
    given a list of nodes as a simplePath, return how many branch points there are on that path
    """
    return sum([1 for point in simplePath if point in listBranchIndices])


def _removeNodesOnPath(simplePaths, skel):
    """
    remove the nodes on simplePath from 3D volume skelD
    """
    for simplePath in simplePaths:
        for pointsSmallBranches in simplePath[:-1]:
            skel[pointsSmallBranches] = 0


def getPrunedSkeleton(skeletonStack, networkxGraph, cutoff=9):
    """
    takes ND-array of skeletonized data and prune branches shorter than length 'cutoff'
    """
    start_prune = time.time()
    ndd = nx.degree(networkxGraph)
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    branchEndPermutations = list(itertools.product(listEndIndices, listBranchIndices))
    totalSteps = len(branchEndPermutations)
    for index, (endPoint, branchPoint) in enumerate(branchEndPermutations):
        if nx.has_path(networkxGraph, endPoint, branchPoint):  # is it on the same subgraph
            simplePaths = [simplePath for simplePath in nx.all_simple_paths(networkxGraph, source=endPoint,
                           target=branchPoint, cutoff=cutoff) if _countBranchPointsOnSimplePath(simplePath, listBranchIndices)]
            _removeNodesOnPath(simplePaths, skeletonStack)
        progress = int((100 * index) / totalSteps)
        print("pruning in progress {}% \r".format(progress), end="", flush=True)
    print("time taken to prune is %0.3f seconds" % (time.time() - start_prune))
    return skeletonStack


