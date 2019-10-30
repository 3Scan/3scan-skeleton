from __future__ import print_function
import itertools
import time
import networkx as nx
import numpy as np
cimport cython


"""
program to prune segments of length less than cutoff in  a 3D/2D Array
"""

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _countBranchPointsOnSimplePath(list simplePath, list listBranchIndices):
    """
    Find number of branch points on a path
    Parameters
    ----------
    simplePath : list
        list of nodes on the path

    listBranchIndices : list
        list of branch nodes

    Returns
    -------
    integer
        number of branch nodes on the path
    """
    return sum([1 for point in simplePath if point in listBranchIndices])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _removeNodesOnPath(list simplePaths, skel):
    """
    Returns array changed in place after zeroing out the nodes on simplePath
    Parameters
    ----------
    simplePath : list
        list of list of nodes on simple paths

    skel : numpy array
        2D or 3D numpy array

    """
    cdef list simplePath
    for simplePath in simplePaths:
        for pointsSmallBranches in simplePath[1:]:
            skel[pointsSmallBranches] = 0
#
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getPrunedSkeleton( skeletonStack, networkxGraph, int cutoff=9):
    """
    Returns an array changed in place with segments less than cutoff removed
    Parameters
    ----------
    skeletonStack : numpy array
        2D or 3D numpy array

    networkxGraph : Networkx graph
        graph to remove cliques from

    cutoff : integer
        cutoff of segment length to be removed

    """
    start_prune = time.time()
    ndd = nx.degree(networkxGraph)
    ndd=dict(ndd) 
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    cdef list listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    cdef list branchEndPermutations = list(itertools.product(listEndIndices, listBranchIndices))
    cdef int totalSteps = len(branchEndPermutations)
    cdef list simplePath
    cdef int index
    for index, (endPoint, branchPoint) in enumerate(branchEndPermutations):
        if nx.has_path(networkxGraph, endPoint, branchPoint):  # is it on the same subgraph
            simplePaths = [simplePath for simplePath in nx.all_simple_paths(networkxGraph, source=endPoint,
                           target=branchPoint,  cutoff=cutoff) if _countBranchPointsOnSimplePath(simplePath,listBranchIndices)]
            _removeNodesOnPath(simplePaths, skeletonStack)
        progress = int((100 * (index + 1)) / totalSteps)
        print ("pruning in progress {}% \r".format(progress), end = "", flush = True)
    print("time taken to prune is %0.3f seconds" % (time.time() - start_prune))
    return skeletonStack
