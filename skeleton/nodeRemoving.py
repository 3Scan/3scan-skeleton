import numpy as np

import time
import networkx as nx

from skeleton.numberOfBranches import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.segmentLengths import _getSort


def removeIntermediateNodes(networkxGraph):
    """
       remove intermediate joint points/nodes
    """
    startt = time.time()
    newGraph = nx.path_graph(0)
    nodeDegreedict = nx.degree(networkxGraph)
    degreeList = list(nodeDegreedict.values())
    endPointdegree = min(degreeList)
    branchPointdegree = max(degreeList)
    branchEndpointsdict = {k: v for (k, v) in nodeDegreedict.items() if v == endPointdegree or v == branchPointdegree}
    branchpointsdict = {k: v for (k, v) in nodeDegreedict.items() if v == branchPointdegree}
    branchpoints = list(branchpointsdict.keys())
    nodeDim = len(branchpoints[0])
    branchEndpoints = list(branchEndpointsdict.keys())
    _getSort(branchpoints, nodeDim)
    _getSort(branchEndpoints, nodeDim)
    for i, sourceOnTree in enumerate(branchpoints):
        print("sourceOnTree", sourceOnTree)
        for item in branchEndpoints:
            print("initial", item)
            i = 0
            for simplePath in nx.all_simple_paths(networkxGraph, source=sourceOnTree, target=item):
                i += 1
                print("ith path", i)
                if len(list(set(branchEndpoints) & set(simplePath))) != 2:
                    continue
            newGraph.add_edge(sourceOnTree, item)
    print("time taken to remove intermediate nodes is", time.time() - startt, "seconds")
    return newGraph


if __name__ == '__main__':
    shskel = np.load("/Users/3scan_editing/records/shortestPathSkel1.npy")
    networkxGraph, dict1 = getNetworkxGraphFromarray(shskel)
    removeCliqueEdges(networkxGraph)
    newGraph = removeIntermediateNodes(networkxGraph)
