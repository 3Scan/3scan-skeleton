import numpy as np
import networkx as nx
import itertools
import time

from six.moves import cPickle
from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.segmentLengths import _removeEdgesInVisitedPath


def getBinWrite(imArray, pathTosave):
    """
       takes in a numpy array and converts it to a obj file and writes it to pathTosave
    """
    startt = time.time()  # for calculating time taken to write to an obj file
    if type(imArray) == np.ndarray:
        networkxGraph = getNetworkxGraphFromarray(imArray, True)  # converts array to a networkx graph(based on non zero coordinates and the adjacent nonzeros)
        networkxGraph = removeCliqueEdges(networkxGraph)  # remove cliques in the graph
    else:
        networkxGraph = imArray
    binFile = open(pathTosave, "wb")  # open a obj file in the given path
    GraphList = nx.to_dict_of_lists(networkxGraph)  # convert the graph to a dictionary with keys as nodes and list of adjacent nodes as the values
    verticesSorted = list(GraphList.keys())  # list and sort the keys so they are geometrically in the same order when writing to an obj file as (l_prefix paths)
    verticesSorted.sort()
    mapping = {}  # initialize variables for writing string of vertex v followed by x, y, x coordinates in the obj file
    #  for each of the sorted vertices
    for index, vertex in enumerate(verticesSorted):
        mapping[vertex] = index + 1  # a mapping to transform the vertices (x, y, z) to indexes (beginining with 1)
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    # line prefixes for the obj file
    disjointGraphs = list(nx.connected_component_subgraphs(networkGraphIntegerNodes))
    dictBranchPoints = {}
    Fibers = []
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            continue
            """ if there are more than one nodes decide what kind of subgraph it is
                if it has cycles alone, or a straight line or a directed cyclic/acyclic graph"""
        nodes.sort()
        cycleList = nx.cycle_basis(subGraphskeleton)
        cycleCount = len(cycleList)
        nodeDegreedict = nx.degree(subGraphskeleton)
        degreeList = list(nodeDegreedict.values())
        endPointdegree = min(degreeList)
        branchPointdegree = max(degreeList)
        if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount != 0:
            """ if the maximum degree is equal to minimum degree it is a circle"""
            cycle = cycleList[0]
            cycle.append(cycle[0])
            _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
        elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
            """ each node is connected to one or two other nodes implies and there is a
            one branch point at a distance not equal to one it is a single dichotomous tree"""
            edges = subGraphskeleton.edges()
            subGraphskeleton.remove_edges_from(edges)
        else:
            "acyclic tree characteristics"""
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
            endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
            branchpoints.sort()
            visitedSources = []
            listOfPerms = list(itertools.product(branchpoints, endpoints))
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                    for simplePath in simplePaths:
                        Fibers.append(simplePath)
                        fiberIndex = len(Fibers)
                        if sourceOnTree not in visitedSources:
                            "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                            dictBranchPoints[sourceOnTree] = fiberIndex
                        else:
                            if type(dictBranchPoints[sourceOnTree]) is int:
                                fiberList = [dictBranchPoints[sourceOnTree]]
                            else:
                                fiberList = dictBranchPoints[sourceOnTree]
                            fiberList.append(fiberIndex)
                            dictBranchPoints[sourceOnTree] = fiberList
                        visitedSources.append(sourceOnTree)
                        _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
            if subGraphskeleton.number_of_edges() != 0:
                listOfPerms = list(itertools.permutations(branchpoints, 2))
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            Fibers.append(simplePath)
                            fiberIndex = len(Fibers)
                            if sourceOnTree not in visitedSources:
                                "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                                dictBranchPoints[sourceOnTree] = fiberIndex
                            else:
                                if type(dictBranchPoints[sourceOnTree]) is int:
                                    fiberList = [dictBranchPoints[sourceOnTree]]
                                else:
                                    fiberList = dictBranchPoints[sourceOnTree]
                                fiberList.append(fiberIndex)
                                dictBranchPoints[sourceOnTree] = fiberList
                            visitedSources.append(sourceOnTree)
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
        assert subGraphskeleton.number_of_edges() == 0
    dictBinary = {}
    dictBinary["edge"] = Fibers
    dictBinary["vertices"] = dictBranchPoints
    dictBinary["mapping"] = mapping
    cPickle.dump(dictBinary, binFile, protocol=cPickle.HIGHEST_PROTOCOL)
    print("binFile file write took %0.3f seconds" % (time.time() - startt))
    binFile.close()
    return Fibers, dictBranchPoints, mapping


if __name__ == '__main__':
    skeletonIm = np.load(input("enter a path to shortest path skeleton volume------"))
    Fibers, dictBranchPoints, mapping = getBinWrite(skeletonIm, "PV_rT.bionet")
