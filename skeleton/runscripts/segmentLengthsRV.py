import itertools

import numpy as np
import networkx as nx
from scipy.ndimage.filters import convolve

"""
    this version assumes its calculating 3D Volume's length and tortuosity
"""


def _getDistanceBetweenPointsInpath(cyclePath, cycle=0):
    """
       finds distance between points in a given path
       if it is a cycle then distance from last coordinate
       to first coordinate in the path is added to the
       distance list (since a cycle contains this edge as
       the last edge in its path)
    """
    distList = []
    if cycle:
        for index, item in enumerate(cyclePath):
            if index + 1 < len(cyclePath):
                item2 = cyclePath[index + 1]
            elif index + 1 == len(cyclePath):
                item2 = cyclePath[0]
            dist = np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2))
            distList.append(dist)
    else:
        for index, item in enumerate(cyclePath):
            if index + 1 != len(cyclePath):
                item2 = cyclePath[index + 1]
                dist = np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2))
                distList.append(dist)
    return sum(distList)


def _removeEdgesInVisitedPath(subGraphskeleton, path, cycle):
    """
       given a visited path in variable path, the edges in the
       path are removed in the graph
       if cycle = 1 , the given path belongs to a cycle,
       so an additional edge is formed between the last and the
       first node to form a closed cycle/ path and is removed
    """
    shortestPathedges = []
    if cycle == 0:
        for index, item in enumerate(path):
            if index + 1 != len(path):
                shortestPathedges.append(tuple((item, path[index + 1])))
        subGraphskeleton.remove_edges_from(shortestPathedges)
    else:
        for index, item in enumerate(path):
            if index + 1 != len(path):
                shortestPathedges.append(tuple((item, path[index + 1])))
            else:
                item = path[0]
                shortestPathedges.append(tuple((item, path[-1])))
        subGraphskeleton.remove_edges_from(shortestPathedges)

# permutations of (-1, 0, 1) in three/two dimensional tuple format
# representing 8 and 26 increments around a pixel at origin (0, 0, 0)
stepDirect = itertools.product((-1, 0, 1), repeat=3)
listStepDirect = list(stepDirect)
listStepDirect.remove((0, 0, 0))
adjtemplate = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                       [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                       [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)


def _getIncrements(configNumber):
    """
       takes in a configNumber and converts into
       binary sequence of 1s and 0s, returns the tuple
       increments corresponding to them
    """
    configNumber = np.int64(configNumber)
    neighborValues = [(configNumber >> digit) & 0x01 for digit in range(26)]
    return [neighborValue * increment for neighborValue, increment in zip(neighborValues, listStepDirect)]


def _setAdjacencylistarray(arr):
    """
        takes in an array and returns a dictionary with nonzero voxels/ pixels
        and their adjcent nonzero coordinates
    """
    result = convolve(np.uint64(arr), adjtemplate, mode='constant', cval=0)
    result[arr == 0] = 0
    dictOfIndicesAndAdjacentcoordinates = {}
    # list of nonzero tuples
    nonZeros = set(map(tuple, np.transpose(np.nonzero(arr))))
    if np.sum(arr) == 1:
        # if there is just one nonzero elemenet there are no adjacent coordinates
        dictOfIndicesAndAdjacentcoordinates[nonZeros[0]] = []
        return dictOfIndicesAndAdjacentcoordinates
    else:
        for item in nonZeros:
            adjacentCoordinatelist = []
            for increments in _getIncrements(result[item]):
                if increments == (()):
                    continue
                adjCoord = np.array(item) + np.array(increments)
                adjacentCoordinatelist.append(tuple(adjCoord))
            dictOfIndicesAndAdjacentcoordinates[item] = adjacentCoordinatelist
    return dictOfIndicesAndAdjacentcoordinates


def getSegmentsAndLengths(imArray):
    """
        algorithm - 1) go through each of the disjoint graphs
                    2) decide if it is one of the following a) line
                    b) cycle c) acyclic tree like structure d) cyclic tree like structure
                    e) single node
                    3) Find all the paths in a given disjoint graph from a point whose degree is greater
                    than or equal to 2 to a point whose degree equal to one
                    4) calculate distance between edges in each path and displacement to find curve length and
                    curve displacement to find tortuosity
                    5) Remove all the edges in this path once they are traced
    """
    dictOfIndicesAndAdjacentcoordinates = _setAdjacencylistarray(imArray)
    networkxGraph = nx.from_dict_of_lists(dictOfIndicesAndAdjacentcoordinates)
    cliques = nx.find_cliques_recursive(networkxGraph)
    # all the nodes/vertices of 3 cliques
    cliques2 = [clq for clq in cliques if len(clq) == 3]
    if len(cliques2) != 0:
        combEdge = [list(itertools.combinations(clique, 2)) for clique in cliques2]
        subGraphEdgelengths = []
        # different combination of edges in the cliques and their lengths
        for combedges in combEdge:
            subGraphEdgelengths.append([np.sum((np.array(item[0]) - np.array(item[1])) ** 2) for item in combedges])
        cliquEdges = []
        # clique edges to be removed are collected here
        # the edges with maximum edge length
        for mainDim, item in enumerate(subGraphEdgelengths):
            if len(set(item)) != 1:
                for subDim, length in enumerate(item):
                    if length == max(item):
                        cliquEdges.append(combEdge[mainDim][subDim])
            else:
                specialCase = combEdge[mainDim]
                diffOfEdges = []
                for numSpcledges in range(0, 3):
                    l1 = list(specialCase[numSpcledges][0]); l2 = list(specialCase[numSpcledges][1])
                    diffOfEdges.append([i - j for i, j in zip(l1, l2)])
                for index, val in enumerate(diffOfEdges):
                    if val[1] == 0:
                        subDim = index
                        break
                cliquEdges.append(combEdge[mainDim][subDim])
        networkxGraph.remove_edges_from(cliquEdges)
    # intitialize all the common variables
    segmentCount = 0
    segmentLength = 0
    segmentTortuosity = 0
    cycles = len(nx.cycle_basis(networkxGraph))
    for subGraphskeleton in nx.connected_component_subgraphs(networkxGraph):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            " if it is a single node"
            continue
        """ if there are more than one nodes decide what kind of subgraph it is
            if it has cycles alone, or a straight line or a directed cyclic/acyclic graph"""
        nodes.sort()
        nodeDegreedict = nx.degree(subGraphskeleton)
        degreeSet = set(list(nodeDegreedict.values()))
        if degreeSet == {2} and nx.is_biconnected(subGraphskeleton):
            """ if the maximum degree is equal to minimum degree it is a circle, set
            tortuosity to infinity (NaN) set to zero here"""
            cycleList = nx.cycle_basis(subGraphskeleton)
            segmentCount += 1
            segmentLength += _getDistanceBetweenPointsInpath(cycleList[0], 1)
            segmentTortuosity += 0
        elif degreeSet == {1, 2} or degreeSet == {1}:
            """ straight line or dichtonomous tree"""
            """ each node is connected to one or two other nodes implies it is a line,
            set tortuosity to 1"""
            segmentCount += 1
            segmentLength += _getDistanceBetweenPointsInpath(nodes, 0)
            segmentTortuosity += 1
        else:
            """ cyclic or acyclic tree """
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
            endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
            branchendpoints = branchpoints + endpoints
            branchpoints.sort(); endpoints.sort()
            listOfPerms = (list(itertools.product(branchpoints, endpoints)) + list(itertools.permutations(branchpoints, 2)))
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                    simplePaths = [simplePath for simplePath in simplePaths if len((set(branchendpoints) & set(simplePath))) == 2]
                    for simplePath in simplePaths:
                        segmentCount += 1
                        curveLength = _getDistanceBetweenPointsInpath(simplePath)
                        curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                        segmentLength += curveLength
                        segmentTortuosity += (curveLength / curveDisplacement)
                        _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                cycleList = nx.cycle_basis(subGraphskeleton)
                if subGraphskeleton.number_of_edges() != 0 and len(cycleList) != 0:
                    for cycle in cycleList:
                        segmentCount += 1
                        segmentLength += _getDistanceBetweenPointsInpath(cycle, 1)
                        segmentTortuosity += 0
    return segmentCount, segmentLength, segmentTortuosity, cycles
