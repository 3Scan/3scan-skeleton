import itertools
import time

import numpy as np
import networkx as nx
from scipy.ndimage.filters import convolve

from math import pi, acos
from collections import Counter


"""
    Find the segments, lengths and tortuosity of a network
"""


def _getAverageDirectionvec(cyclePath, cycle=0):
    """
       finds distance between points in a given path
       if it is a cycle then distance from last coordinate
       to first coordinate in the path is added to the
       distance list (since a cycle contains this edge as
       the last edge in its path)
    """
    directionVector = np.array([0] * len(cyclePath[0]))
    if cycle:
        for index, item in enumerate(cyclePath):
            if index + 1 < len(cyclePath):
                item2 = cyclePath[index + 1]
            elif index + 1 == len(cyclePath):
                item2 = cyclePath[0]
            directionVector += np.array(item) - np.array(item2)
    else:
        for index, item in enumerate(cyclePath):
            if index + 1 != len(cyclePath):
                item2 = cyclePath[index + 1]
                directionVector += np.array(item) - np.array(item2)
    avgDirVec = directionVector / np.array([len(cyclePath) - 1] * directionVector.size, dtype=np.uint8)
    return avgDirVec


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


def getBranchAngles(imArray):
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
        INPUT: can be either a networkx graph or a  numpy array
        if it is a numpy array set arrayOrNot == True
        if the array is already skeletonized, set skelOrNot == True
        aspectRatio = scale the voxels in 3D volume with aspectRatio
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
                    if val[0] == 0:
                        subDim = index
                        break
                cliquEdges.append(combEdge[mainDim][subDim])
        networkxGraph.remove_edges_from(cliquEdges)
    assert networkxGraph.number_of_selfloops() == 0
    # intitialize all the common variables
    startt = time.time()
    branchAngledict = {}
    segmentCountdict = {}
    visitedSources = [];
    startt = time.time()
    # list of disjointgraphs
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
    for subGraphskeleton in disjointGraphs:
        nodes = subGraphskeleton.nodes()
        if time.time() > startt + 60:
            return 0, 0, 0
        else:
            if len(nodes) == 1:
                " if it is a single node"
            else:
                """ if there are more than one nodes decide what kind of subgraph it is
                    if it has cycles alone, or a straight line or a directed cyclic/acyclic graph"""
                nodes.sort()
                cycleList = nx.cycle_basis(subGraphskeleton)
                cycleCount = len(cycleList)
                nodeDegreedict = nx.degree(subGraphskeleton)
                degreeSet = set(list(nodeDegreedict.values()))
                if degreeSet == {2} and nx.is_biconnected(subGraphskeleton) and cycleCount == 1:
                    """ if the maximum degree is equal to minimum degree it is a circle, set
                    tortuosity to infinity (NaN) set to zero here"""
                    cycle = cycleList[0]
                    sourceOnCycle = cycle[0]
                    segmentCountdict[sourceOnCycle] = 1
                    branchAngledict[1, sourceOnCycle] = _getAverageDirectionvec(cycle, 1)
                    _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
                elif degreeSet == set((1, 2)) or degreeSet == {1}:
                    """ straight line or dichtonomous tree"""
                    listOfPerms = list(itertools.combinations(nodes, 2))
                    if type(nodes[0]) == int:
                        modulus = [[start - end] for start, end in listOfPerms]
                        dists = [abs(i[0]) for i in modulus]
                    else:
                        modulus = [[start[dim] - end[dim] for dim in range(0, 3)] for start, end in listOfPerms]
                        dists = [sum(modulus[i][dim] * modulus[i][dim] for dim in range(0, 3)) for i in range(0, len(modulus))]
                    if len(list(nx.articulation_points(subGraphskeleton))) == 1 and set(dists) != 1:
                        """ each node is connected to one or two other nodes which are not a distance of 1 implies there is a
                            one branch point with two end points in a single dichotomous tree"""
                        continue
                    else:
                        """ each node is connected to one or two other nodes implies it is a line,
                        set tortuosity to 1"""
                        segmentCountdict[nodes[0]] = 1
                        edges = subGraphskeleton.edges()
                        subGraphskeleton.remove_edges_from(edges)
                else:
                    """ cyclic or acyclic tree """
                    if len(cycleList) != 0:
                        for nthcycle, cycle in enumerate(cycleList):
                            nodeDegreedictFilt = {key: value for key, value in nodeDegreedict.items() if key in cycle}
                            branchpoints = [k for (k, v) in nodeDegreedictFilt.items() if v != 2 and v != 1]
                            branchpoints.sort()
                            listOfPerms = list(itertools.combinations(branchpoints, 2))
                            for sourceOnTree, item in listOfPerms:
                                simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                                for simplePath in simplePaths:
                                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                                        if sum([1 for point in simplePath if point in branchpoints]) == 2:
                                            if sourceOnTree not in visitedSources:
                                                "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                                                segmentCountdict[sourceOnTree] = 1
                                                visitedSources.append(sourceOnTree)
                                            else:
                                                segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                                            branchAngledict[segmentCountdict[sourceOnTree], sourceOnTree] = _getAverageDirectionvec(cycle, 1)
                                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                    branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 and v != 1]
                    endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                    branchendpoints = branchpoints + endpoints
                    branchpoints.sort(); endpoints.sort()
                    listOfPerms = list(itertools.product(branchpoints, endpoints))
                    for sourceOnTree, item in listOfPerms:
                        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                            simplePath = simplePaths[0]
                            if sum([1 for point in simplePath if point in branchendpoints]) == 2:
                                if sourceOnTree not in visitedSources:
                                    "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                                    segmentCountdict[sourceOnTree] = 1
                                    visitedSources.append(sourceOnTree)
                                else:
                                    segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                                branchAngledict[segmentCountdict[sourceOnTree], sourceOnTree] = _getAverageDirectionvec(simplePath, 0)
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                    if subGraphskeleton.number_of_edges() != 0:
                        listOfPerms = list(itertools.combinations(branchpoints, 2))
                        for sourceOnTree, item in listOfPerms:
                            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                                simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                                simplePath = simplePaths[0]
                                if sum([1 for point in simplePath if point in branchpoints]) == 2:
                                    if sourceOnTree not in visitedSources:
                                        "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                                        segmentCountdict[sourceOnTree] = 1
                                        visitedSources.append(sourceOnTree)
                                    else:
                                        segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                                    branchAngledict[segmentCountdict[sourceOnTree], sourceOnTree] = _getAverageDirectionvec(simplePath, 0)
                                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                    cycleList = nx.cycle_basis(subGraphskeleton)
                    if subGraphskeleton.number_of_edges() != 0 and len(cycleList) != 0:
                        for cycle in cycleList:
                            sourceOnCycle = cycle[0]
                            if sourceOnCycle not in visitedSources:
                                segmentCountdict[sourceOnCycle] = 1
                                visitedSources.append(sourceOnCycle)
                            else:
                                segmentCountdict[sourceOnCycle] += 1
                            branchAngledict[segmentCountdict[sourceOnCycle], sourceOnCycle] = _getAverageDirectionvec(cycle, 1)
                            _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            # assert subGraphskeleton.number_of_edges() == 0
    ba = {}
    segmentCountdict = {key: segmentCountdict[key] for key, value in segmentCountdict.items() if value > 1}
    if len(segmentCountdict) != 0:
        for key, value in segmentCountdict.items():
            dlist = []
            for i in range(1, value + 1):
                if i + 1 != value + 1:
                    a = branchAngledict[i, key]
                    b = branchAngledict[i + 1, key]
                else:
                    a = branchAngledict[i, key]
                    b = branchAngledict[1, key]
                norm = (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))
                if norm != 0:
                    cos_angle = np.dot(a, b) / norm
                    cos_angle = min(1, max(cos_angle, -1))
                    dlist.append(round(180 * acos(cos_angle) / pi, 2))
                    dlist = [90.0 if item == 180 else item for item in dlist]
                    ba[key] = dlist
        listCounts = list(segmentCountdict.values())
        listAngles = list(ba.values())
        ang = []
        for item in listAngles:
            for subItem in item:
                ang.append(subItem)
        avgBranching = sum(listCounts) / len(segmentCountdict)
        counter = Counter(ang)
        max_count = max(counter.values())
        modeA = [k for k, v in counter.items() if v == max_count]
        counter = Counter(listCounts)
        max_count = max(counter.values())
        modeC = [k for k, v in counter.items() if v == max_count]
        return avgBranching, modeC, modeA
    else:
        return 0, 0, 0
