import itertools

import numpy as np
import networkx as nx
from scipy import ndimage
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges

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
    imArray = ndimage.interpolation.zoom(imArray, zoom=[0.7, 0.7, 0.7], order=0)
    networkxGraph = getNetworkxGraphFromarray(imArray)
    networkxGraph = removeCliqueEdges(networkxGraph)
    # assert networkxGraph.number_of_selfloops() == 0
    # intitialize all the common variables
    segmentCountdict = {}
    segmentLengthdict = {}
    segmentTortuositydict = {}
    # list of disjointgraphs
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            " if it is a single node"
        else:
            """ if there are more than one nodes decide what kind of subgraph it is
                if it has cycles alone, or a straight line or a directed cyclic/acyclic graph"""
            nodes.sort()
            cycleList = nx.cycle_basis(subGraphskeleton)
            cycleCount = len(cycleList)
            nodeDegreedict = nx.degree(subGraphskeleton)
            degreeList = list(nodeDegreedict.values())
            endPointdegree = min(degreeList)
            branchPointdegree = max(degreeList)
            if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount == 1:
                """ if the maximum degree is equal to minimum degree it is a circle, set
                tortuosity to infinity (NaN) set to zero here"""
                cycle = cycleList[0]
                sourceOnCycle = cycle[0]
                segmentCountdict[sourceOnCycle] = 1
                segmentLengthdict[1, sourceOnCycle, sourceOnCycle] = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentTortuositydict[1, sourceOnCycle, sourceOnCycle] = 0
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                """ straight line or dichtonomous tree"""
                subGraphskeleton.remove_edges_from(subGraphskeleton.edges())
            else:
                """ cyclic or acyclic tree """
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                listOfPerms = list(itertools.product(branchpoints, endpoints))
                branchpoints.sort()
                visitedSources = [];
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            if sourceOnTree not in visitedSources:
                                "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                                segmentCountdict[sourceOnTree] = 1
                                visitedSources.append(sourceOnTree)
                            else:
                                segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                            curveLength = _getDistanceBetweenPointsInpath(simplePath)
                            curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                            segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                            segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                if subGraphskeleton.number_of_edges() != 0:
                    listOfPerms = list(itertools.permutations(branchpoints, 2))
                    for sourceOnTree, item in listOfPerms:
                        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                            for simplePath in simplePaths:
                                if sourceOnTree not in visitedSources:
                                    "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                                    segmentCountdict[sourceOnTree] = 1
                                    visitedSources.append(sourceOnTree)
                                else:
                                    segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                                segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
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
                        segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, sourceOnCycle] = _getDistanceBetweenPointsInpath(cycle, 1)
                        segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, sourceOnCycle] = 0
                        _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
    cycles = len(nx.cycle_basis(networkxGraph))
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, cycles
