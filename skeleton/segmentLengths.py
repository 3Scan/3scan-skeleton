import itertools
import time

import numpy as np
import networkx as nx
from scipy import ndimage
from math import log

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges


"""
    Find the segments, lengths and tortuosity of a network
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
        for index, item in enumerate(cyclePath[:-1]):
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


def getSegmentsAndLengths(imArray, skelOrNot=True, arrayOrNot=True, aspectRatio=[1, 1, 1]):
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
    if arrayOrNot is False:
        networkxGraph = imArray
    else:
        imArray = ndimage.interpolation.zoom(imArray, zoom=aspectRatio, order=0)
        networkxGraph = getNetworkxGraphFromarray(imArray, skelOrNot)
        networkxGraph = removeCliqueEdges(networkxGraph)
    assert networkxGraph.number_of_selfloops() == 0
    # intitialize all the common variables
    startt = time.time()
    segmentCountdict = {}
    segmentLengthdict = {}
    segmentTortuositydict = {}
    segmentContractiondict = {}
    segmentFractalDimensiondict = {}
    totalSegments = 0
    typeGraphdict = {}
    # list of disjointgraphs
    visitedSources = []
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
    ndd = nx.degree(networkxGraph)
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            " if it is a single node"
            typeGraphdict[ithDisjointgraph] = 0
        else:
            """ if there are more than one nodes decide what kind of subgraph it is
                if it has cycles alone, or a straight line or a directed cyclic/acyclic graph"""
            nodes.sort()
            cycleList = nx.cycle_basis(subGraphskeleton)
            cycleCount = len(cycleList)
            if cycleCount != 0:
                typeGraphdict[ithDisjointgraph] = 3
            else:
                typeGraphdict[ithDisjointgraph] = 4
            nodeDegreedict = nx.degree(subGraphskeleton)
            degreeList = list(nodeDegreedict.values())
            endPointdegree = min(degreeList)
            branchPointdegree = max(degreeList)
            if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount == 1:
                """ if the maximum degree is equal to minimum degree it is a circle, set
                tortuosity to infinity (NaN) set to zero here"""
                typeGraphdict[ithDisjointgraph] = 1
                cycle = cycleList[0]
                sourceOnCycle = cycle[0]
                if sourceOnCycle not in visitedSources:
                    segmentCountdict[sourceOnCycle] = 1
                    visitedSources.append(sourceOnCycle)
                else:
                    segmentCountdict[sourceOnCycle] += 1
                segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
                segmentContractiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                """ straight line or dichtonomous tree"""
                listOfPerms = list(itertools.combinations(nodes, 2))
                if type(nodes[0]) == int:
                    modulus = [[start - end] for start, end in listOfPerms]
                    dists = [abs(i[0]) for i in modulus]
                else:
                    dims = len(nodes[0])
                    modulus = [[start[dim] - end[dim] for dim in range(0, dims)] for start, end in listOfPerms]
                    dists = [sum(modulus[i][dim] * modulus[i][dim] for dim in range(0, dims)) for i in range(0, len(modulus))]
                if len(list(nx.articulation_points(subGraphskeleton))) == 1 and set(dists) != 1:
                    """ each node is connected to one or two other nodes which are not a distance of 1 implies there is a
                        one branch point with two end points in a single dichotomous tree"""
                    edges = subGraphskeleton.edges()
                    subGraphskeleton.remove_edges_from(edges)
                else:
                    """ each node is connected to one or two other nodes implies it is a line,
                    set tortuosity to 1"""
                    endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                    sourceOnLine = endpoints[0]; targetOnLine = endpoints[1]
                    if sourceOnLine not in visitedSources:
                        segmentCountdict[sourceOnLine] = 1
                        visitedSources.append(sourceOnLine)
                    else:
                        segmentCountdict[sourceOnLine] += 1
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnLine, target=targetOnLine))
                    simplePath = simplePaths[0]
                    curveLength = _getDistanceBetweenPointsInpath(simplePath, 0)
                    curveDisplacement = np.sqrt(np.sum((np.array(sourceOnLine) - np.array(targetOnLine)) ** 2))
                    segmentLengthdict[segmentCountdict[sourceOnLine], sourceOnLine, targetOnLine] = curveLength
                    segmentTortuositydict[segmentCountdict[sourceOnLine], sourceOnLine, targetOnLine] = curveLength / curveDisplacement
                    segmentContractiondict[segmentCountdict[sourceOnLine], sourceOnLine, targetOnLine] = curveDisplacement / curveLength
                    edges = subGraphskeleton.edges()
                    subGraphskeleton.remove_edges_from(edges)
                    typeGraphdict[ithDisjointgraph] = 2
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
                                    curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                    curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                    segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                                    segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                                    segmentContractiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
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
                        curveLength = _getDistanceBetweenPointsInpath(simplePath)
                        curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                        segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                        segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                        segmentContractiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                        if log(curveLength) != 0.0:
                            segmentFractalDimensiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = log(curveDisplacement) / log(curveLength)
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
                            curveLength = _getDistanceBetweenPointsInpath(simplePath)
                            curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                            segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                            segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                            segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                            segmentFractalDimensiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = log(curveDisplacement) / log(curveLength)
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
                    segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = _getDistanceBetweenPointsInpath(cycle, 1)
                    segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
                    segmentContractiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
                    _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
        assert subGraphskeleton.number_of_edges() == 0
    totalSegments = len(segmentLengthdict)
    listCounts = list(segmentCountdict.values())
    avgBranching = sum(listCounts) / len(segmentCountdict)
    endP = [1 for key, value in ndd.items() if value == 1]
    branchP = [1 for key, value in ndd.items() if value > 2]
    print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - startt))
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, sum(endP), sum(branchP), segmentContractiondict, segmentFractalDimensiondict

if __name__ == '__main__':
    shskel = np.load(input("enter a path to shortest path skeleton volume------"))
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentFractalDimensiondict = getSegmentsAndLengths(shskel)
