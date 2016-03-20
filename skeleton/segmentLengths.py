import itertools
import time

import numpy as np
import networkx as nx
from scipy import ndimage

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
    totalSegments = 0
    typeGraphdict = {}
    # list of disjointgraphs
    visitedSources = []
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
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
                segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, sourceOnCycle] = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, sourceOnCycle] = 0
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                """ straight line or dichtonomous tree"""
                branchpoints = nodes
                listOfPerms = list(itertools.permutations(branchpoints, 2))
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
                    for sourceOnTree, item in listOfPerms:
                        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                            for simplePath in nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item):
                                if len(list(set(branchpoints) & set(simplePath))) == 2:
                                    if sourceOnTree not in visitedSources:
                                        "check if the same source has multiple segments"
                                        segmentCountdict[sourceOnTree] = 1
                                        visitedSources.append(sourceOnTree)
                                    else:
                                        segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                                    curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                    curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                    segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                                    segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                else:
                    """ each node is connected to one or two other nodes implies it is a line,
                    set tortuosity to 1"""
                    sourceOnLine = nodes[0]; targetOnLine = nodes[-1]
                    if sourceOnLine not in visitedSources:
                        segmentCountdict[sourceOnLine] = 1
                        visitedSources.append(sourceOnLine)
                    else:
                        segmentCountdict[sourceOnLine] += 1
                    segmentLengthdict[segmentCountdict[sourceOnLine], sourceOnLine, targetOnLine] = _getDistanceBetweenPointsInpath(nodes, 0)
                    segmentTortuositydict[segmentCountdict[sourceOnLine], sourceOnLine, targetOnLine] = 1
                    edges = subGraphskeleton.edges()
                    subGraphskeleton.remove_edges_from(edges)
                    typeGraphdict[ithDisjointgraph] = 2
            else:
                """ cyclic or acyclic tree """
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                branchendpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
                branchpoints.sort(); endpoints.sort()
                listOfPerms = list(itertools.product(branchpoints, endpoints))
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                        for simplePath in nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item):
                            if len(list(set(branchendpoints) & set(simplePath))) == 2:
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
                            for simplePath in nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item):
                                if len(list(set(branchpoints) & set(simplePath))) == 2:
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
        assert subGraphskeleton.number_of_edges() == 0
    totalSegments = len(segmentLengthdict)
    print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - startt))
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict


if __name__ == '__main__':
    shskel = np.load(input("enter a path to shortest path skeleton volume------"))
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict = getSegmentsAndLengths(shskel)
