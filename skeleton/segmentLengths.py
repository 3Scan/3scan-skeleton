import itertools
import time

import numpy as np
import networkx as nx

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges


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


def _removeEdgesInVisitedPath(subGraphskeleton, path, cycle=0):
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
            if index + 1 < len(path):
                shortestPathedges.append(tuple((item, path[index + 1])))
            elif index + 1 == len(path):
                item = path[0]
                shortestPathedges.append(tuple((item, path[-1])))
        subGraphskeleton.remove_edges_from(shortestPathedges)


def getSegmentsAndLengths(imArray, skelOrNot=True, arrayOrNot=True):
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
    if arrayOrNot is False:
        networkxGraph = imArray
    else:
        networkxGraph = getNetworkxGraphFromarray(imArray, skelOrNot)
        removeCliqueEdges(networkxGraph)
    assert networkxGraph.number_of_selfloops() == 0
    # intitialize all the common variables
    startt = time.time()
    segmentCountdict = {}
    segmentLengthdict = {}
    segmentTortuositydict = {}
    # list of disjointgraphs
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            " if it is a single node"
            segmentCountdict[nodes[0]] = 1
            segmentLengthdict[nodes[0], nodes[0]] = 0
            segmentTortuositydict[nodes[0], nodes[0]] = 1
            totalSegments = 1
        else:
            """ if there are more than one nodes decide what kind of subgraph it is
                if it has cycles alone, or is a cyclic graph with a tree or an
                acyclic graph with tree """
            nodes.sort()
            cycleList = nx.cycle_basis(subGraphskeleton)
            cycleCount = len(cycleList)
            nodeDegreedict = nx.degree(subGraphskeleton)
            degreeList = list(nodeDegreedict.values())
            endPointdegree = min(degreeList)
            branchPointdegree = max(degreeList)
            if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount != 0:
                """ if the maximum degree is equal to minimum degree it is a circle, set
                tortuosity to infinity (NaN) """
                cycle = cycleList[0]
                sourceOnCycle = cycle[0]
                segmentCountdict[sourceOnCycle] = 1
                segmentLengthdict[sourceOnCycle, cycle[-1]] = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentTortuositydict[sourceOnCycle, cycle[-1]] = 0
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)):
                """ doesn't have points with degree greater than 2, a straight line or a dichotomous tree"""
                branchpoints = list(nx.articulation_points(subGraphskeleton))
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                listOfPerms = list(itertools.product(branchpoints, endpoints))
                if type(nodes[0]) == int:
                    modulus = [[start - end] for start, end in listOfPerms]
                    dists = [abs(i[0]) for i in modulus]
                else:
                    dims = len(nodes[0])
                    modulus = [[start[dim] - end[dim] for dim in range(0, dims)] for start, end in listOfPerms]
                    dists = [sum(modulus[i][dim] * modulus[i][dim] for dim in range(0, dims)) for i in range(0, len(modulus))]
                if len(branchpoints) == 1 and set(dists) != 1:
                    """ each node is connected to one or two other nodes which are not a distance of 1 implies there is a
                        one branch point with two end points in a single dichotomous tree"""
                    branchpoints.sort(); endpoints.sort();
                    visitedSources = []
                    segmentOnSamesource = 1
                    for sourceOnTree, item in listOfPerms:
                        if nx.has_path(subGraphskeleton, sourceOnTree, item):
                            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                            for simplePath in simplePaths:
                                if sourceOnTree not in visitedSources:
                                    "check if the same source has multiple segments"
                                    segmentCountdict[sourceOnTree] = 1
                                else:
                                    segmentOnSamesource += 1
                                    segmentCountdict[sourceOnTree] = segmentOnSamesource
                                visitedSources.append(sourceOnTree)
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segmentLengthdict[segmentOnSamesource, sourceOnTree, item] = curveLength
                                segmentTortuositydict[segmentOnSamesource, sourceOnTree, item] = curveLength / curveDisplacement
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                else:
                    """ each node is connected to one or two other nodes implies it is a line,
                    set tortuosity to 1"""
                    sourceOnLine = nodes[0]; targetOnLine = nodes[-1]
                    segmentCountdict[sourceOnLine] = 1
                    segmentLengthdict[sourceOnLine, targetOnLine] = _getDistanceBetweenPointsInpath(nodes, 0)
                    segmentTortuositydict[sourceOnLine, targetOnLine] = 1
                    _removeEdgesInVisitedPath(subGraphskeleton, nodes, 0)

            elif cycleCount >= 1:
                visitedSources = []
                """go through each of the cycles and find the lengths, set tortuosity to NaN (circle)"""
                for nthCycle, cyclePath in enumerate(cycleList):
                    cycleOnSamesource = 1
                    sourceOnCycle = cyclePath[0]
                    if sourceOnCycle not in visitedSources:
                        "check if the same source has multiple loops/cycle"
                        segmentCountdict[sourceOnCycle] = 1
                    else:
                        cycleOnSamesource += 1
                        segmentCountdict[sourceOnCycle] = cycleOnSamesource
                    visitedSources.append(sourceOnCycle)
                    segmentLengthdict[sourceOnCycle, cyclePath[-1]] = _getDistanceBetweenPointsInpath(cyclePath, 1)
                    segmentTortuositydict[sourceOnCycle, cyclePath[-1]] = 0
                    _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
                if subGraphskeleton.number_of_edges() != 0:
                    "all the cycles in the graph are checked now look for the tree characteristics in this subgraph"
                    # collecting all the branch and endpoints
                    branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
                    endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                    branchpoints.sort(); endpoints.sort();
                    listOfPerms = list(itertools.product(branchpoints, endpoints))
                    visitedSources = []
                    segmentOnSamesource = 1
                    for sourceOnTree, item in listOfPerms:
                        if nx.has_path(subGraphskeleton, sourceOnTree, item):
                            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                            for simplePath in simplePaths:
                                if sourceOnTree not in visitedSources:
                                    "check if the same source has multiple segments"
                                    segmentCountdict[sourceOnTree] = 1
                                else:
                                    segmentOnSamesource += 1
                                    segmentCountdict[sourceOnTree] = segmentOnSamesource
                                visitedSources.append(sourceOnTree)
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segmentLengthdict[segmentOnSamesource, sourceOnTree, item] = curveLength
                                segmentTortuositydict[segmentOnSamesource, sourceOnTree, item] = curveLength / curveDisplacement
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
            else:
                """ acyclic tree """
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                branchpoints.sort(); endpoints.sort();
                listOfPerms = list(itertools.product(branchpoints, endpoints))
                visitedSources = []; segmentOnSamesource = 1
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item):
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            if sourceOnTree not in visitedSources:
                                "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                                segmentCountdict[sourceOnTree] = 1
                            else:
                                segmentOnSamesource += 1
                                segmentCountdict[sourceOnTree] = segmentOnSamesource
                            visitedSources.append(sourceOnTree)
                            curveLength = _getDistanceBetweenPointsInpath(simplePath)
                            curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                            segmentLengthdict[segmentOnSamesource, sourceOnTree, item] = curveLength
                            segmentTortuositydict[segmentOnSamesource, sourceOnTree, item] = curveLength / curveDisplacement
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
        assert subGraphskeleton.number_of_edges() == 0
    assert sum(segmentCountdict.values()) == len(segmentTortuositydict) == len(segmentLengthdict)
    totalSegments = len(segmentLengthdict)
    print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - startt))
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments


if __name__ == '__main__':
    from skeleton.orientationStatisticsSpline import plotKde
    shskel = np.load("/home/pranathi/Downloads/shortestPathSkel.npy")
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments = getSegmentsAndLengths(shskel)
    # getStatistics(segmentCountdict, 'segmentCount')
    # getStatistics(segmentLengthdict, 'segmentLength')
    # getStatistics(segmentTortuositydict, 'segmentTortuosity')
    plotKde(segmentCountdict)
    plotKde(segmentLengthdict)
    plotKde(segmentTortuositydict)
