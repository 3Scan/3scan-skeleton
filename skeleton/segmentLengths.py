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
    # print(" in _getDistanceBetweenPointsInpath function")
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


def _getSort(cycles, nodeDim):
    """
       sort one or two or three dimensional coordinates from least
       first coordinate, second coordinate, third coordinate
       to highest only if it is not sorted
    """
    # print("sorting", len(cycles), "number of nodes")
    if nodeDim != 1:
        for i in range(nodeDim):
            cycles.sort(key=lambda x: (x[i]))
        return cycles
    elif sorted(cycles) != cycles:
        return cycles.sort()


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
    """ check if the nodes are of one dimensions or more than one dimensions
        and intitialize the variable nodeDim used in sorting nodes later """
    nodesFindDimension = disjointGraphs[0].nodes()
    if type(nodesFindDimension[0]) == tuple:
        nodeDim = len(nodesFindDimension[0])
    else:
        nodeDim = 0
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        # starttDisjoint = time.time()
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
            _getSort(nodes, nodeDim)
            cycleList = nx.cycle_basis(subGraphskeleton)
            cycleCount = len(cycleList)
            nodeDegreedict = nx.degree(subGraphskeleton)
            degreeList = list(nodeDegreedict.values())
            endPointdegree = min(degreeList)
            branchPointdegree = max(degreeList)
            if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount != 0:
                # print("1 cycle in disjoint graph and no other components")
                # print("number of cycles in the graph is", cycleCount)
                """ if the maximum degree is equal to minimum degree it is a circle, set
                tortuosity to infinity (NaN) """
                cycle = cycleList[0]
                sourceOnCycle = cycle[0]
                segmentCountdict[sourceOnCycle] = 1
                segmentLengthdict[sourceOnCycle, cycle[-1]] = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentTortuositydict[sourceOnCycle, cycle[-1]] = 0
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)):
                # print("line segment with no tree structure")
                """ each node is connected to one or two other nodes implies it is a line,
                set tortuosity to 1"""
                sourceOnLine = nodes[0]; targetOnLine = nodes[-1]
                segmentCountdict[sourceOnLine] = 1
                segmentLengthdict[sourceOnLine, targetOnLine] = _getDistanceBetweenPointsInpath(nodes, 0)
                segmentTortuositydict[sourceOnLine, targetOnLine] = 1
                edges = subGraphskeleton.edges()
                subGraphskeleton.remove_edges_from(edges)
            elif cycleCount >= 1:
                # print("cycle (more than 1) and tree like structures")
                visitedSources = []
                """go through each of the cycles and find the lengths, set tortuosity to NaN (circle)"""
                for nthCycle, cyclePath in enumerate(cycleList):
                    cycleOnSamesource = 1
                    sourceOnCycle = cyclePath[0]
                    if sourceOnCycle not in visitedSources:
                        "check if the same source has multiple loops/cycle"
                        segmentCountdict[sourceOnCycle] = cycleOnSamesource
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
                    branchEndpoints = [k for (k, v) in nodeDegreedict.items() if v == endPointdegree or v == branchPointdegree]
                    branchpoints = [k for (k, v) in nodeDegreedict.items() if v == branchPointdegree]
                    _getSort(branchpoints, nodeDim)
                    _getSort(branchEndpoints, nodeDim)
                    for i, sourceOnTree in enumerate(branchpoints):
                        segment = 0
                        for item in branchEndpoints:
                            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                            for simplePath in simplePaths:
                                if len(list(set(branchEndpoints) & set(simplePath))) != 2:
                                    continue
                                if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                                    segment += 1
                                    curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                    curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                    segmentLengthdict[sourceOnTree, item] = curveLength
                                    segmentTortuositydict[sourceOnTree, item] = curveLength / curveDisplacement
                                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                            segmentCountdict[sourceOnTree] = segment

            else:
                "acyclic tree characteristics"
                # print("tree like structure")
                branchEndpoints = [k for (k, v) in nodeDegreedict.items() if v == endPointdegree or v == branchPointdegree]
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v == branchPointdegree]
                _getSort(branchpoints, nodeDim)
                _getSort(branchEndpoints, nodeDim)
                for i, sourceOnTree in enumerate(branchpoints):
                    segment = 0
                    for item in branchEndpoints:
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            if len(list(set(branchEndpoints) & set(simplePath))) != 2:
                                continue
                            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                                segment += 1
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segmentLengthdict[segment, sourceOnTree, item] = curveLength
                                segmentTortuositydict[segment, sourceOnTree, item] = curveLength / curveDisplacement
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                        segmentCountdict[sourceOnTree] = segment
            if subGraphskeleton.number_of_edges() == 0:
                continue
            else:
                print("edges", subGraphskeleton.number_of_edges());
                break
            assert subGraphskeleton.number_of_edges() == 0
        # print("time taken in {} disjoint graph is {}".format(ithDisjointgraph, time.time() - starttDisjoint), "seconds")
    print(sum(segmentCountdict.values()), len(segmentLengthdict), len(segmentTortuositydict))
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
