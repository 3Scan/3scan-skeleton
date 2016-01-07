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
    if sorted(cycles) != cycles:
        if nodeDim == 2:
            return cycles.sort(key=lambda x: (x[0], x[1]))
        elif nodeDim == 3:
            return cycles.sort(key=lambda x: (x[0], x[1], x[2]))
        else:
            return cycles.sort()
    else:
        return cycles


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


def getSegmentsAndLengths(networkxGraph):
    """
       the following function takes in a networkx graph
       and process each of its disjoint graph to obtain
       the characteristics of segments in the disjoint
       subgraph
       ALGORTIHM KEY: Find all paths from a nodes with a
       maximum outgoing nodes(branch point) to all the nodes
       with minimum outgoing node(end point) and maximum outgoing nodes
       if the graph has a path between these nodes and it is already not
       visited. Each time the visited paths are deleted so that
       it minimizes the time of checking if a path is already
       visited by reducing/ removing the premutation of paths
       already checked
    """
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
            segmentTortuositydict = 1
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
                segmentLengthdict[1, sourceOnCycle, cycle[-1]] = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentTortuositydict[1, sourceOnCycle, cycle[-1]] = float('NaN')
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)):
                # print("line segment with no tree structure")
                """ each node is connected to one or two other nodes implies it is a line,
                set tortuosity to 1"""
                sourceOnLine = nodes[0]; targetOnLine = nodes[-1]
                segmentCountdict[sourceOnLine] = 1
                segmentLengthdict[1, sourceOnLine, targetOnLine] = _getDistanceBetweenPointsInpath(nodes, 0)
                segmentTortuositydict[1, sourceOnLine, targetOnLine] = 1
                _removeEdgesInVisitedPath(subGraphskeleton, nodes, 0)
            elif cycleCount >= 1:
                # print("cycle (more than 1) and tree like structures")
                visitedSources = []
                cycleOnSamesource = 0
                """go through each of the cycles and find the lengths, set tortuosity to NaN (circle)"""
                for nthCycle, cyclePath in enumerate(cycleList):
                    sourceOnCycle = cyclePath[0]
                    visitedSources.append(sourceOnCycle)
                    if sourceOnCycle not in visitedSources:
                        "check if the same source has multiple loops/cycles"
                        segmentCountdict[sourceOnCycle] = 1
                    else:
                        cycleOnSamesource += 1
                        segmentCountdict[sourceOnCycle] = cycleOnSamesource
                    segmentLengthdict[nthCycle, sourceOnCycle, cyclePath[-1]] = _getDistanceBetweenPointsInpath(cyclePath, 1)
                    segmentTortuositydict[nthCycle, sourceOnCycle, cyclePath[-1]] = float('NaN')
                    _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
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
                                segmentCountdict[sourceOnTree] = segment
                                segmentLengthdict[segment, sourceOnTree, item] = curveLength
                                segmentTortuositydict[segment, sourceOnTree, item] = curveLength / curveDisplacement
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
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
                                shortestPath = nx.shortest_path(subGraphskeleton, source=sourceOnTree, target=item)
                                curveLength = _getDistanceBetweenPointsInpath(shortestPath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segmentCountdict[sourceOnTree] = segment
                                segmentLengthdict[segment, sourceOnTree, item] = curveLength
                                segmentTortuositydict[segment, sourceOnTree, item] = curveLength / curveDisplacement
                                _removeEdgesInVisitedPath(subGraphskeleton, shortestPath, 0)
            assert subGraphskeleton.number_of_edges() == 0
        # print("time taken in {} disjoint graph is {}".format(ithDisjointgraph, time.time() - starttDisjoint), "seconds")
        totalSegments = sum(segmentCountdict.values())
    print("time taken to calculate segments and their lengths is", time.time() - startt, "seconds", totalSegments)
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments


if __name__ == '__main__':
    shskel = np.load("/home/pranathi/Downloads/shortestPathSkel.npy")
    networkxGraph = getNetworkxGraphFromarray(shskel)
    removeCliqueEdges(networkxGraph)
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments = getSegmentsAndLengths(networkxGraph)
