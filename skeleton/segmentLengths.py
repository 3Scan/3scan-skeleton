import time

import numpy as np
import networkx as nx

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges


def _getDistanceBetweenPointsInpath(cyclePath, cycle=0):
    distList = []
    print(" in _getDistanceBetweenPointsInpath function")
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


def _getIfnotvisited(shortestPath, pathAlreadyVisitedgraph):
    visitedEdges = pathAlreadyVisitedgraph.edges()
    shortestPathedges = []
    for index, item in enumerate(shortestPath):
        if index + 1 != len(shortestPath):
            shortestPathedges.append(tuple((item, shortestPath[index + 1])))
    visited = 0
    for item in visitedEdges:
        for shortestPathedge in shortestPathedges:
            if shortestPathedge == item or tuple(reversed(shortestPathedge)) == item:
                visited += 1
                break
            else:
                continue
    pathAlreadyVisitedgraph.add_edges_from(shortestPathedges)
    if visited != 0:
        print("checking if visited or not", visited)
        return 1
    else:
        print("checking if visited or not", visited)
        return 0


def _getSort(cycles, nodeDim):
    print("sorting", len(cycles), "number of nodes")
    if sorted(cycles) != cycles:
        if nodeDim == 2:
            return cycles.sort(key=lambda x: (x[0], x[1]))
        elif nodeDim == 3:
            return cycles.sort(key=lambda x: (x[0], x[1], x[2]))
        else:
            return cycles.sort()
    else:
        return cycles


def getSegmentsAndLengths(networkxGraph):
    assert networkxGraph.number_of_selfloops() == 0
    # intitialize all the common variables
    startt = time.time()
    segmentCountdict = {}
    segmentLengthdict = {}
    segmentTortuositydict = {}
    # list of disjointgraphs
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
    pathAlreadyVisitedgraph = nx.Graph()
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        starttDisjoint = time.time()
        nodes = subGraphskeleton.nodes()
        if type(nodes[0]) == tuple:
            nodeDim = len(nodes[0])
        else:
            nodeDim = 0
        if len(nodes) == 1:
            segmentCountdict[nodes[0]] = 1
            segmentLengthdict[nodes[0], nodes[0]] = 0
            segmentTortuositydict = 1
            totalSegments = 1
        else:
            _getSort(nodes, nodeDim)
            cycleList = nx.cycle_basis(subGraphskeleton)
            cycleCount = len(cycleList)
            nodeDegreedict = nx.degree(subGraphskeleton)
            degreeList = list(nodeDegreedict.values())
            endPointdegree = min(degreeList)
            branchPointdegree = max(degreeList)
            articulationPoints = list(nx.articulation_points(subGraphskeleton))
            if endPointdegree == branchPointdegree and len(articulationPoints) == 0 and cycleCount != 0:
                print("1 cycle in disjoint graph and no other components")
                print("number of cycles in the graph is", cycleCount)
                cycle = cycleList[0]
                sourceOnCycle = cycle[0]
                segmentCountdict[sourceOnCycle] = 1
                segmentTortuositydict[sourceOnCycle, cycle[-1]] = float('NaN')
                pathAlreadyVisitedgraph.add_path(cycle)
                segmentLengthdict[(sourceOnCycle, cycle[-1])] = _getDistanceBetweenPointsInpath(cycle, 1)
            elif set(degreeList) == set((1, 2)):
                print("line segment with no tree structure")
                _getSort(nodes, nodeDim)
                sourceOnLine = nodes[0]; targetOnLine = nodes[-1]
                segmentCountdict[sourceOnLine] = 1
                segmentTortuositydict[sourceOnLine, targetOnLine] = 1
                segmentLengthdict[(sourceOnLine, targetOnLine)] = _getDistanceBetweenPointsInpath(nodes, 0)
            elif cycleCount >= 1:
                print("cycle and tree like structures")
                for cycle in cycleList:
                    sourceOnCycle = cycle[0]
                    segmentCountdict[sourceOnCycle] = 1
                    segmentTortuositydict[sourceOnCycle, cycle[-1]] = float('NaN')
                    shortestPathedges = []
                    for index, item in enumerate(cycle):
                        if index + 1 < len(cycle):
                            shortestPathedges.append(tuple((item, cycle[index + 1])))
                        elif index + 1 == len(cycle):
                            item = cycle[0]
                            shortestPathedges.append(tuple((item, cycle[-1])))
                    pathAlreadyVisitedgraph.add_edges_from(shortestPathedges)
                    segmentLengthdict[(sourceOnCycle, cycle[-1])] = _getDistanceBetweenPointsInpath(cycle, 1)
                branchEndpointsdict = {k: v for (k, v) in nodeDegreedict.items() if v == endPointdegree or v == branchPointdegree}
                branchpointsdict = {k: v for (k, v) in nodeDegreedict.items() if v == branchPointdegree}
                branchpoints = list(branchpointsdict.keys())
                branchEndpoints = list(branchEndpointsdict.keys())
                _getSort(branchpoints, nodeDim)
                _getSort(branchEndpoints, nodeDim)
                for i, sourceOnTree in enumerate(branchpoints):
                    segment = 0
                    for item in branchEndpoints:
                        simplePaths = nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item)
                        for simplePath in simplePaths:
                            if len(list(set(branchEndpoints) & set(simplePath))) != 2:
                                continue
                            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item and _getIfnotvisited(simplePath, pathAlreadyVisitedgraph) == 0:
                                curveLength = nx.shortest_path_length(subGraphskeleton, source=sourceOnTree, target=item)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segment += 1
                                segmentLengthdict[segment, sourceOnTree, item] = curveLength
                                segmentCountdict[sourceOnTree] = segment
                                segmentTortuositydict[segment, sourceOnTree, item] = curveLength / curveDisplacement
            else:
                print("tree like structure")
                branchEndpointsdict = {k: v for (k, v) in nodeDegreedict.items() if v == 1 or v >= 2}
                branchpointsdict = {k: v for (k, v) in nodeDegreedict.items() if v >= 2}
                branchpoints = list(branchpointsdict.keys())
                branchEndpoints = list(branchEndpointsdict.keys())
                _getSort(branchpoints, nodeDim)
                _getSort(branchEndpoints, nodeDim)
                for i, sourceOnTree in enumerate(branchpoints):
                    segment = 0
                    for item in branchEndpoints:
                        simplePaths = nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item)
                        for simplePath in simplePaths:
                            if len(list(set(branchEndpoints) & set(simplePath))) != 2:
                                continue
                            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item and _getIfnotvisited(simplePath, pathAlreadyVisitedgraph) == 0:
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                segment += 1
                                segmentLengthdict[segment, sourceOnTree, item] = curveLength
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segmentCountdict[sourceOnTree] = segment
                                segmentTortuositydict[segment, sourceOnTree, item] = curveLength / curveDisplacement
                # if subGraphskeleton.number_of_edges() != pathAlreadyVisitedgraph.edges():
                #     # do something to trace from the source of edge to the point
        print("time taken in {} disjoint graph is {}".format(ithDisjointgraph, time.time() - starttDisjoint), "seconds")
        totalSegments = sum(segmentCountdict.values())
        print("time taken to calculate segments and their lengths is", time.time() - startt, "seconds", totalSegments)
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments


if __name__ == '__main__':
    shskel = np.load("/home/pranathi/Downloads/shortestPathSkel.npy")
    networkxGraph = getNetworkxGraphFromarray(shskel)
    removeCliqueEdges(networkxGraph)
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments = getSegmentsAndLengths(networkxGraph)
