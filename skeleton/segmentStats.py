import itertools
import time

import numpy as np
import networkx as nx
from math import log

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges


"""
    Find the segments, lengths and tortuosity of a network
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


def mod(point):
    return point[0] ** 2 + point[1] ** 2 + point[2] ** 2


def _singleCycle(subGraphskeleton, segmentCountdict, segmentLengthdict, segmentTortuositydict,
                 segmentContractiondict, segmentHausdorffDimensiondict, cycle, visitedSources, cycles, cycleInfo):
    print(cycles, cycleInfo)
    sourceOnCycle = cycle[0]
    if sourceOnCycle not in visitedSources:
        segmentCountdict[sourceOnCycle] = 1
        visitedSources.append(sourceOnCycle)
    else:
        segmentCountdict[sourceOnCycle] += 1
    curveLength = _getDistanceBetweenPointsInpath(cycle, 1)
    cycleInfo[cycles] = [0, curveLength]
    segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = curveLength
    segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
    segmentContractiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
    _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
    print("cycles after", cycles, cycleInfo)


def _cyclicTree(subGraphskeleton, nodeDegreedict, visitedSources, segmentCountdict, segmentLengthdict, cycleInfo,
                segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, cycleList, cycles):
    visitedPaths = []
    for nthcycle, cycle in enumerate(cycleList):
        nodeDegreedictFilt = {key: value for key, value in nodeDegreedict.items() if key in cycle}
        branchpoints = [k for (k, v) in nodeDegreedictFilt.items() if v != 2 and v != 1]
        mods = [mod(point) for point in branchpoints]
        sourceOnCycle = [point for point in branchpoints if point in branchpoints and mod(point) == min(mods)][0]
        for item in branchpoints:
            if nx.has_path(subGraphskeleton, sourceOnCycle, item) and sourceOnCycle != item:
                simplePaths = nx.all_simple_paths(subGraphskeleton, source=sourceOnCycle, target=item)
                for simplePath in simplePaths:
                    if sum([1 for point in simplePath if point in branchpoints]) == 2 and simplePath not in visitedPaths:
                        if sourceOnCycle not in visitedSources:
                            segmentCountdict[sourceOnCycle] = 1
                            visitedSources.append(sourceOnCycle)
                        else:
                            segmentCountdict[sourceOnCycle] += 1
                        curveLength = _getDistanceBetweenPointsInpath(simplePath)
                        cycleInfo[cycles + nthcycle] = [len(branchpoints), curveLength]
                        segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, item] = curveLength
                        segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, item] = 0
                        segmentContractiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, item] = 0
                        visitedPaths.append(simplePath)
        sourceOnCycle = item
    for path in visitedPaths:
        _removeEdgesInVisitedPath(subGraphskeleton, path, 0)
    cycles += len(cycleList)


def _tree(subGraphskeleton, visitedSources, segmentCountdict, segmentLengthdict,
          segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, branchpoints, endpoints):
    listOfPerms = list(itertools.product(branchpoints, endpoints))
    for sourceOnTree, item in listOfPerms:
        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
            for simplePath in simplePaths:
                if sum([1 for point in simplePath if point in branchpoints]) == 1:
                    if sourceOnTree not in visitedSources:
                        "trees - check if the same source has multiple segments, if it doesn't number of segments is 1"""
                        segmentCountdict[sourceOnTree] = 1
                        visitedSources.append(sourceOnTree)
                    else:
                        segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                    curveLength = _getDistanceBetweenPointsInpath(simplePath)
                    curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                    segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                    segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                    segmentContractiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                    if log(curveDisplacement) != 0.0:
                        segmentHausdorffDimensiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = log(curveLength) / log(curveDisplacement)
                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)


def _branchToBranch(subGraphskeleton, visitedSources, segmentCountdict, segmentLengthdict,
                    segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, branchpoints):
    listOfPerms = list(itertools.combinations(branchpoints, 2))
    for sourceOnTree, item in listOfPerms:
        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
            for simplePath in simplePaths:
                if sum([1 for point in simplePath if point in branchpoints]) == 2:
                    if sourceOnTree not in visitedSources:
                        "edges !=0 check if the same source has multiple segments, if it doesn't number of segments is 1"""
                        segmentCountdict[sourceOnTree] = 1
                        visitedSources.append(sourceOnTree)
                    else:
                        segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                    curveLength = _getDistanceBetweenPointsInpath(simplePath)
                    curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                    segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                    segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                    segmentContractiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                    if log(curveDisplacement) != 0.0:
                        segmentHausdorffDimensiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = log(curveLength) / log(curveDisplacement)
                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)


def _remCycles(subGraphskeleton, segmentCountdict, segmentLengthdict, segmentTortuositydict, cycleInfo,
               segmentContractiondict, segmentHausdorffDimensiondict, cycle, cycles, visitedSources, nodeDegreedict):
    cycleList = nx.cycle_basis(subGraphskeleton)
    nodeDegreedictFilt = {key: value for key, value in nodeDegreedict.items() if key in cycle}
    branchpoints = [k for (k, v) in nodeDegreedictFilt.items() if v != 2 and v != 1]
    for nthcycle, cycle in enumerate(cycleList):
        sourceOnCycle = cycle[0]
        if sourceOnCycle not in visitedSources:
            segmentCountdict[sourceOnCycle] = 1
            visitedSources.append(sourceOnCycle)
        else:
            segmentCountdict[sourceOnCycle] += 1
        curveLength = _getDistanceBetweenPointsInpath(cycle, 1)
        cycleInfo[cycles + nthcycle] = [len(branchpoints), curveLength]
        segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = curveLength
        segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
        segmentContractiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
        _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
    cycles = cycles + len(cycleList)


def getSegmentStats(imArray, nxg=False):
    """

        INPUT: should be a skeletonized binary[0, 1] or [True, False] array
        tortuosity = curveLength / curveDisplacement
        contraction = curveDisplacement / curveLength (better becuase there is no change of instability (undefined) in case of cycles)
        Hausdorff Dimension = log(curveLength) / log(curveDisplacement) https://en.wikipedia.org/wiki/Hausdorff_dimension
        Type of subgraphs:
        0 = if graph a single node
        1 = if graph is a single cycle
        2 = line (highest degree in the subgraph is 2)
        3 = undirected acyclic graph
        4 = undirected cyclic graph

        OUTPUT
               segmentCountdict - A dictionary with key as the node(branch or end point) and number of branches connected to it
               segmentLengthdict - A dictionary with key as the (branching index (nth branch from the start node), start node, end node of the branch) value = length of the branch
               segmentTortuositydict - A dictionary with key as the (branching index (nth branch from the start node), start node, end node of the branch) value = tortuosity of the branch
               totalSegments - total number of branches (segments between branch and branch points, branch and end points)
               typeGraphdict - A dictionary with the nth disjoint graph as the key and the type of graph as the value
               avgBranching - Average branching index (number of branches at a branch point) of the network
               sum(endP) - Number of nodes with only one other node connected to them
               sum(branchP) - Number of nodes with more than 2 nodes connected to them
               segmentContractiondict - A dictionary with key as the (branching index (nth branch from the start node), start node, end node of the branch) value = contraction of the branch
               segmentHausdorffDimensiondict - A dictionary with key as the (branching index (nth branch from the start node), start node, end node of the branch) value = hausdorff dimension of the branch
    """
    if nxg:
        networkxGraph = imArray
    else:
        networkxGraph = getNetworkxGraphFromarray(imArray)
        networkxGraph = removeCliqueEdges(networkxGraph)
        assert networkxGraph.number_of_selfloops() == 0
    # intitialize all the common variables
    startt = time.time()
    segmentCountdict = {}
    segmentLengthdict = {}
    segmentTortuositydict = {}
    segmentContractiondict = {}
    segmentHausdorffDimensiondict = {}
    totalSegments = 0
    typeGraphdict = {}
    cycleInfo = {}
    cycles = 1
    # list of disjointgraphs
    visitedSources = []
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
    ndd = nx.degree(networkxGraph)
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        startDis = time.time()
        numNodes = subGraphskeleton.number_of_nodes()
        print("processing {}th disjoint graph with {} nodes".format(ithDisjointgraph, numNodes))
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
                print("in loop")
                print(cycle, cycleCount, cycles, cycleInfo)
                _singleCycle(subGraphskeleton, segmentCountdict, segmentLengthdict, segmentTortuositydict,
                             segmentContractiondict, segmentHausdorffDimensiondict, cycle, visitedSources, cycles, cycleInfo)
                cycles = cycles + 1
            elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                edges = subGraphskeleton.edges()
                subGraphskeleton.remove_edges_from(edges)
                typeGraphdict[ithDisjointgraph] = 2
            else:
                """ cyclic or acyclic tree """
                if cycleCount != 0:
                    _cyclicTree(subGraphskeleton, nodeDegreedict, visitedSources, segmentCountdict, segmentLengthdict, cycleInfo,
                                segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, cycleList, cycles)
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 and v != 1]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                "sorted branch and end points in trees"
                _tree(subGraphskeleton, visitedSources, segmentCountdict, segmentLengthdict,
                      segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, branchpoints, endpoints)
                if subGraphskeleton.number_of_edges() != 0:
                    _branchToBranch(subGraphskeleton, visitedSources, segmentCountdict, segmentLengthdict,
                                    segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, branchpoints)
                if subGraphskeleton.number_of_edges() != 0:
                    _remCycles(subGraphskeleton, segmentCountdict, segmentLengthdict, segmentTortuositydict, cycleInfo,
                               segmentContractiondict, segmentHausdorffDimensiondict, cycle, cycles, visitedSources, nodeDegreedict)
            print(subGraphskeleton.number_of_edges())
            print("{}th disjoint graph took {} seconds".format(ithDisjointgraph, time.time() - startDis))
    totalSegments = len(segmentLengthdict)
    listCounts = list(segmentCountdict.values())
    avgBranching = 0
    if len(segmentCountdict) != 0:
        avgBranching = sum(listCounts) / len(segmentCountdict)
    endP = [1 for key, value in ndd.items() if value == 1]
    branchP = [1 for key, value in ndd.items() if value > 2]
    print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - startt))
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, sum(endP), sum(branchP), segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo

if __name__ == '__main__':
    shskel = np.load(input("enter a path to shortest path skeleton volume------"))
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(shskel)
