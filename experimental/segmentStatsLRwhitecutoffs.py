import itertools
import time
import copy
import numpy as np
import networkx as nx
from math import log

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges

from six.moves import cPickle


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


def _removePathInCycle(cycle, simplePath, branchpoints):
    for pathpt in simplePath:
        if pathpt not in branchpoints:
            cycle.remove(pathpt)


def checkBackTracing(simplePath, sortedSegments):
    check = 1
    for path in sortedSegments:
        if len(set(path) & set(simplePath)) > 2:
            check = 0
    return check


def _singleCycle(subGraphskeleton, segmentCountdict, segmentLengthdict, segmentTortuositydict,
                 segmentContractiondict, segmentHausdorffDimensiondict, cycle, visitedSources, cycles, cycleInfo):
    sourceOnCycle = cycle[0]
    if sourceOnCycle not in visitedSources:
        segmentCountdict[sourceOnCycle] = 1
        visitedSources.append(sourceOnCycle)
    else:
        segmentCountdict[sourceOnCycle] += 1
    curveLength = _getDistanceBetweenPointsInpath(cycle, 1)
    cycleInfo[cycles] = [0, curveLength]
    if curveLength > 100:
        segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = curveLength
        segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
        segmentContractiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
    _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)


def _isolatedSegment(subGraphskeleton, nodes, nodeDegreedict, stupidEdges, segmentCountdict, segmentLengthdict, segmentTortuositydict,
                     segmentContractiondict, segmentHausdorffDimensiondict, isolatedEdgeInfo, visitedSources):
    edges = subGraphskeleton.edges()
    stupidEdges += sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in edges])
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
        for sourceOnTree, item in listOfPerms:
            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                simplePath = simplePaths[0]
                if sum([1 for point in simplePath if point in nodes]) == 2:
                    curveLength = _getDistanceBetweenPointsInpath(simplePath, 0)
                    if curveLength > 5:
                        if sourceOnTree not in visitedSources:
                            "check if the same source has multiple segments, if it doesn't number of segments is 1"""
                            segmentCountdict[sourceOnTree] = 1
                            visitedSources.append(sourceOnTree)
                        else:
                            segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                        curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                        segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                        segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                        segmentContractiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                        if curveDisplacement != 0.0:
                            if log(curveDisplacement) != 0.0:
                                segmentHausdorffDimensiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = log(curveLength) / log(curveDisplacement)
                    isolatedEdgeInfo[sourceOnTree, item] = curveLength
                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
    else:
        """ each node is connected to one or two other nodes implies it is a line,
        set tortuosity to 1"""
        endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
        sourceOnLine = endpoints[0]
        targetOnLine = endpoints[1]
        simplePath = nx.shortest_path(subGraphskeleton, source=sourceOnLine, target=targetOnLine)
        curveLength = _getDistanceBetweenPointsInpath(simplePath, 0)
        isolatedEdgeInfo[sourceOnLine, targetOnLine] = curveLength
        if curveLength > 5:
            segmentCountdict[sourceOnLine] = 1
            segmentLengthdict[1, sourceOnLine, targetOnLine] = curveLength
            segmentContractiondict[1, sourceOnLine, targetOnLine] = 1
            segmentTortuositydict[1, sourceOnLine, targetOnLine] = 1
            segmentHausdorffDimensiondict[1, sourceOnLine, targetOnLine] = 1
        edges = subGraphskeleton.edges()
        subGraphskeleton.remove_edges_from(edges)


def _cyclicTree(subGraphskeleton, nodeDegreedict, visitedSources, segmentCountdict, segmentLengthdict, cycleInfo,
                segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, cycleList, cycles, visitedPaths, sortedSegments):
    for nthcycle, cycle in enumerate(cycleList):
        nodeDegreedictFilt = {key: value for key, value in nodeDegreedict.items() if key in cycle}
        branchpoints = [k for (k, v) in nodeDegreedictFilt.items() if v != 2 and v != 1]
        sourceOnCycle = branchpoints[0]
        if len(branchpoints) == 1:
            _singleCycle(subGraphskeleton, segmentCountdict, segmentLengthdict, segmentTortuositydict,
                         segmentContractiondict, segmentHausdorffDimensiondict, cycle, visitedSources, cycles, cycleInfo)
        else:
            for point in cycle:
                if point in branchpoints:
                    sourceOnCycle, point
                    if nx.has_path(subGraphskeleton, source=sourceOnCycle, target=point) and sourceOnCycle != point:
                        simplePath = nx.shortest_path(subGraphskeleton, source=sourceOnCycle, target=point)
                        sortedSegment = sorted(simplePath)
                        if sortedSegment not in sortedSegments and sum([1 for pathpoint in simplePath if pathpoint in branchpoints]) == 2 and checkBackTracing(simplePath, sortedSegments):
                            if sourceOnCycle not in visitedSources:
                                segmentCountdict[sourceOnCycle] = 1
                                visitedSources.append(sourceOnCycle)
                            else:
                                segmentCountdict[sourceOnCycle] += 1
                            curveLength = _getDistanceBetweenPointsInpath(simplePath)
                            curveDisplacement = np.sqrt(np.sum((np.array(sourceOnCycle) - np.array(point)) ** 2))
                            segmentLengthdict[segmentCountdict[sourceOnCycle], sourceOnCycle, point] = curveLength
                            segmentTortuositydict[segmentCountdict[sourceOnCycle], sourceOnCycle, point] = curveLength / curveDisplacement
                            segmentContractiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, point] = curveDisplacement / curveLength
                            if curveDisplacement != 0.0:
                                if log(curveDisplacement) != 0.0:
                                    segmentHausdorffDimensiondict[segmentCountdict[sourceOnCycle], sourceOnCycle, point] = log(curveLength) / log(curveDisplacement)
                            visitedPaths.append(simplePath)
                            sortedSegments.append(sortedSegment)
                    sourceOnCycle = point
        cycleInfo[cycles + nthcycle] = [len(branchpoints), _getDistanceBetweenPointsInpath(cycle, 1)]
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
                "simplePath"
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
                    if curveDisplacement != 0.0:
                        if log(curveDisplacement) != 0.0:
                            segmentHausdorffDimensiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = log(curveLength) / log(curveDisplacement)
                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)


def _branchToBranch(subGraphskeleton, visitedSources, segmentCountdict, segmentLengthdict,
                    segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, branchpoints):
    listOfPerms = list(itertools.permutations(branchpoints, 2))
    for sourceOnTree, item in listOfPerms:
        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
            for simplePath in simplePaths:
                simplePath
                if sum([1 for point in simplePath if point in branchpoints]) == 2:
                    if sourceOnTree not in visitedSources:
                        """edges !=0 check if the same source has multiple segments, if it doesn't number of segments is 1"""
                        segmentCountdict[sourceOnTree] = 1
                        visitedSources.append(sourceOnTree)
                    else:
                        segmentCountdict[sourceOnTree] = segmentCountdict[sourceOnTree] + 1
                    curveLength = _getDistanceBetweenPointsInpath(simplePath)
                    curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                    segmentLengthdict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength
                    segmentTortuositydict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                    segmentContractiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                    if curveDisplacement != 0.0:
                        if log(curveDisplacement) != 0.0:
                            segmentHausdorffDimensiondict[segmentCountdict[sourceOnTree], sourceOnTree, item] = log(curveLength) / log(curveDisplacement)
                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)


def getSegmentStats(imArray, nxg=True):
    """

        INPUT: should be a skeletonized binary[0, 1] or [True, False] array or a network graph, if it is
        then nxG is set to true
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
    cycles = 0
    e = 0
    stupidEdges = 0
    isolatedEdgeInfo = {}
    visitedSources = []
    visitedPaths = []
    sortedSegments = []
    # list of disjointgraphs
    # b = sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in networkxGraph.edges()])
    disjointGraphs = list(nx.connected_component_subgraphs(networkxGraph))
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        orig = copy.deepcopy(subGraphskeleton)
        startDis = time.time()
        numNodes = subGraphskeleton.number_of_nodes()
        print("processing {}th disjoint graph with {} nodes".format(ithDisjointgraph, numNodes))
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            " if it is a single node"
            typeGraphdict[ithDisjointgraph] = 0
        else:
            """ if there are more than one nodes decide what kind of subgraph it is
                if it has cycles alone, or a straight line or a undirected cyclic/acyclic graph"""
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
                _singleCycle(subGraphskeleton, segmentCountdict, segmentLengthdict, segmentTortuositydict,
                             segmentContractiondict, segmentHausdorffDimensiondict, cycle, visitedSources, cycles, cycleInfo)
                cycles = cycles + 1
            elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                """ disjoint line or a bent line at 45 degrees appearing as dichtonomous tree but an error due to
                    improper binarization, so remove them and do not account for statistics"""
                """ straight line or dichtonomous tree"""
                _isolatedSegment(subGraphskeleton, nodes, nodeDegreedict, stupidEdges, segmentCountdict, segmentLengthdict, segmentTortuositydict,
                                 segmentContractiondict, segmentHausdorffDimensiondict, isolatedEdgeInfo, visitedSources)
                typeGraphdict[ithDisjointgraph] = 2
            else:
                """ cyclic or acyclic tree """
                if cycleCount != 0:
                    _cyclicTree(subGraphskeleton, nodeDegreedict, visitedSources, segmentCountdict, segmentLengthdict, cycleInfo, segmentTortuositydict,
                                segmentContractiondict, segmentHausdorffDimensiondict, cycleList, cycles, visitedPaths, sortedSegments)
                "sorted branch and end points in trees"
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 and v != 1]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                branchpoints.sort()
                endpoints.sort()
                _tree(subGraphskeleton, visitedSources, segmentCountdict, segmentLengthdict,
                      segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, branchpoints, endpoints)
                if subGraphskeleton.number_of_edges() != 0:
                    _branchToBranch(subGraphskeleton, visitedSources, segmentCountdict, segmentLengthdict,
                                    segmentTortuositydict, segmentContractiondict, segmentHausdorffDimensiondict, branchpoints)
            assert subGraphskeleton.number_of_edges() == 0, cPickle.dump(orig, open('graph.p', 'wb'))
            e += subGraphskeleton.number_of_edges()
            print("{}th disjoint graph took {} seconds".format(ithDisjointgraph, time.time() - startDis))
    print("edges not removed are", e)
    totalSegments = len(segmentLengthdict)
    listCounts = list(segmentCountdict.values())
    avgBranching = 0
    if len(segmentCountdict) != 0:
        avgBranching = sum(listCounts) / len(segmentCountdict)
    ndd = nx.degree(networkxGraph)
    endP = [1 for key, value in ndd.items() if value == 1]
    branchP = [1 for key, value in ndd.items() if value > 2]
    print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - startt))
    # testing the total network length is same after and before tracing
    # a = sum(list(segmentLengthdict.values())) + stupidEdges
    # np.testing.assert_allclose(a, b)
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, sum(endP), sum(branchP), segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo, isolatedEdgeInfo


if __name__ == '__main__':
    shskel = np.load(input("enter a path to shortest path skeleton volume------"))
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo, isolatedEdgeInfo = getSegmentStats(shskel)
