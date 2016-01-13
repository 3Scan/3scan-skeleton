import time

import numpy as np
import networkx as nx

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.segmentLengths import _getSort, _getDistanceBetweenPointsInpath, _removeEdgesInVisitedPath


def getSegmentsAndLengths(imArray):
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
       already checked.
    """
    networkxGraph = getNetworkxGraphFromarray(imArray, True)
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
            segmentLengthdict[1, nodes[0], nodes[0]] = 0
            segmentTortuositydict[1, nodes[0], nodes[0]] = 1
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
                segmentTortuositydict[1, sourceOnCycle, cycle[-1]] = 0
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
                    segmentTortuositydict[nthCycle, sourceOnCycle, cyclePath[-1]] = 0
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
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                segmentCountdict[sourceOnTree] = segment
                                segmentLengthdict[segment, sourceOnTree, item] = curveLength
                                segmentTortuositydict[segment, sourceOnTree, item] = curveLength / curveDisplacement
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
            # assert subGraphskeleton.number_of_edges() == 0
        # print("time taken in {} disjoint graph is {}".format(ithDisjointgraph, time.time() - starttDisjoint), "seconds")
        totalSegments = len(segmentLengthdict)
    print("time taken to calculate segments and their lengths is", time.time() - startt, "seconds")
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments
