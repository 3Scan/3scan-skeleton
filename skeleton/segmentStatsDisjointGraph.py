import time
import xlsxwriter

import numpy as np
import networkx as nx

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.segmentLengths import _getSort, _getDistanceBetweenPointsInpath, _removeEdgesInVisitedPath


def plotGraphWithCount(nxG, dlinecount):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(nxG)
    nx.draw_networkx_nodes(nxG, pos, nodelist=list(dlinecount.keys()))
    nx.draw_networkx_edges(nxG, pos)
    nx.draw_networkx_labels(nxG, pos, dlinecount)
    plt.axis('off')
    plt.show()


def getSegmentsAndLengths(imArray, skelOrNot=True, arrayOrNot=True):
    """
       return   segmentdict[sourceOnLine] = [number of segments attached to the source, [list of all the lengths]] , [list of all tortuosity]]]
                disjointgraphDict[ithDisjointgraph] = [number of segments in the disjoint graph,
                maximum of all pathLengths in disjointgraph, average of all pathLengths in disjointgraph]
    """
    if arrayOrNot is False:
        networkxGraph = imArray
    else:
        networkxGraph = getNetworkxGraphFromarray(imArray, skelOrNot)
        removeCliqueEdges(networkxGraph)
    assert networkxGraph.number_of_selfloops() == 0
    # intitialize all the common variables
    startt = time.time()
    segmentdict = {}
    disjointgraphDict = {}
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
            segmentdict[nodes[0]] = [1, 0, 1]
            disjointgraphDict[ithDisjointgraph] = [1, 1, 1]
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
                tortuosity to infinity (NaN) represented by zero here"""
                cycle = cycleList[0]
                sourceOnCycle = cycle[0]
                segmentdict[sourceOnCycle] = [1, 0, 1]
                pathLength = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentdict[sourceOnCycle] = [1, pathLength, 0]
                disjointgraphDict[ithDisjointgraph] = [1, pathLength, pathLength]
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)):
                # print("line segment with no tree structure")
                """ each node is connected to one or two other nodes implies it is a line,
                set tortuosity to 1"""
                sourceOnLine = nodes[0]
                pathLength = _getDistanceBetweenPointsInpath(nodes, 0)
                segmentdict[sourceOnLine] = [1, pathLength, 0]
                disjointgraphDict[ithDisjointgraph] = [1, pathLength, pathLength]
                _removeEdgesInVisitedPath(subGraphskeleton, nodes, 0)
            elif cycleCount >= 1:
                # print("cycle (more than 1) and tree like structures")
                visitedSources = []
                cycleOnSamesource = 0
                """go through each of the cycles and find the lengths, set tortuosity to NaN represented by zero here(circle)"""
                for nthCycle, cyclePath in enumerate(cycleList):
                    sourceOnCycle = cyclePath[0]
                    pathLength = _getDistanceBetweenPointsInpath(cyclePath, 1)
                    pathLengths.append(pathLength)
                    visitedSources.append(sourceOnCycle)
                    if sourceOnCycle not in visitedSources:
                        "check if the same source has multiple loops/cycles"
                        segmentdict[sourceOnCycle] = [1, pathLengths, [0] * len(pathLengths)]
                    else:
                        cycleOnSamesource += 1
                        segmentdict[sourceOnCycle] = [cycleOnSamesource, pathLengths, [0] * len(pathLengths)]
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
            # assert subGraphskeleton.number_of_edges() == 0
        # print("time taken in {} disjoint graph is {}".format(ithDisjointgraph, time.time() - starttDisjoint), "seconds")
    assert sum(segmentCountdict.values()) == len(segmentTortuositydict) == len(segmentLengthdict)
    totalSegments = len(segmentLengthdict)
    print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - startt))
    return segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments


def xlsxWrite():

    workbook = xlsxwriter.Workbook('data.xlsx')
    worksheet = workbook.add_worksheet()

    d = {'a': ['e1', 'e2', 'e3'], 'b': ['e1', 'e2'], 'c': ['e1']}
    row = 0
    col = 0

    for key in d.keys():
        row += 1
        worksheet.write(row, col, key)
        for item in d[key]:
            worksheet.write(row, col + 1, item)
            row += 1

    workbook.close()
    pass
