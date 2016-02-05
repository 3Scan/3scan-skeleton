import itertools
import time
import xlsxwriter

import numpy as np
import networkx as nx
from statistics import mean

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.segmentLengths import _getDistanceBetweenPointsInpath, _removeEdgesInVisitedPath


def plotGraphWithCount(nxG, dlinecount):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(nxG)
    nx.draw_networkx_nodes(nxG, pos, nodelist=list(dlinecount.keys()))
    nx.draw_networkx_edges(nxG, pos)
    nx.draw_networkx_labels(nxG, pos, dlinecount)
    plt.axis('off')
    plt.show()


def getStatsDisjoint(imArray, skelOrNot=True, arrayOrNot=True):
    """
       return   segmentdict[sourceOnLine] = [number of segments attached to the source, [list of all the lengths]] , [list of all tortuosity]]]
                disjointgraphDict[ithDisjointgraph] = [number of segments in the disjoint graph,
                maximum of all pathLengths in disjointgraph, average of all pathLengths in disjointgraph, max(tortuosityList), mean(tortuosityList)]
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
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        # starttDisjoint = time.time()
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            " if it is a single node"
            segmentdict[nodes[0]] = [1, 0, 0]
            disjointgraphDict[ithDisjointgraph] = [1, 0, 0, 0, 0]
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
                tortuosity to infinity (NaN) represented by zero here"""
                cycle = cycleList[0]
                sourceOnCycle = cycle[0]
                pathLength = _getDistanceBetweenPointsInpath(cycle, 1)
                segmentdict[sourceOnCycle] = [1, pathLength, 0]
                disjointgraphDict[ithDisjointgraph] = [1, pathLength, pathLength, 0, 0]
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
                    visitedSources = []; pathLengths = [];
                    segmentOnSamesource = 1
                    for sourceOnTree, item in listOfPerms:
                        if nx.has_path(subGraphskeleton, sourceOnTree, item):
                            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                            for simplePath in simplePaths:
                                pathLength = _getDistanceBetweenPointsInpath(simplePath, 1)
                                pathLengths.append(pathLength)
                                if sourceOnTree not in visitedSources:
                                    "check if the same source has multiple segments"
                                    segmentdict[sourceOnTree] = [1, pathLengths, [1] * len(pathLengths)]
                                else:
                                    segmentOnSamesource += 1
                                segmentdict[sourceOnCycle] = [segmentOnSamesource, pathLengths, [1] * len(pathLengths)]
                                visitedSources.append(sourceOnTree)
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                    disjointgraphDict[ithDisjointgraph] = [len(pathLengths), max(pathLength), mean(pathLength), 1, 1]
                else:
                    sourceOnLine = nodes[0]
                    pathLength = _getDistanceBetweenPointsInpath(nodes, 0)
                    segmentdict[sourceOnLine] = [1, pathLength, 1]
                    disjointgraphDict[ithDisjointgraph] = [1, pathLength, pathLength, 1, 1]
                    _removeEdgesInVisitedPath(subGraphskeleton, nodes, 0)
            elif cycleCount >= 1:
                visitedSources = []
                pathLengths = []
                cycleOnSamesource = 0
                totalSegmentLengths = []; totalSegmentsTortuosity = []
                """go through each of the cycles and find the lengths, set tortuosity to NaN represented by zero here(circle)"""
                for nthCycle, cyclePath in enumerate(cycleList):
                    sourceOnCycle = cyclePath[0]
                    pathLength = _getDistanceBetweenPointsInpath(cyclePath, 1)
                    totalSegmentLengths.append(pathLength)
                    if sourceOnCycle not in visitedSources:
                        "check if the same source has multiple loops/cycles"
                        segmentdict[sourceOnCycle] = [1, pathLength, [0] * len(pathLengths)];
                        pathLengths = []
                    else:
                        pathLengths.append(pathLength)
                        cycleOnSamesource += 1
                        segmentdict[sourceOnCycle] = [cycleOnSamesource, pathLengths, [0] * len(pathLengths)]
                    visitedSources.append(sourceOnCycle)
                    _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
                tortuosityList = [0] * len(cycleList)
                totalSegmentLengths.append(pathLengths)
                totalSegmentsTortuosity = [0] * len(cycleList)
                "all the cycles in the graph are checked now look for the tree characteristics in this subgraph"
                # collecting all the branch and endpoints
                totalSegmentsinDisjoint = len(cycleList)
                # collecting all the branch and endpoints
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                branchpoints.sort(); endpoints.sort();
                # get a list of permutations with branch points - points with degree greater than 2
                # as the starting vertex and end points - points with degree equal to one as the
                # ending vertex
                listOfPerms = list(itertools.product(branchpoints, endpoints))
                visitedSources = [];lengthList = []; tortuosityList = [];
                segmentOnSamesource = 1
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item):
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            curveLength = _getDistanceBetweenPointsInpath(simplePath)
                            curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                            tortuosity = curveLength / curveDisplacement
                            if sourceOnTree not in visitedSources:
                                "check if the same source has multiple segments"
                                segmentOnSamesource = 1; lengthList = []; tortuosityList = [];
                            else:
                                segmentOnSamesource += 1
                            lengthList[segmentOnSamesource] = curveLength
                            tortuosityList[segmentOnSamesource] = tortuosity
                            totalSegmentLengths.append(curveLength)
                            totalSegmentsTortuosity.append(tortuosity)
                            segmentdict[sourceOnTree] = [segmentOnSamesource, lengthList, tortuosityList
                            visitedSources.append(sourceOnTree)
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                totalSegmentsinDisjoint += len(totalSegmentLengths)
                disjointgraphDict[ithDisjointgraph] = [totalSegmentsinDisjoint, max(totalSegmentLengths), mean(totalSegmentLengths), max(totalSegmentsTortuosity), mean(totalSegmentsTortuosity)]
            else:
                "acyclic tree characteristics"
                totalSegmentLengths = []; totalSegmentsTortuosity = []
                # collecting all the branch and endpoints
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                branchpoints.sort(); endpoints.sort();
                # get a list of permutations with branch points - points with degree greater than 2
                # as the starting vertex and end points - points with degree equal to one as the
                # ending vertex
                listOfPerms = list(itertools.product(branchpoints, endpoints))
                visitedSources = [];lengthList = []; tortuosityList = [];
                segmentOnSamesource = 1
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item):
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            curveLength = _getDistanceBetweenPointsInpath(simplePath)
                            curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                            tortuosity = curveLength / curveDisplacement
                            if sourceOnTree not in visitedSources:
                                "check if the same source has multiple segments"
                                segmentOnSamesource = 1; lengthList = []; tortuosityList = [];
                            else:
                                segmentOnSamesource += 1
                            lengthList[segmentOnSamesource] = curveLength
                            tortuosityList[segmentOnSamesource] = tortuosity
                            totalSegmentLengths.append(curveLength)
                            totalSegmentsTortuosity.append(tortuosity)
                            segmentdict[sourceOnTree] = [segmentOnSamesource, lengthList, tortuosityList
                            visitedSources.append(sourceOnTree)
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                totalSegmentsinDisjoint = len(totalSegmentLengths)
                disjointgraphDict[ithDisjointgraph] = [totalSegmentsinDisjoint, max(totalSegmentLengths), mean(totalSegmentLengths), max(totalSegmentsTortuosity), mean(totalSegmentsTortuosity)]
            assert subGraphskeleton.number_of_edges() == 0
    print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - startt))
    return segmentdict, disjointgraphDict


def xlsxWrite(dictionary):

    workbook = xlsxwriter.Workbook('data.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    for key in dictionary.keys():
        row += 1
        worksheet.write(row, col, key)
        for item in dictionary[key]:
            worksheet.write(row, col + 1, item)
            row += 1

    workbook.close()


if __name__ == '__main__':
    shskel = np.load("/home/pranathi/Downloads/shortestPathSkel.npy")
    segmentdict, disjointgraphDict = getStatsDisjoint(shskel)
