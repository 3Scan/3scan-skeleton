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
        networkxGraph = removeCliqueEdges(networkxGraph)
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
            segmentdict[nodes[0]] = [0, 0, 0]
            disjointgraphDict[ithDisjointgraph] = [0, 0, 0, 0, 0]
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
            elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                """ doesn't have points with degree greater than 2, a straight line or a dichotomous tree"""
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
                    visitedSources = []; pathLengths = [];
                    totalSegmentLengths = []
                    for sourceOnTree, item in listOfPerms:
                        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                            for simplePath in simplePaths:
                                if len(list(set(branchpoints) & set(simplePath))) == 2:
                                    pathLength = _getDistanceBetweenPointsInpath(simplePath, 1)
                                    if sourceOnTree not in visitedSources:
                                        "check if the same source has multiple segments"
                                        segmentdict[sourceOnTree] = {}
                                        pathLengths = []
                                        segmentdict[sourceOnTree][0] = 1
                                    else:
                                        segmentdict[sourceOnTree][0] = segmentdict[sourceOnTree][0] + 1
                                    pathLengths.append(pathLength)
                                    totalSegmentLengths.append(pathLength)
                                    segmentdict[sourceOnTree] = [segmentdict[sourceOnTree][0], pathLengths, [1] * len(pathLengths)]
                                    visitedSources.append(sourceOnTree)
                                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                    disjointgraphDict[ithDisjointgraph] = [len(pathLengths), max(totalSegmentLengths), mean(totalSegmentLengths), 1, 1]
                else:
                    sourceOnLine = nodes[0]
                    pathLength = _getDistanceBetweenPointsInpath(nodes, 0)
                    segmentdict[sourceOnLine] = [1, pathLength, 1]
                    disjointgraphDict[ithDisjointgraph] = [1, pathLength, pathLength, 1, 1]
                    edges = subGraphskeleton.edges()
                    subGraphskeleton.remove_edges_from(edges)
            elif cycleCount >= 1:
                visitedSources = []
                pathLengths = []
                totalSegmentLengths = []; totalSegmentsTortuosity = []
                """go through each of the cycles and find the lengths, set tortuosity to NaN represented by zero here(circle)"""
                for nthCycle, cyclePath in enumerate(cycleList):
                    sourceOnCycle = cyclePath[0]
                    pathLength = _getDistanceBetweenPointsInpath(cyclePath, 1)
                    if sourceOnCycle not in visitedSources:
                        "check if the same source has multiple loops/cycles"
                        segmentdict[sourceOnCycle] = {}
                        segmentdict[sourceOnCycle][0] = 1
                        pathLengths = []
                    else:
                        segmentdict[sourceOnCycle][0] += 1
                    pathLengths.append(pathLength)
                    totalSegmentLengths.append(pathLength)
                    segmentdict[sourceOnCycle] = [segmentdict[sourceOnCycle][0], pathLengths, [0] * len(pathLengths)]
                    visitedSources.append(sourceOnCycle)
                    _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
                totalSegmentsTortuosity = [0] * len(cycleList)
                "all the cycles in the graph are checked now look for the tree characteristics in this subgraph"
                # collecting all the branch and endpoints
                # collecting all the branch and endpoints
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
                # get a list of permutations with branch points - points with degree greater than 2
                # as the starting vertex and end points - points with degree equal to one as the
                # ending vertex
                listOfPerms = list(itertools.permutations(branchpoints, 2))
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            if len(list(set(branchpoints) & set(simplePath))) == 2:
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                tortuosity = curveLength / curveDisplacement
                                if sourceOnTree not in visitedSources:
                                    "check if the same source has multiple segments"
                                    segmentdict[sourceOnTree] = {}
                                    segmentdict[sourceOnTree][0] = 1; lengthList = []; tortuosityList = [];
                                else:
                                    segmentdict[sourceOnTree][0] = segmentdict[sourceOnTree][0] + 1
                                    lengthList.insert(0, segmentdict[sourceOnTree][1]); tortuosityList.insert(0, segmentdict[sourceOnTree][2])
                                lengthList.append(curveLength); tortuosityList.append(tortuosity)
                                totalSegmentLengths.append(curveLength)
                                totalSegmentsTortuosity.append(tortuosity)
                                visitedSources.append(sourceOnTree)
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                                segmentdict[sourceOnTree] = [segmentdict[sourceOnTree][0], lengthList, tortuosityList]
                totalSegmentsinDisjoint = len(totalSegmentLengths)
                disjointgraphDict[ithDisjointgraph] = [totalSegmentsinDisjoint, max(totalSegmentLengths), mean(totalSegmentLengths), max(totalSegmentsTortuosity), mean(totalSegmentsTortuosity)]
            else:
                "acyclic tree characteristics"
                totalSegmentLengths = []; totalSegmentsTortuosity = []
                # collecting all the branch and endpoints
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
                # get a list of permutations with branch points - points with degree greater than 2
                # as the starting vertex and end points - points with degree equal to one as the
                # ending vertex
                listOfPerms = list(itertools.permutations(branchpoints, 2))
                visitedSources = []; lengthList = []; tortuosityList = [];
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            if len(list(set(branchpoints) & set(simplePath))) == 2:
                                curveLength = _getDistanceBetweenPointsInpath(simplePath)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                                tortuosity = curveLength / curveDisplacement
                                if sourceOnTree not in visitedSources:
                                    "check if the same source has multiple segments"
                                    segmentdict[sourceOnTree] = {}
                                    segmentdict[sourceOnTree][0] = 1; lengthList = []; tortuosityList = [];
                                else:
                                    segmentdict[sourceOnTree][0] += 1
                                    lengthList.insert(segmentdict[0, sourceOnTree][1]); tortuosityList.insert(0, segmentdict[sourceOnTree][2])
                                lengthList.append(curveLength)
                                tortuosityList.append(tortuosity)
                                totalSegmentLengths.append(curveLength)
                                totalSegmentsTortuosity.append(tortuosity)
                                visitedSources.append(sourceOnTree)
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
                                segmentdict[sourceOnTree] = [segmentdict[sourceOnTree][0], lengthList, tortuosityList]
                totalSegmentsinDisjoint = len(totalSegmentLengths)
                disjointgraphDict[ithDisjointgraph] = [totalSegmentsinDisjoint, max(totalSegmentLengths), mean(totalSegmentLengths), max(totalSegmentsTortuosity), mean(totalSegmentsTortuosity)]
            assert subGraphskeleton.number_of_edges() == 0
    print("time taken to calculate segments and lengths in a disjoint graph is %0.3f seconds" % (time.time() - startt))
    return segmentdict, disjointgraphDict


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def xlsxWrite(listOfDicts, path):

    dictR = listOfDicts[0]
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    row = 0
    for key in list(dictR.keys()):  # for each of the nodes with radius
        row += 1; col = 0
        worksheet.write(row, col, str(key))
        for i in range(0, 4):    # increment a column and copy the corresponding lengths and tortuosities at the branch node
            d = listOfDicts[i]
            if i not in [0, 1]:   # for dictionaries with a key that includes end points
                listOfKeys = [allKeys for allKeys in list(d.keys()) if key == tuple(list(allKeys[1]))]
                visKeys = []
                if len(listOfKeys) > 1:
                    listOfVals = []
                    for j in listOfKeys:  # multiple segments at the same branch node
                        listOfVals.append(d[j])
                        visKeys.append(j)
                    worksheet.write(row, col + i + 1, str(listOfVals))
                elif key not in visKeys:  # if only one branch at the node
                    for allKeys in list(d.keys()):
                        if key == tuple(list(allKeys[1])):
                            worksheet.write(row, col + i + 1, d[allKeys])
            else:
                worksheet.write(row, col + i + 1, d[key])
    workbook.close()


def excelWrite():
    from skeleton.segmentLengths import getSegmentsAndLengths
    from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline
    shskel = np.load(input("enter a path to shortest path skeleton volume------"))
    boundaryIm = np.load(input("enter a path to boundary of thresholded volume------"))
    path = input("enter a path to save analysis xlsx file at")
    segmentdict, disjointgraphDict = getStatsDisjoint(shskel)
    d1, d2, d3, t = getSegmentsAndLengths(shskel)
    d, di = getRadiusByPointsOnCenterline(shskel, boundaryIm)
    dictR = {your_key: d[your_key] for your_key in d1.keys()}
    d = {}
    for keys in list(dictR.keys()):
        d[keys] = (str(round(dictR[keys], 2)), str(d1[keys]))
        d[keys] = '   '.join(d[keys])
    listOfDicts = [dictR, d1, d2, d3]
    xlsxWrite(listOfDicts, path)


if __name__ == '__main__':
    excelWrite()
    # from mayavi import mlab
    # # to plot text on an mlab figure
    # for coord in list(d.keys()):
    #     x = coord[0]; y = coord[1]; z = coord[2];
    #     mlab.text3d(x, y, z, d[coord], color=(0, 0, 0), scale=2.0)
