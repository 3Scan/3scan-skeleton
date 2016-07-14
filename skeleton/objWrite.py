import numpy as np
import networkx as nx
import itertools

import time

from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline
from skeleton.segmentStats import _removeEdgesInVisitedPath


"""
    write an array to a wavefront obj file - time takes upto 3 minutes for a 512 * 512 * 512 array,
    input array can be either 3D or 2D. Function needs to be called with an array you want to save
    as a .obj file and the location on obj file - primarily written for netmets (comparison of 2 networks)
    Link to the software - http://stim.ee.uh.edu/resources/software/netmets/
"""


def getObjWrite(imArray, pathTosave, aspectRatio=None):
    """
       takes in a numpy array and converts it to a obj file and writes it to pathTosave
    """
    startt = time.time()  # for calculating time taken to write to an obj file
    if type(imArray) == np.ndarray:
        networkxGraph = getNetworkxGraphFromarray(imArray)  # converts array to a networkx graph(based on non zero coordinates and the adjacent nonzeros)
        networkxGraph = removeCliqueEdges(networkxGraph)  # remove cliques in the graph
    else:
        networkxGraph = imArray
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    GraphList = nx.to_dict_of_lists(networkxGraph)  # convert the graph to a dictionary with keys as nodes and list of adjacent nodes as the values
    verticesSorted = list(GraphList.keys())  # list and sort the keys so they are geometrically in the same order when writing to an obj file as (l_prefix paths)
    verticesSorted.sort()
    mapping = {}  # initialize variables for writing string of vertex v followed by x, y, x coordinates in the obj file
    #  for each of the sorted vertices
    strsVertices = []
    for index, vertex in enumerate(verticesSorted):
        mapping[vertex] = index + 1  # a mapping to transform the vertices (x, y, z) to indexes (beginining with 1)
        if aspectRatio is not None:
            strsVertices.append("v " + " ".join(str(vertex[i] * (1 / aspect)) for i, aspect in zip([1, 0, 2], aspectRatio)) + "\n")  # add strings of vertices to obj file
    objFile.writelines(strsVertices)  # write strings to obj file
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    getWriteLinesInObj(networkGraphIntegerNodes, objFile)
    print("obj file write took %0.3f seconds" % (time.time() - startt))


def getWriteLinesInObj(networkGraphIntegerNodes, objFile):
    strsSeq = []
    visitedPaths = []
    sortedSegments = []
    # line prefixes for the obj file
    disjointGraphs = list(nx.connected_component_subgraphs(networkGraphIntegerNodes))
    totalSteps = len(disjointGraphs)
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) != 1:
            """ if there are more than one nodes decide what kind of subgraph it is
                if it has cycles alone, or a straight line or a directed cyclic/acyclic graph"""
            cycleList = nx.cycle_basis(subGraphskeleton)
            cycleCount = len(cycleList)
            nodeDegreedict = nx.degree(subGraphskeleton)
            degreeList = list(nodeDegreedict.values())
            endPointdegree = min(degreeList)
            branchPointdegree = max(degreeList)
            if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount == 1:
                """ if the maximum degree is equal to minimum degree it is a circle, set
                tortuosity to infinity (NaN) set to zero here"""
                cycle = cycleList[0]
                _singleCycle(strsSeq, cycle, subGraphskeleton)
            elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                """ disjoint line or a bent line at 45 degrees appearing as dichtonomous tree but an error due to
                    improper binarization, so remove them and do not account for statistics"""
                """ straight line or dichtonomous tree"""
                edges = subGraphskeleton.edges()
                subGraphskeleton.remove_edges_from(edges)
            else:
                """ cyclic or acyclic tree """
                if cycleCount != 0:
                    _cyclicTree(strsSeq, subGraphskeleton, cycleList, nodeDegreedict, visitedPaths, sortedSegments)
                "sorted branch and end points in trees"
                branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 and v != 1]
                endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
                branchpoints.sort()
                endpoints.sort()
                _tree(strsSeq, subGraphskeleton, branchpoints, endpoints)
                if subGraphskeleton.number_of_edges() != 0:
                    _branchToBranch(strsSeq, subGraphskeleton, branchpoints)
            progress = int((100 * ithDisjointgraph) / totalSteps)
            print("object writing in progress {}% \r".format(progress), end="", flush=True)
            assert subGraphskeleton.number_of_edges() == 0
    objFile.writelines(strsSeq)
    # Close opend file
    objFile.close()


def checkBackTracing(simplePath, sortedSegments):
    check = 1
    for path in sortedSegments:
        if len(set(path) & set(simplePath)) > 2:
            check = 0
    return check


def _singleCycle(strsSeq, cycle, subGraphskeleton):
    strsSeq.append("l " + " ".join(str(x) for x in cycle) + "\n")
    _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)


def _cyclicTree(strsSeq, subGraphskeleton, cycleList, nodeDegreedict, visitedPaths, sortedSegments):
    for nthcycle, cycle in enumerate(cycleList):
        nodeDegreedictFilt = {key: value for key, value in nodeDegreedict.items() if key in cycle}
        branchpoints = [k for (k, v) in nodeDegreedictFilt.items() if v != 2 and v != 1]
        sourceOnCycle = branchpoints[0]
        if len(branchpoints) == 1:
            _singleCycle(strsSeq, cycle, subGraphskeleton)
        else:
            for point in cycle:
                if point in branchpoints:
                    if nx.has_path(subGraphskeleton, source=sourceOnCycle, target=point) and sourceOnCycle != point:
                        simplePath = nx.shortest_path(subGraphskeleton, source=sourceOnCycle, target=point)
                        sortedSegment = sorted(simplePath)
                        if sortedSegment not in sortedSegments and sum([1 for pathpoint in simplePath if pathpoint in branchpoints]) == 2 and checkBackTracing(simplePath, sortedSegments):
                            strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                            visitedPaths.append(simplePath)
                            sortedSegments.append(sortedSegment)
                    sourceOnCycle = point
    for path in visitedPaths:
        _removeEdgesInVisitedPath(subGraphskeleton, path, 0)


def _tree(strsSeq, subGraphskeleton, branchpoints, endpoints):
    listOfPerms = list(itertools.product(branchpoints, endpoints))
    for sourceOnTree, item in listOfPerms:
        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
            for simplePath in simplePaths:
                if sum([1 for point in simplePath if point in branchpoints]) == 1:
                    strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)


def _branchToBranch(strsSeq, subGraphskeleton, branchpoints):
    listOfPerms = list(itertools.permutations(branchpoints, 2))
    for sourceOnTree, item in listOfPerms:
        if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
            simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
            for simplePath in simplePaths:
                if sum([1 for point in simplePath if point in branchpoints]) == 2:
                    strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                    _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)


def getObjBranchPointsWrite(imArray, pathTosave):
    """
       takes in a numpy array and converts it to a obj file and writes it to pathTosave
    """
    if type(imArray) == np.ndarray:
        networkxGraph = getNetworkxGraphFromarray(imArray)  # converts array to a networkx graph(based on non zero coordinates and the adjacent nonzeros)
        networkxGraph = removeCliqueEdges(networkxGraph)  # remove cliques in the graph
    else:
        networkxGraph = imArray
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    nodeDegreedict = nx.degree(networkxGraph)
    branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 and v != 1]
    #  for each of the sorted vertices
    strsVertices = []
    for index, vertex in enumerate(branchpoints):
        strsVertices.append("v " + " ".join(str(vertex[i] * (1 / aspect)) for i, aspect in zip([1, 0, 2], [0.7, 0.7, 5])) + "\n")  # add strings of vertices to obj file
    objFile.writelines(strsVertices)  # write strings to obj file
    objFile.close()


def getObjWriteWithradius(imArray, pathTosave, dictOfNodesAndRadius, aspectRatio=None):
    """
       takes in a numpy array and converts it to a obj file and writes it to pathTosave
    """
    startt = time.time()  # for calculating time taken to write to an obj file
    if type(imArray) == np.ndarray:
        networkxGraph = getNetworkxGraphFromarray(imArray, True)  # converts array to a networkx graph(based on non zero coordinates and the adjacent nonzeros)
        networkxGraph = removeCliqueEdges(networkxGraph)  # remove cliques in the graph
    else:
        networkxGraph = imArray
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    GraphList = nx.to_dict_of_lists(networkxGraph)  # convert the graph to a dictionary with keys as nodes and list of adjacent nodes as the values
    verticesSorted = list(GraphList.keys())  # list and sort the keys so they are geometrically in the same order when writing to an obj file as (l_prefix paths)
    verticesSorted.sort()
    mapping = {}  # initialize variables for writing string of vertex v followed by x, y, x coordinates in the obj file
    #  for each of the sorted vertices write both v and vy followed by radius
    strsVertices = [0] * (2 * len(verticesSorted))
    for index, vertex in enumerate(verticesSorted):
        mapping[vertex] = index + 1  # a mapping to transform the vertices (x, y, z) to indexes (beginining with 1)
        if aspectRatio is not None:
            originalVertex = list(vertex)
            newVertex = [0] * len(vertex)
            newVertex[0] = originalVertex[0] * aspectRatio[0]
            newVertex[1] = originalVertex[2] * aspectRatio[1]
            newVertex[2] = originalVertex[1] * aspectRatio[2]
            vertex = tuple(newVertex)
        strsVertices[index] = "v " + " ".join(str(vertex[i - 2]) for i in range(0, len(vertex))) + "\n"  # add strings of vertices to obj file
        strsVertices[index + len(verticesSorted)] = "vt " + " " + str(dictOfNodesAndRadius[vertex]) + "\n"
    objFile.writelines(strsVertices)  # write strings to obj file
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    getWriteLinesInObj(networkGraphIntegerNodes, objFile)
    print("obj file write took %0.3f seconds" % (time.time() - startt))


if __name__ == '__main__':
    # read points into array
    skeletonIm = np.load(input("enter a path to shortest path skeleton volume------"))
    boundaryIm = np.load(input("enter a path to boundary of thresholded volume------"))
    aspectRatio = input("please enter resolution of a voxel in 3D with resolution in z followed by y and x")
    aspectRatio = [float(item) for item in aspectRatio.split(' ')]
    path = input("please enter a path to save resultant obj file with no texture coordinates")
    path2 = input("please enter a path to save resultant obj file with texture coordinates")
    dictOfNodesAndRadius, distTransformedIm = getRadiusByPointsOnCenterline(skeletonIm, boundaryIm)
    getObjWrite(skeletonIm, path, aspectRatio)
    getObjWriteWithradius(skeletonIm, path2, aspectRatio)
