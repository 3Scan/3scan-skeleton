import numpy as np
import networkx as nx
import itertools

import time

from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.segmentLengths import _getSort, _removeEdgesInVisitedPath


"""
   program to look up adjacent elements and calculate degree
   this dictionary can be used for graph creation
   since networkx graph based on looking up the array and the
   adjacent coordinates takes long time. create a dict
   using dictOfIndicesAndAdjacentcoordinates. Refer the following link
   https://networkx.github.io/documentation/development/reference/generated/networkx.convert.from_dict_of_lists.html
"""


def getObjWrite(imArray, pathTosave):
    """
       takes in a numpy skeleton array and converts it to a obj file
    """
    startt = time.time()
    networkxGraph = getNetworkxGraphFromarray(imArray, True)
    removeCliqueEdges(networkxGraph)
    objFile = open(pathTosave, "w")
    GraphList = nx.to_dict_of_lists(networkxGraph)
    verticesSorted = list(GraphList.keys())
    verticesSorted.sort()
    # print(verticesSorted)
    # http://www.tutorialspoint.com/python/file_writelines.html
    # http://learnpythonthehardway.org/book/ex16.html
    # vertices of the network
    strsVertices = []
    mapping = {}
    for index, line in enumerate(verticesSorted):
        mapping[line] = index + 1
        strsVertices.append("v " + " ".join(str(x) for x in line) + "\n")
    objFile.writelines(strsVertices)
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    strsSeq = []
    # lines/edges in the network
    disjointGraphs = list(nx.connected_component_subgraphs(networkGraphIntegerNodes))
    print("number of disjoint graphs is", len(disjointGraphs))
    """ check if the nodes are of one dimensions or more than one dimensions
        and intitialize the variable nodeDim used in sorting nodes later """
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        # starttDisjoint = time.time()
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            " if it is a single node"
            # print("single node")
            _removeEdgesInVisitedPath(subGraphskeleton, nodes, 1)
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
                # print("1 cycle in disjoint graph and no other components")
                # print("number of cycles in the graph is", cycleCount)
                """ if the maximum degree is equal to minimum degree it is a circle"""
                # print("single cycle")
                cycle = cycleList[0]
                cycle.append(cycle[0])
                strsSeq.append("l " + " ".join(str(x) for x in cycle) + "\n")
                _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
            elif set(degreeList) == set((1, 2)):
                # print("line segment with no tree structure")
                # print("straight line")
                """ each node is connected to one or two other nodes implies it is a line"""
                strsSeq.append("l " + " ".join(str(x) for x in nodes) + "\n")
                edges = subGraphskeleton.edges()
                subGraphskeleton.remove_edges_from(edges)
            elif cycleCount >= 1:
                # print("cycle tree graph")
                # print("cycle (more than 1) and tree like structures")
                """go through each of the cycles"""
                # print("multiple cycles")
                for nthCycle, cyclePath in enumerate(cycleList):
                    cyclePath.append(cyclePath[0])
                    strsSeq.append("l " + " ".join(str(x) for x in cyclePath) + "\n")
                    _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
                "all the cycles in the graph are checked now look for the tree characteristics in this subgraph"
                # collecting all the branch and endpoints
                branchEndpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 or v == 1]
                branchEndpoints.sort()
                listOfPerms = list(itertools.permutations(branchEndpoints, 2))
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item):
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
            else:
                "acyclic tree characteristics"""
                # print("acyclic")
                branchEndpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 or v == 1]
                branchEndpoints.sort()
                listOfPerms = list(itertools.permutations(branchEndpoints, 2))
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item):
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
            assert subGraphskeleton.number_of_edges() == 0
    objFile.writelines(strsSeq)
    print("obj file write took %0.3f seconds" % (time.time() - startt))
    # Close opend file
    objFile.close()


def getObjWriteEdges(imArray, pathTosave):
    startt = time.time()
    networkxGraph = getNetworkxGraphFromarray(imArray, True)
    removeCliqueEdges(networkxGraph)
    objFile = open(pathTosave, "w")
    GraphList = nx.to_dict_of_lists(networkxGraph)
    verticesSorted = list(GraphList.keys())
    _getSort(verticesSorted, 3)
    # http://www.tutorialspoint.com/python/file_writelines.html
    # http://learnpythonthehardway.org/book/ex16.html
    # vertices of the network
    strsVertices = []
    mapping = {}
    for index, line in enumerate(verticesSorted):
        mapping[line] = index + 1
        strsVertices.append("v " + " ".join(str(x) for x in line) + "\n")
    objFile.writelines(strsVertices)
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    edgesList = networkGraphIntegerNodes.edges()
    strsEdges = []
    for edge in edgesList:
        strsEdges.append("l " + " ".join(str(x) for x in edge) + "\n")
        # print(strs)
    objFile.writelines(strsEdges)
    print("obj file write took %0.3f seconds" % (time.time() - startt))
    objFile.close()


if __name__ == '__main__':
    # read points into array
    truthCase = np.load("/home/pranathi/Downloads/twodimageslices/output/Skeleton.npy")
    groundTruth = np.load("/home/pranathi/Downloads/twodimageslices/output/Skeleton-gt.npy")
    getObjWrite(truthCase, "/media/pranathi/DATA/PV_T.obj")
    getObjWrite(groundTruth, "/media/pranathi/DATA/SPV_GT.obj")
