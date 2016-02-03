import numpy as np
import networkx as nx
import itertools

import time

from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.segmentLengths import _removeEdgesInVisitedPath


"""
    write an array to a wavefront obj file - time takes upto 3 minutes for a 512 * 512 * 512 array,
    input array can be either 3D or 2D. Function needs to be called with an array you want to save
    as a .obj file and the location on obj file
"""


def getObjWrite(imArray, pathTosave):
    """
       takes in a numpy array and converts it to a obj file and writes it to pathTosave
    """
    startt = time.time()  # for calculating time taken to write to an obj file
    networkxGraph = getNetworkxGraphFromarray(imArray, True)  # converts array to a networkx graph(based on non zero coordinates and the adjacent nonzeros)
    removeCliqueEdges(networkxGraph)  # remove cliques in the graph
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    GraphList = nx.to_dict_of_lists(networkxGraph)  # convert the graph to a dictionary with keys as nodes and list of adjacent nodes as the values
    verticesSorted = list(GraphList.keys())  # list and sort the keys so they are geometrically in the same order when writing to an obj file as (l_prefix paths)
    verticesSorted.sort()
    strsVertices = []; mapping = {}  # initialize variables for writing string of vertex v followed by x, y, x coordinates in the obj file
    #  for each of the sorted vertices
    for index, vertex in enumerate(verticesSorted):
        mapping[vertex] = index + 1  # a mapping to transform the vertices (x, y, z) to indexes (beginining with 1)
        strsVertices.append("v " + " ".join(str(dim) for dim in vertex) + "\n")  # add strings of vertices to obj file
    objFile.writelines(strsVertices)  # write strings to obj file
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    strsSeq = []
    # line prefixes for the obj file
    disjointGraphs = list(nx.connected_component_subgraphs(networkGraphIntegerNodes))
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            continue
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
            """ if the maximum degree is equal to minimum degree it is a circle"""
            cycle = cycleList[0]
            cycle.append(cycle[0])
            strsSeq.append("l " + " ".join(str(x) for x in cycle) + "\n")
            _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
        elif set(degreeList) == set((1, 2)):
            """ each node is connected to one or two other nodes implies it is a line"""
            strsSeq.append("l " + " ".join(str(x) for x in nodes) + "\n")
            _removeEdgesInVisitedPath(subGraphskeleton, nodes, 0)
        elif cycleCount >= 1:
            """go through each of the cycles"""
            for nthCycle, cyclePath in enumerate(cycleList):
                cyclePath.append(cyclePath[0])
                strsSeq.append("l " + " ".join(str(x) for x in cyclePath) + "\n")
                _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
            "all the cycles in the graph are checked now look for the tree characteristics in this subgraph"
            # collecting all the branch and endpoints
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
            endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
            branchpoints.sort(); endpoints.sort();
            listOfPerms = list(itertools.product(branchpoints, endpoints))
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(subGraphskeleton, sourceOnTree, item):
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                    for simplePath in simplePaths:
                        strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                        _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
        else:
            "acyclic tree characteristics"""
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v > 2]
            endpoints = [k for (k, v) in nodeDegreedict.items() if v == 1]
            branchpoints.sort(); endpoints.sort();
            listOfPerms = list(itertools.product(branchpoints, endpoints))
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


if __name__ == '__main__':
    # read points into array
    truthCase = np.load("/home/pranathi/Downloads/twodimageslices/output/Skeleton.npy")
    groundTruth = np.load("/home/pranathi/Downloads/twodimageslices/output/Skeleton-gt.npy")
    getObjWrite(truthCase, "/media/pranathi/DATA/PV_T.obj")
    getObjWrite(groundTruth, "/media/pranathi/DATA/SPV_GT.obj")
