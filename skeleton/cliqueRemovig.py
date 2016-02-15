import itertools
import time

import numpy as np
import networkx as nx

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray

"""
   program to remove 3 vertex cliques
"""


def removeCliqueEdges(networkxGraph):
    """
       remvoe edges of 3 vertex cliques of
       squared length 2
    """
    disjointGraphsBefore = len(list(nx.connected_component_subgraphs(networkxGraph)))
    print("number of graphs in the input networkxGraph is", disjointGraphsBefore)
    networkxGraphAfter = networkxGraph.copy()
    startt = time.time()
    cliques = nx.find_cliques_recursive(networkxGraph)
    # all the nodes/vertices of 3 cliques
    cliques2 = [clq for clq in cliques if len(clq) == 3]
    if len(list(cliques2)) == 0:
        return networkxGraph
    else:
        combEdge = [list(itertools.combinations(clique, 2)) for clique in cliques2]
        subGraphEdgelengths = []
        # different combination of edges in the cliques and their lengths
        for combedges in combEdge:
            subGraphEdgelengths.append([np.sum((np.array(item[0]) - np.array(item[1])) ** 2) for item in combedges])
        cliquEdges = []
        # clique edges to be removed are collected here
        # the edges with maximum edge length
        for mainDim, item in enumerate(subGraphEdgelengths):
            for subDim, length in enumerate(item):
                if length == max(max(list(zip(*subGraphEdgelengths)))):
                    cliquEdges.append(combEdge[mainDim][subDim])
        networkxGraphAfter.remove_edges_from(cliquEdges)
        disjointGraphsAfter = len(list(nx.connected_component_subgraphs(networkxGraphAfter)))
        print("number of graphs in the output networkxGraph after cliques are removed is", disjointGraphsAfter)
        assert networkxGraph.number_of_edges() >= networkxGraphAfter.number_of_edges()
        print("time taken to remove cliques is %0.2f seconds" % (time.time() - startt))
        if disjointGraphsBefore == disjointGraphsAfter:
            print("graph changed to remove cliques")
            return networkxGraphAfter.copy()
        else:
            print("graph unchanged")
            return networkxGraph


if __name__ == '__main__':
    shskel = np.load("/Users/3scan_editing/records/shortestPathSkel1.npy")
    networkxGraph = getNetworkxGraphFromarray(shskel)
    removeCliqueEdges(networkxGraph)
