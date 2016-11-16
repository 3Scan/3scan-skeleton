import itertools
import time

import numpy as np
import networkx as nx


"""
program to remove 3 vertex cliques - Remove the diagonal edge
that forms non-existing triangular loop in the graph
"""


def removeCliqueEdges(networkxGraph):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    networkxGraph : Networkx graph
        graph to remove cliques from

    Returns
    -------
    networkxGraphAfter : Networkx graph
        graph with 3 vertex clique edges removed

    Notes
    ------
    Removes the longest edge in a 3 Vertex cliques,
    Special case edges are the edges with equal
    lengths that form the 3 vertex clique.
    Doesn't deal with any other cliques
    """
    start = time.time()
    networkxGraphAfter = networkxGraph.copy()
    cliques = nx.find_cliques_recursive(networkxGraph)
    # all the nodes/vertices of 3 cliques
    threeVErtexCliques = [clq for clq in cliques if len(clq) == 3]
    if len(list(threeVErtexCliques)) != 0:
        combinationEdges = [list(itertools.combinations(clique, 2)) for clique in threeVErtexCliques]
        subGraphEdgelengths = []
        # different combination of edges in the cliques and their lengths
        for combinationEdge in combinationEdges:
            subGraphEdgelengths.append([np.sum((np.array(item[0]) - np.array(item[1])) ** 2)
                                        for item in combinationEdge])
        cliqueEdges = []
        # clique edges to be removed are collected here
        # the edges with maximum edge length
        for mainDim, item in enumerate(subGraphEdgelengths):
            if len(set(item)) != 1:
                for subDim, length in enumerate(item):
                    if length == max(item):
                        cliqueEdges.append(combinationEdges[mainDim][subDim])
            else:
                specialCase = combinationEdges[mainDim]
                diffOfEdges = []
                for numSpcledges in range(0, 3):
                    source = list(specialCase[numSpcledges][0])
                    target = list(specialCase[numSpcledges][1])
                    diffOfEdges.append([i - j for i, j in zip(source, target)])
                for index, val in enumerate(diffOfEdges):
                    if val[0] == 0:
                        subDim = index
                        cliqueEdges.append(combinationEdges[mainDim][subDim])
                        break
        networkxGraphAfter.remove_edges_from(cliqueEdges)
        print("time taken to remove cliques is %0.2f seconds" % (time.time() - start))
    else:
        networkxGraphAfter = networkxGraph
    return networkxGraphAfter
