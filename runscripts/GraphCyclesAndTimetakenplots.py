import networkx as nx
import time

import matplotlib.pyplot as plt
import numpy as np

from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.segmentStats import getSegmentStats

"""
   program to generate a balanced tree with a root node and the branches extending to the height
   https://networkx.github.io/documentation/latest/reference/generated/networkx.generators.classic.balanced_tree.html
   and generate a cycle graph with n number of nodes
   https://networkx.github.io/documentation/latest/reference/generated/networkx.generators.classic.cycle_graph.html
   To check the time taken to detect the branches in a tree and the lenght of branches to each node
   detects cycle and finds length of cycle with time complexity as O(1) - CONSTANT
   detects tree and finds length of branches with time complexity as O(nodeCount) - linear function of number of nodes
"""


def _cycle(n):
    """
       generate a cyclic graph with number of nodes n
    """
    networkxGraph = nx.cycle_graph(n)
    removeCliqueEdges(networkxGraph)
    print("nth cycle", n)
    startt = time.time()
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(networkxGraph, True)
    timeTaken = time.time() - startt
    print(n, totalSegments)
    assert totalSegments == 1
    densityOfgraph = nx.density(networkxGraph)
    n = networkxGraph.number_of_nodes()
    return timeTaken, densityOfgraph, n


def _tree(n):
    """
       generate a balanced tree with height of tree from root equal to n
    """
    networkxGraph = nx.balanced_tree(2, n)
    removeCliqueEdges(networkxGraph)
    print("nth tree", n)
    startt = time.time()
    segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(networkxGraph, True)
    timeTaken = time.time() - startt
    print(n, totalSegments)
    densityOfgraph = nx.density(networkxGraph)
    n = networkxGraph.number_of_nodes()
    return timeTaken, densityOfgraph, n

functionList = [_cycle, _tree]


def plotTreeAndTimetaken(treeOrCycle=1, heightRange=9):
    """
       plot a figure with x axis as nodes and y axis as the time
       taken to find to segment characteristics in the graph
    """
    timeTree = []
    densityTree = []
    numberOfNodesarraytree = []
    plt.ion()
    op = functionList[treeOrCycle]
    for heightOfTree in range(2, heightRange):
        timeTaken, d, n = op(heightOfTree)
        timeTree.append(timeTaken)
        densityTree.append(d)
        numberOfNodesarraytree.append(n)
    plt.title("nodes in a tree and time taken to find number of segments and lengths")
    plt.xlabel("number of nodes in the tree")
    plt.ylabel("time taken to trace and find segement features in seconds")
    plt.plot(np.array(numberOfNodesarraytree), np.array(timeTree), 'r')

if __name__ == '__main__':
    plotTreeAndTimetaken(1, 15)
