import networkx as nx
import time

from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.segmentLengths import getSegmentsAndLengths


def cycle(n):
    networkxGraph = nx.cycle_graph(n)
    removeCliqueEdges(networkxGraph)
    print("nth cycle", n)
    startt = time.time()
    dcyclecount, dcyclelength, segmentTortuositycycle, totalSegmentsDonut = getSegmentsAndLengths(networkxGraph)
    timeTaken = time.time() - startt
    print(n, totalSegmentsDonut)
    assert totalSegmentsDonut == 1
    densityOfgraph = nx.density(networkxGraph)
    n = networkxGraph.number_of_nodes()
    return timeTaken, densityOfgraph, n


def tree(n):
    networkxGraph = nx.balanced_tree(2, n)
    removeCliqueEdges(networkxGraph)
    print("nth tree", n)
    startt = time.time()
    dcyclecount, dcyclelength, segmentTortuositycycle, totalSegmentsDonut = getSegmentsAndLengths(networkxGraph)
    timeTaken = time.time() - startt
    print(n, totalSegmentsDonut)
    assert totalSegmentsDonut == networkxGraph.number_of_edges()
    densityOfgraph = nx.density(networkxGraph)
    n = networkxGraph.number_of_nodes()
    return timeTaken, densityOfgraph, n


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    timeTree = np.zeros((6))
    densityTree = np.zeros((6))
    numberOfNodesarraytree = np.zeros((6))
    for numberOfNodes in range(3, 9):
        timeTaken, d, n = tree(numberOfNodes)
        timeTree[numberOfNodes - 3] = timeTaken
        densityTree[numberOfNodes - 3] = d
        numberOfNodesarraytree[numberOfNodes - 3] = n
    plt.title("nodes in a tree and time taken to find number of segments and lengths")
    plt.xlabel("number of nodes in the tree")
    plt.ylabel("time taken to trace and find segement features in seconds")
    plt.plot(numberOfNodesarraytree, timeTree, 'r')
    plt.show()
