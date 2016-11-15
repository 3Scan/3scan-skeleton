import numpy as np

from skeleton.networkxGraphFromArray import getNetworkxGraphFromArray
from skeleton.skeleton_testlib import getTinyLoopWithBranches, getDisjointCrosses, getSingleVoxelLine


def getGraphsWithCliques():
    # returns graphs with cliques and nonzeros
    samples = [getTinyLoopWithBranches(), getDisjointCrosses(), getSingleVoxelLine()]
    for sample in samples:
        G = getNetworkxGraphFromArray(sample)
        yield G, set(map(tuple, np.transpose(np.nonzero(sample))))


def test_sameNodesInGraph():
    # no nodes in the input array are missing in the networkx graph
    for G, nonZeros in getGraphsWithCliques():
        assert set(G.nodes()) == nonZeros, "different nodes in input array and graph"
