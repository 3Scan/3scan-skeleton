import networkx as nx
import numpy as np

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.segmentStatsDisjointGraph import getStatsDisjoint
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
# from skeleton.segmentStatsDisjointGraph import plotGraphWithCount

from tests.tests3DSkeletonize import getDonut

"""
   program to test if graphs created using getNetworkxGraphFromarray
   from the dictionary of the coordinate and adjacent nonzero coordinates
   after removing the cliques have the number of segments as expected
   PV TODO:Test if lengths of segments and tortuoisty of the curves as expected
"""


def getCyclesWithBranchesProtrude(size=(10, 10)):
    frame = np.zeros(size, dtype=np.uint8)
    frame[2:-2, 2:-2] = 1
    frame[4:-4, 4:-4] = 0
    frame = getSkeletonize2D(frame)
    frame[1, 5] = 1; frame[7, 5] = 1;
    sampleImage = np.zeros((3, 10, 10), dtype=np.uint8)
    sampleImage[1] = frame
    sampleGraph = getNetworkxGraphFromarray(sampleImage, True)
    removeCliqueEdges(sampleGraph)
    return sampleGraph


def getSingleVoxelLineNobranches(size=(5, 5, 5)):
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    lineGraph = getNetworkxGraphFromarray(sampleLine, True)
    removeCliqueEdges(lineGraph)
    return lineGraph


def getCycleNotree():
    donut = getDonut()
    donutGraph = getNetworkxGraphFromarray(donut, False)
    removeCliqueEdges(donutGraph)
    return donutGraph


def getTreeNoCycle2d(size=(5, 5)):
    cros = np.zeros(size, dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosGraph = getNetworkxGraphFromarray(cros, False)
    removeCliqueEdges(crosGraph)
    return crosGraph


def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    crosPairgraph = getNetworkxGraphFromarray(crosPair, True)
    removeCliqueEdges(crosPairgraph)
    return crosPairgraph


def getDisjointCyclesNoTrees2d(size=(10, 10)):
    tinyLoop = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=bool)
    multiLoop = np.zeros(size, dtype=bool)
    multiLoop[2:5, 2:5] = tinyLoop
    multiLoop[7:10, 7:10] = tinyLoop
    multiloopgraph = getNetworkxGraphFromarray(multiLoop, True)
    removeCliqueEdges(multiloopgraph)
    return multiloopgraph


def test_singlesegment():
    lineGraph = getSingleVoxelLineNobranches()
    segmentdict, disjointgraphDict = getStatsDisjoint(lineGraph, True, False)
    # plotGraphWithCount(lineGraph, dlinecount)
    assert len(segmentdict) == 1 and disjointgraphDict[0][0] == 1 and len(disjointgraphDict) == 1


def test_singlecycle():
    donutGraph = getCycleNotree()
    segmentdict, disjointgraphDict = getStatsDisjoint(donutGraph, True, False)
    # plotGraphWithCount(donutGraph, dcyclecount)
    assert len(segmentdict) == 1 and disjointgraphDict[0][0] == 1 and len(disjointgraphDict) == 1


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    segmentdict, disjointgraphDict = getStatsDisjoint(crosGraph, True, False)
    # plotGraphWithCount(crosGraph, dTreecount)
    assert disjointgraphDict[0][0] == 4 and len(disjointgraphDict) == 1


def test_disjointDoublecycle():
    multiloopgraph = getDisjointCyclesNoTrees2d()
    segmentdict, disjointgraphDict = getStatsDisjoint(multiloopgraph, True, False)
    # plotGraphWithCount(multiloopgraph, ddisjointCyclescount)
    assert len(segmentdict) == 2 and disjointgraphDict[0][0] == 1 and disjointgraphDict[1][0] == 1 and len(disjointgraphDict) == 2


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    segmentdict, disjointgraphDict = getStatsDisjoint(crosPairgraph, True, False)
    # plotGraphWithCount(crosPairgraph, dTreescount)
    assert len(segmentdict) == 8 and disjointgraphDict[0][0] == 4 and disjointgraphDict[1][0] == 4 and len(disjointgraphDict) == 2


def test_balancedtree():
    balancedTree = nx.balanced_tree(2, 1)
    segmentdict, disjointgraphDict = getStatsDisjoint(balancedTree, True, False)
    # plotGraphWithCount(balancedTree, dlinecount)
    assert len(disjointgraphDict) == 1


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    segmentdict, disjointgraphDict = getStatsDisjoint(sampleGraph, True, False)
    # plotGraphWithCount(sampleGraph, dcycleTreecount)
    assert len(segmentdict) == 3 and disjointgraphDict[0][0] == 3 and len(disjointgraphDict) == 1


def test_touchingCycles():
    diamondGraph = nx.diamond_graph()
    segmentdict, disjointgraphDict = getStatsDisjoint(diamondGraph, True, False)
    # plotGraphWithCount(diamondGraph, dlinecount)
    assert len(segmentdict) == 1 and disjointgraphDict[0][0] == 2 and len(disjointgraphDict) == 1
