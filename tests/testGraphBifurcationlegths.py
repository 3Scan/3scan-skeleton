import networkx as nx
import numpy as np

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.segmentLengths import getSegmentsAndLengths
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.BifurcatedsegmentLengths import getBifurcatedSegmentsAndLengths
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
    lineGraph = removeCliqueEdges(lineGraph)
    return lineGraph


def getCycleNotree():
    donut = getDonut()
    donutGraph = getNetworkxGraphFromarray(donut, False)
    donutGraph = removeCliqueEdges(donutGraph)
    return donutGraph


def getTreeNoCycle2d(size=(7, 7)):
    cros = np.zeros(size, dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    cros[4, 3] = 1
    crosGraph = getNetworkxGraphFromarray(cros, False)
    crosGraph = removeCliqueEdges(crosGraph)
    return crosGraph


def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    crosPair = np.zeros(size, dtype=np.uint8)
    crosPair[0, 0:7, 0:7] = cros
    crosPair[7, 7:14, 7:14] = cros
    crosPairgraph = getNetworkxGraphFromarray(crosPair, True)
    crosPairgraph = removeCliqueEdges(crosPairgraph)
    return crosPairgraph


def getDisjointCyclesNoTrees2d(size=(10, 10)):
    tinyLoop = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=bool)
    multiLoop = np.zeros(size, dtype=bool)
    multiLoop[2:5, 2:5] = tinyLoop
    multiLoop[7:10, 7:10] = tinyLoop
    multiloopgraph = getNetworkxGraphFromarray(multiLoop, True)
    multiloopgraph = removeCliqueEdges(multiloopgraph)
    return multiloopgraph


def test_singlesegment():
    lineGraph = getSingleVoxelLineNobranches()
    dlinecount, dlinelength, segmentTortuosityline, totalSegmentsLine = getSegmentsAndLengths(lineGraph, True, False)
    # plotGraphWithCount(lineGraph, dlinecount)
    assert totalSegmentsLine == 1


def test_singlecycle():
    donutGraph = getCycleNotree()
    dcyclecount, dcyclelength, segmentTortuositycycle, totalSegmentsDonut = getSegmentsAndLengths(donutGraph, True, False)
    # plotGraphWithCount(donutGraph, dcyclecount)
    assert totalSegmentsDonut == 1


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    dcycleTreecount, dcycleTreelength, segmentTortuositycycletree, totalSegmentsSampleGraph = getSegmentsAndLengths(sampleGraph, True, False)
    # plotGraphWithCount(sampleGraph, dcycleTreecount)
    assert totalSegmentsSampleGraph == 3


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    dTreecount, dTreelength, segmentTortuositytree, totalSegmentsTree = getSegmentsAndLengths(crosGraph, True, False)
    # plotGraphWithCount(crosGraph, dTreecount)
    assert totalSegmentsTree == 4


def test_disjointDoublecycle():
    multiloopgraph = getDisjointCyclesNoTrees2d()
    disjointCyclescount, ddisjointCycleslength, segmentTortuositycycles, totalSegmentsDisjointCycles = getSegmentsAndLengths(multiloopgraph, True, False)
    # plotGraphWithCount(multiloopgraph, ddisjointCyclescount)
    assert totalSegmentsDisjointCycles == 2


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    dTreescount, dTreeslength, segmentTortuositytrees, totalSegmentsTrees = getSegmentsAndLengths(crosPairgraph, True, False)
    # plotGraphWithCount(crosPairgraph, dTreescount)
    assert totalSegmentsTrees == 8


def test_balancedtree():
    balancedTree = nx.balanced_tree(2, 1)
    dlinecountbaltree, dlinebaltree, segmentTortuositybaltree, totalSegmentsBalancedTree = getSegmentsAndLengths(balancedTree, True, False)
    # plotGraphWithCount(balancedTree, dlinecount)
    assert totalSegmentsBalancedTree == 2


def test_touchingCycles():
    diamondGraph = nx.diamond_graph()
    dcyclescount, dcycleslength, segmentTortuositycycles, totalSegmentsCycles = getSegmentsAndLengths(diamondGraph, True, False)
    # plotGraphWithCount(diamondGraph, dlinecount)
    assert totalSegmentsCycles == 2
