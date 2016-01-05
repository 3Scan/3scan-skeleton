import networkx as nx
import numpy as np

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.segmentLengths import getSegmentsAndLengths

from tests.tests3DSkeletonize import getDonut


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
    dlinecount, dlinelength, segmentTortuosityline, totalSegmentsLine = getSegmentsAndLengths(lineGraph)
    assert totalSegmentsLine == 1


def test_singlecycle():
    donutGraph = getCycleNotree()
    dcyclecount, dcyclelength, segmentTortuositycycle, totalSegmentsDonut = getSegmentsAndLengths(donutGraph)
    assert totalSegmentsDonut == 1


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    dcycleTreecount, dcycleTreelength, segmentTortuositycycletree, totalSegmentsSampleGraph = getSegmentsAndLengths(sampleGraph)
    assert totalSegmentsSampleGraph == 3


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    dTreecount, dTreelength, segmentTortuositytree, totalSegmentsTree = getSegmentsAndLengths(crosGraph)
    assert totalSegmentsTree == crosGraph.number_of_edges()


def test_disjointDoublecycle():
    multiloopgraph = getDisjointCyclesNoTrees2d()
    ddisjointCyclescount, ddisjointCycleslength, segmentTortuositycycles, totalSegmentsDisjointCycles = getSegmentsAndLengths(multiloopgraph)
    assert totalSegmentsDisjointCycles == 2


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    dTreescount, dTreeslength, segmentTortuositytrees, totalSegmentsTrees = getSegmentsAndLengths(crosPairgraph)
    assert totalSegmentsTrees == crosPairgraph.number_of_edges()


def test_touchingCycles():
    diamondGraph = nx.diamond_graph()
    dlinecount, dlinelength, segmentTortuosityline, totalSegmentsLine = getSegmentsAndLengths(diamondGraph)
    assert totalSegmentsLine == 1
