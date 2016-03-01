import networkx as nx
import numpy as np
<<<<<<< HEAD
from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.BifurcatedsegmentLengths import getBifurcatedSegmentsAndLengths
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
=======

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.segmentLengths import getSegmentsAndLengths
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemovig import removeCliqueEdges
from skeleton.BifurcatedsegmentLengths import getBifurcatedSegmentsAndLengths
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be
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
<<<<<<< HEAD
    sampleGraph = removeCliqueEdges(sampleGraph)
=======
    removeCliqueEdges(sampleGraph)
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be
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
<<<<<<< HEAD
    crosGraph = getNetworkxGraphFromarray(cros, True)
=======
    crosGraph = getNetworkxGraphFromarray(cros, False)
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be
    crosGraph = removeCliqueEdges(crosGraph)
    return crosGraph


<<<<<<< HEAD
def getDisjointTreesNoCycle3d(size=(14, 14, 14)):
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((7, 7), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    cros[4, 3] = 1
=======
def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    crosPair = np.zeros(size, dtype=np.uint8)
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be
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
<<<<<<< HEAD
    multiloopgraph = getNetworkxGraphFromarray(multiLoop, False)
=======
    multiloopgraph = getNetworkxGraphFromarray(multiLoop, True)
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be
    multiloopgraph = removeCliqueEdges(multiloopgraph)
    return multiloopgraph


def test_singlesegment():
    lineGraph = getSingleVoxelLineNobranches()
<<<<<<< HEAD
    dlinecount, dlinelength, segmentTortuosityline, totalSegmentsLine = getBifurcatedSegmentsAndLengths(lineGraph, True, False)
    assert totalSegmentsLine == 0
=======
    dlinecount, dlinelength, segmentTortuosityline, totalSegmentsLine = getSegmentsAndLengths(lineGraph, True, False)
    # plotGraphWithCount(lineGraph, dlinecount)
    assert totalSegmentsLine == 1
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be


def test_singlecycle():
    donutGraph = getCycleNotree()
<<<<<<< HEAD
    dcyclecount, dcyclelength, segmentTortuositycycle, totalSegmentsDonut = getBifurcatedSegmentsAndLengths(donutGraph, True, False)
    assert totalSegmentsDonut == 1


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    dTreecount, dTreelength, segmentTortuositytree, totalSegmentsTree = getBifurcatedSegmentsAndLengths(crosGraph, True, False)
    assert totalSegmentsTree == 1
=======
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
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be


def test_disjointDoublecycle():
    multiloopgraph = getDisjointCyclesNoTrees2d()
<<<<<<< HEAD
    disjointCyclescount, ddisjointCycleslength, segmentTortuositycycles, totalSegmentsDisjointCycles = getBifurcatedSegmentsAndLengths(multiloopgraph, True, False)
=======
    disjointCyclescount, ddisjointCycleslength, segmentTortuositycycles, totalSegmentsDisjointCycles = getSegmentsAndLengths(multiloopgraph, True, False)
    # plotGraphWithCount(multiloopgraph, ddisjointCyclescount)
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be
    assert totalSegmentsDisjointCycles == 2


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
<<<<<<< HEAD
    dTreescount, dTreeslength, segmentTortuositytrees, totalSegmentsTrees = getBifurcatedSegmentsAndLengths(crosPairgraph, True, False)
    assert totalSegmentsTrees == 2
=======
    dTreescount, dTreeslength, segmentTortuositytrees, totalSegmentsTrees = getSegmentsAndLengths(crosPairgraph, True, False)
    # plotGraphWithCount(crosPairgraph, dTreescount)
    assert totalSegmentsTrees == 8
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be


def test_balancedtree():
    balancedTree = nx.balanced_tree(2, 1)
<<<<<<< HEAD
    dlinecountbaltree, dlinebaltree, segmentTortuositybaltree, totalSegmentsBalancedTree = getBifurcatedSegmentsAndLengths(balancedTree, True, False)
    assert totalSegmentsBalancedTree == 0
=======
    dlinecountbaltree, dlinebaltree, segmentTortuositybaltree, totalSegmentsBalancedTree = getSegmentsAndLengths(balancedTree, True, False)
    # plotGraphWithCount(balancedTree, dlinecount)
    assert totalSegmentsBalancedTree == 2
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be


def test_touchingCycles():
    diamondGraph = nx.diamond_graph()
<<<<<<< HEAD
    dcyclescount, dcycleslength, segmentTortuositycycles, totalSegmentsCycles = getBifurcatedSegmentsAndLengths(diamondGraph, True, False)
    assert totalSegmentsCycles == 2


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    dcycleTreecount, dcycleTreelength, segmentTortuositycycletree, totalSegmentsSampleGraph = getBifurcatedSegmentsAndLengths(sampleGraph, True, False)
    assert totalSegmentsSampleGraph == 1
=======
    dcyclescount, dcycleslength, segmentTortuositycycles, totalSegmentsCycles = getSegmentsAndLengths(diamondGraph, True, False)
    # plotGraphWithCount(diamondGraph, dlinecount)
    assert totalSegmentsCycles == 2
>>>>>>> dabc0f3b6bcb1524a4adaa5d6dd20d93dc5e79be
