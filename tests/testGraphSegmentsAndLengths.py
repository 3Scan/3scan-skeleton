import numpy as np

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.segmentLengths import getSegmentsAndLengths
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.cliqueRemoving import removeCliqueEdges

from tests.test3DThinning import getDonut

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
    sampleGraph = removeCliqueEdges(sampleGraph)
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


def getTreeNoCycle2d(size=(5, 5)):
    cros = np.zeros(size, dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosGraph = getNetworkxGraphFromarray(cros, True)
    crosGraph = removeCliqueEdges(crosGraph)
    return crosGraph


def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
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
    multiloopgraph = getNetworkxGraphFromarray(multiLoop, False)
    multiloopgraph = removeCliqueEdges(multiloopgraph)
    return multiloopgraph


def test_singlesegment():
    lineGraph = getSingleVoxelLineNobranches()
    dlinecount, dlinelength, segmentTortuosityline, totalSegmentsLine, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentFractalDimensiondict = getSegmentsAndLengths(lineGraph, True, False)
    # plotGraphWithCount(lineGraph, dlinecount)
    print(endP, branchP)
    assert totalSegmentsLine == 1 and typeGraphdict[0] == 2 and endP == 2 and branchP == 0 and segmentFractalDimensiondict == {}


def test_singlecycle():
    donutGraph = getCycleNotree()
    dcyclecount, dcyclelength, segmentTortuositycycle, totalSegmentsDonut, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentFractalDimensiondict = getSegmentsAndLengths(donutGraph, True, False)
    # plotGraphWithCount(donutGraph, dcyclecount)
    print(endP, branchP)
    assert totalSegmentsDonut == 1 and typeGraphdict[0] == 1 and endP == 0 and branchP == 0 and segmentFractalDimensiondict == {}


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    dcycleTreecount, dcycleTreelength, segmentTortuositycycletree, totalSegmentsSampleGraph, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentFractalDimensiondict = getSegmentsAndLengths(sampleGraph, True, False)
    # plotGraphWithCount(sampleGraph, dcycleTreecount)
    print(endP, branchP)
    assert totalSegmentsSampleGraph == 4 and typeGraphdict[0] == 3 and endP == 2 and branchP == 2


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    dTreecount, dTreelength, segmentTortuositytree, totalSegmentsTree, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentFractalDimensiondict = getSegmentsAndLengths(crosGraph, True, False)
    # plotGraphWithCount(crosGraph, dTreecount)
    print(endP, branchP)
    assert totalSegmentsTree == 4 and typeGraphdict[0] == 4 and endP == 4 and branchP == 1


def test_disjointDoublecycle():
    multiloopgraph = getDisjointCyclesNoTrees2d()
    disjointCyclescount, ddisjointCycleslength, segmentTortuositycycles, totalSegmentsDisjointCycles, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentFractalDimensiondict = getSegmentsAndLengths(multiloopgraph, True, False)
    # plotGraphWithCount(multiloopgraph, ddisjointCyclescount)
    print(endP, branchP)
    assert totalSegmentsDisjointCycles == 2 and typeGraphdict[0] == 1 and endP == 0 and branchP == 0


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    dTreescount, dTreeslength, segmentTortuositytrees, totalSegmentsTrees, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentFractalDimensiondict = getSegmentsAndLengths(crosPairgraph, True, False)
    # plotGraphWithCount(crosPairgraph, dTreescount)
    print(endP, branchP)
    assert totalSegmentsTrees == 8 and typeGraphdict[0] == 4 and endP == 8 and branchP == 2
