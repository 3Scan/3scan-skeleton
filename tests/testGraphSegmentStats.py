import numpy as np

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.segmentStats import getSegmentStats
from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
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
    frame[1, 5] = 1
    frame[7, 5] = 1
    sampleImage = np.zeros((3, 10, 10), dtype=np.uint8)
    sampleImage[1] = frame
    return removeCliqueEdges(getNetworkxGraphFromarray(sampleImage))


def getSingleVoxelLineNobranches(size=(5, 5, 5)):
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    return removeCliqueEdges(getNetworkxGraphFromarray(sampleLine))


def getCycleNotree():
    from skeleton.thin3DVolume import getThinned3D
    from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton
    donut = getDonut()
    return removeCliqueEdges(getNetworkxGraphFromarray(getShortestPathSkeleton(getThinned3D(donut))))


def getTreeNoCycle2d(size=(5, 5)):
    cros = np.zeros(size, dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    return removeCliqueEdges(getNetworkxGraphFromarray(cros))


def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    return removeCliqueEdges(getNetworkxGraphFromarray(crosPair))


def getDisjointCyclesNoTrees2d(size=(10, 10)):
    tinyLoop = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=bool)
    multiLoop = np.zeros(size, dtype=bool)
    multiLoop[2:5, 2:5] = tinyLoop
    multiLoop[7:10, 7:10] = tinyLoop
    return removeCliqueEdges(getNetworkxGraphFromarray(multiLoop))


def test_singlesegment():
    lineGraph = getSingleVoxelLineNobranches()
    dlinecount, dlinelength, segmentTortuosityline, totalSegmentsLine, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(lineGraph)
    assert sum(list(dlinelength.values())) == 0
    assert totalSegmentsLine == 0 and typeGraphdict[0] == 2 and endP == 2 and branchP == 0 and segmentHausdorffDimensiondict == {} and cycleInfo == {}


def test_singlecycle():
    donutGraph = getCycleNotree()
    dcyclecount, dcyclelength, segmentTortuositycycle, totalSegmentsDonut, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(donutGraph)
    a = sum(list(dcyclelength.values()))
    b = sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in donutGraph.edges()])
    np.testing.assert_allclose(a, b)
    assert totalSegmentsDonut == 1 and typeGraphdict[0] == 1 and endP == 0 and branchP == 0 and segmentHausdorffDimensiondict == {} and cycleInfo[0][0] == 0


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    dTreecount, dTreelength, segmentTortuositytree, totalSegmentsTree, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(crosGraph)
    a = sum(list(dTreelength.values()))
    b = sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in crosGraph.edges()])
    np.testing.assert_allclose(a, b)
    assert totalSegmentsTree == 4 and typeGraphdict[0] == 4 and endP == 4 and branchP == 1 and cycleInfo == {}


def test_disjointDoublecycle():
    multiloopgraph = getDisjointCyclesNoTrees2d()
    disjointCyclescount, ddisjointCycleslength, segmentTortuositycycles, totalSegmentsDisjointCycles, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(multiloopgraph)
    a = sum(list(ddisjointCycleslength.values()))
    b = sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in multiloopgraph.edges()])
    np.testing.assert_allclose(a, b)
    assert totalSegmentsDisjointCycles == 2 and typeGraphdict[0] == 1 and endP == 0 and branchP == 0 and len(cycleInfo) == 2 and cycleInfo[0][0] == 0 and cycleInfo[1][0] == 0


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    dTreescount, dTreeslength, segmentTortuositytrees, totalSegmentsTrees, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(crosPairgraph)
    a = sum(list(dTreeslength.values()))
    b = sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in crosPairgraph.edges()])
    np.testing.assert_allclose(a, b)
    assert totalSegmentsTrees == 8 and typeGraphdict[0] == 4 and endP == 8 and branchP == 2 and cycleInfo == {}


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    dcycleTreecount, dcycleTreelength, segmentTortuositycycletree, totalSegmentsSampleGraph, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(sampleGraph)
    a = sum(list(dcycleTreelength.values()))
    b = sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in sampleGraph.edges()])
    np.testing.assert_allclose(a, b)
    print(a, b)
    print(dcycleTreelength)
    assert totalSegmentsSampleGraph == 4 and typeGraphdict[0] == 3 and endP == 2 and branchP == 2 and cycleInfo[0][0] == 2

