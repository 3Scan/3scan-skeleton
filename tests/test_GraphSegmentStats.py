import numpy as np

from metrics.segmentStats import SegmentStats
from skeleton.skeletonClass import Skeleton
from tests.test_3DThinning import getDonut

"""
Program to test if graphs created using networkxGraphFromArray and removeCliqueEdges
and from skeletons generated using the functions in the skeleton.skeletonClass, metrics.segmentStats
from the dictionary of the coordinate and adjacent nonzero coordinates
after removing the cliques have the number of segments as expected
PV TODO:Test if lengths of segments and tortuoisty of the curves as expected
"""


def getCycleNoTree():
    donut = getDonut()
    skel = Skeleton(donut)
    skel.setNetworkGraph(True)
    return skel.graph


def getCyclesWithBranchesProtrude(size=(10, 10)):
    from skimage.morphology import skeletonize as getSkeletonize2D
    # a loop and a branches coming at end of the cycle
    frame = np.zeros(size, dtype=np.uint8)
    frame[2:-2, 2:-2] = 1
    frame[4:-4, 4:-4] = 0
    frame = getSkeletonize2D(frame)
    frame[1, 5] = 1
    frame[7, 5] = 1
    sampleImage = np.zeros((3, 10, 10), dtype=np.uint8)
    sampleImage[1] = frame
    skel = Skeleton(sampleImage)
    skel.setNetworkGraph(False)
    return skel.graph


def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    # two disjoint crosses
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    skel = Skeleton(crosPair)
    skel.setNetworkGraph(False)
    return skel.graph


def getSingleVoxelLineNobranches(size=(5, 5, 5)):
    # no branches single line
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    skel = Skeleton(sampleLine)
    skel.setNetworkGraph(False)
    return skel.graph


def test_cycleAndTree():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a cyclic tree
    sampleGraph = getCyclesWithBranchesProtrude()
    stats = SegmentStats(sampleGraph)
    stats.setStats()
    assert stats.totalSegments == 4, "totalSegments in cycleAndTree sample should be 4, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 3, "type of graph in cycleAndTree sample should be 3, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 2, "number of end points in cycleAndTree sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 2, "number of branch points in cycleAndTree sample should be 2, it is {}".format(stats.countBranchPoints)
    assert stats.cycleInfoDict[0][0] == 2, "number of branch points on the cycle must be 2, it is {}".format(stats.cycleInfoDict[0][0])


def test_singleSegment():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a single segment
    lineGraph = getSingleVoxelLineNobranches()
    stats = SegmentStats(lineGraph)
    stats.setStats()
    assert stats.totalSegments == 0, "totalSegments in singleSegment sample should be 0, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 2, "type of graph in singleSegment sample should be 2, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 2, "number of end points in singleSegment sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 0, "number of branch points in singleSegment sample should be 0, it is {}".format(stats.countBranchPoints)
    assert stats.hausdorffDimensionDict == {}, "hausdorffDimensionDict must be empty, it is {}".format(stats.hausdorffDimensionDict)
    assert stats.cycleInfoDict == {}, "cycleInfoDict must be empty, it is {}".format(stats.cycleInfoDict)


def test_singleCycle():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a single cycle
    donutGraph = getCycleNoTree()
    stats = SegmentStats(donutGraph)
    stats.setStats()
    assert stats.totalSegments == 1, "totalSegments in singleCycle sample should be 1, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 1, "type of graph in singleCycle sample should be 1, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 0, "number of end points in singleCycle sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 0, "number of branch points in singleCycle sample should be 0, it is {}".format(stats.countBranchPoints)
    assert stats.hausdorffDimensionDict == {}, "hausdorffDimensionDict must be empty, it is {}".format(stats.hausdorffDimensionDict)
    assert stats.cycleInfoDict[0][0] == 0, "number of branch points on the cycle must be 0, it is {}".format(stats.cycleInfoDict[0][0])


def test_treeNoCycle3D():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a tree like structure
    crosPairgraph = getDisjointTreesNoCycle3d()
    stats = SegmentStats(crosPairgraph)
    stats.setStats()
    assert stats.totalSegments == 8, "totalSegments in treeNoCycle3D sample should be 8, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 4, "type of graph in treeNoCycle3D sample should be 4, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 8, "number of end points in treeNoCycle3D sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 2, "number of branch points in treeNoCycle3D sample should be 2, it is {}".format(stats.countBranchPoints)
    assert stats.cycleInfoDict == {}, "cycleInfoDict must be empty, it is {}".format(stats.cycleInfoDict)


