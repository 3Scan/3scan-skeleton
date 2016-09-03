import numpy as np
import pickle

from metrics.segmentStats import SegmentStats
from skeleton.skeletonClass import Skeleton
from skimage.morphology import skeletonize as getSkeletonize2D
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
    skel.setNetworkGraph()
    return skel.graph


def getCyclesWithBranchesProtrude(size=(10, 10)):
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
    skel.setNetworkGraph(findSkeleton=False)
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
    skel.setNetworkGraph()
    return skel.graph


def getSingleVoxelLineNobranches(size=(5, 5, 5)):
    # no branches single line
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    skel = Skeleton(sampleLine)
    skel.setNetworkGraph()
    return skel.graph


def test_cycleAndTree():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a cyclic tree
    sampleGraph = getCyclesWithBranchesProtrude()
    stats = SegmentStats(sampleGraph)
    stats.setStats()
    assert (stats.totalSegments == 4 and stats.typeGraphdict[0] == 3 and stats.countEndPoints == 2 and
            stats.countBranchPoints == 2 and stats.cycleInfo[0][0] == 2), pickle.dump(stats, open("unExpectedStats_cycleAndTree.p", "wb"))


def test_singleSegment():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a single segment
    lineGraph = getSingleVoxelLineNobranches()
    stats = SegmentStats(lineGraph)
    stats.setStats()
    assert (stats.totalSegments == 0 and stats.typeGraphdict[0] == 2 and
            stats.countEndPoints == 2 and stats.countBranchPoints == 0 and
            stats.hausdorffDimensionDict == {} and stats.cycleInfo == {}), pickle.dump(stats, open("unExpectedStats_singleSegment.p", "wb"))


def test_singleCycle():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a single cycle
    donutGraph = getCycleNoTree()
    stats = SegmentStats(donutGraph)
    stats.setStats()
    assert (stats.totalSegments == 1 and stats.typeGraphdict[0] == 1 and stats.countEndPoints == 0 and
            stats.countBranchPoints == 0 and
            stats.hausdorffDimensionDict == {} and stats.cycleInfo[0][0] == 0), pickle.dump(stats, open("unExpectedStats_singleCycle.p", "wb"))


def test_treeNoCycle3D():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a tree like structure
    crosPairgraph = getDisjointTreesNoCycle3d()
    stats = SegmentStats(crosPairgraph)
    stats.setStats()
    assert (stats.totalSegments == 8 and stats.typeGraphdict[0] == 4 and stats.countEndPoints == 8 and
            stats.countBranchPoints == 2 and stats.cycleInfo == {}), pickle.dump(stats, open("unExpectedStats_treeNoCycle3D.p", "wb"))



