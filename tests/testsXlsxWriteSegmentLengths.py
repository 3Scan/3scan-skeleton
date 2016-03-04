import numpy as np

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.convOptimize import getSkeletonize3D
from skeleton.xlsxwrite import excelWrite

from tests.tests3DSkeletonize import getDonut

"""
   program to test if obj files written using (i=xlsxwrite.py) getNetworkxGraphFromarray and
   from the dictionary of the coordinate and adjacent nonzero coordinates
   after removing the cliques have the number of segments as expected
"""


def getCyclesWithBranchesProtrude(size=(10, 10)):
    frame = np.zeros(size, dtype=np.uint8)
    frame[2:-2, 2:-2] = 1
    frame[4:-4, 4:-4] = 0
    frame = getSkeletonize2D(frame)
    frame[1, 5] = 1; frame[7, 5] = 1;
    sampleImage = np.zeros((3, 10, 10), dtype=np.uint8)
    sampleImage[1] = frame
    return sampleImage


def getSingleVoxelLineNobranches(size=(5, 5, 5)):
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    return sampleLine


def getCycleNotree():
    donut = getSkeletonize3D(getDonut())
    return donut


def getTreeNoCycle2d(size=(1, 7, 7)):
    cros = np.zeros(size, dtype=np.uint8)
    cros[:, :, 2] = 1
    cros[:, 2, :] = 1
    cros[:, 4, 3] = 1
    return cros


def getDisjointTreesNoCycle3d(size=(14, 14, 14)):
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((7, 7), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    cros[4, 3] = 1
    crosPair[0, 0:7, 0:7] = cros
    crosPair[7, 7:14, 7:14] = cros
    return crosPair


def test_singlesegment():
    lineGraph = getSingleVoxelLineNobranches()
    d = excelWrite(lineGraph, lineGraph, "Line.xlsx")
    assert len(d) == 0


def test_singlecycle():
    donutGraph = getCycleNotree()
    d = excelWrite(donutGraph, donutGraph, "OneCycle.xlsx")
    assert len(d) == 1


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    d = excelWrite(sampleGraph, sampleGraph, "CycleAndGraph.xlsx")
    assert len(d) == 1


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    d = excelWrite(crosGraph, crosGraph, "Cross.xlsx")
    assert len(d) == 2


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    d = excelWrite(crosPairgraph, crosPairgraph, "twoCrosses.xlsx")
    assert len(d) == 4
