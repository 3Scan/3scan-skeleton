import numpy as np

import os

from skimage.morphology import skeletonize as getSkeletonize2D

from skeleton.thin3DVolume import getThinned3D
from skeleton.objWrite import getObjWrite

from tests.test3DThinning import getDonut

"""
   program to test if obj files written using (i=objWrite.py) getNetworkxGraphFromarray and
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
    donut = getThinned3D(getDonut())
    return donut


def getTreeNoCycle2d(size=(5, 5)):
    cros = np.zeros(size, dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    return cros


def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    return crosPair


def getDisjointCyclesNoTrees2d(size=(10, 10)):
    tinyLoop = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=bool)
    multiLoop = np.zeros(size, dtype=bool)
    multiLoop[2:5, 2:5] = tinyLoop
    multiLoop[7:10, 7:10] = tinyLoop
    return multiLoop


def test_singlesegment():
    lineGraph = getSingleVoxelLineNobranches()
    getObjWrite(lineGraph, "Line.obj")
    objFile = open("Line.obj", "r")
    totalSegments = 0
    for line in objFile:
        for index, items in enumerate(line):
            if items == 'l':
                totalSegments = totalSegments + 1
    os.remove("Line.obj")
    assert totalSegments == 1


def test_singlecycle():
    donutGraph = getCycleNotree()
    getObjWrite(donutGraph, "OneCycle.obj")
    objFile = open("OneCycle.obj", "r")
    totalSegmentsCycle = 0
    for line in objFile:
        for index, items in enumerate(line):
            if items == 'l':
                totalSegmentsCycle = totalSegmentsCycle + 1
    os.remove("OneCycle.obj")
    assert totalSegmentsCycle == 0


def test_cycleAndTree():
    sampleGraph = getCyclesWithBranchesProtrude()
    getObjWrite(sampleGraph, "CycleAndGraph.obj")
    objFile = open("CycleAndGraph.obj", "r")
    totalSegmentsCyclesTree = 0
    for line in objFile:
        for index, items in enumerate(line):
            if items == 'l':
                totalSegmentsCyclesTree = totalSegmentsCyclesTree + 1
    os.remove("CycleAndGraph.obj")
    assert totalSegmentsCyclesTree == 4


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    getObjWrite(crosGraph, "Cross.obj")
    objFile = open("Cross.obj", "r")
    totalSegmentsCycles = 0
    for line in objFile:
        for index, items in enumerate(line):
            if items == 'l':
                totalSegmentsCycles = totalSegmentsCycles + 1
    os.remove("Cross.obj")
    assert totalSegmentsCycles == 4


def test_disjointDoublecycle():
    multiloopgraph = getDisjointCyclesNoTrees2d()
    getObjWrite(multiloopgraph, "DisjointCycles.obj")
    objFile = open("DisjointCycles.obj", "r")
    totalSegmentsDisjointCycles = 0
    for line in objFile:
        for index, items in enumerate(line):
            if items == 'l':
                totalSegmentsDisjointCycles = totalSegmentsDisjointCycles + 1
    os.remove("DisjointCycles.obj")
    assert totalSegmentsDisjointCycles == 0


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    getObjWrite(crosPairgraph, "twoCrosses.obj")
    objFile = open("twoCrosses.obj", "r")
    totalSegmentsCrosses = 0
    for line in objFile:
        for index, items in enumerate(line):
            if items == 'l':
                totalSegmentsCrosses = totalSegmentsCrosses + 1
    os.remove("twoCrosses.obj")
    assert totalSegmentsCrosses == 8

