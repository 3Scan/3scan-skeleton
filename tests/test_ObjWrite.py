import os
import shutil
import tempfile

from runscripts.objWrite import getObjWrite
from tests.test_GraphSegmentStats import (getCyclesWithBranchesProtrude, getSingleVoxelLineNobranches,
                                          getCycleNoTree, getDisjointTreesNoCycle3d)


"""
Program to test if obj files written using objWrite.py) have expected lines starting with prefix 'l'
"""


def _getLinePrefixes(graph, path):
    # get number of lines in obj with prefix l
    tempDir = tempfile.mkdtemp() + os.sep
    tempDirPath = tempDir + path
    getObjWrite(graph, tempDirPath)
    objFile = open(tempDirPath, "r")
    totalSegments = 0
    for line in objFile:
        for index, items in enumerate(line):
            if items == 'l':
                totalSegments = totalSegments + 1
    shutil.rmtree(tempDir)
    return totalSegments


def test_singleSegment():
    # Test 1 Prefixes of l for a single segment
    lineGraph = getSingleVoxelLineNobranches()
    totalSegments = _getLinePrefixes(lineGraph, "Line.obj")
    assert totalSegments == 1, "number of line prefixes in obj {} is not 1".format(totalSegments)


def test_singleCycle():
    # Test 2 Prefixes of l for a single cycle
    donutGraph = getCycleNoTree()
    totalSegmentsCycle = _getLinePrefixes(donutGraph, "OneCycle.obj")
    assert totalSegmentsCycle == 0, "number of line prefixes in obj {} is not 0".format(totalSegmentsCycle)


def test_cycleAndTree():
    # Test 3 Prefixes of l for a cyclic tree
    sampleGraph = getCyclesWithBranchesProtrude()
    totalSegmentsCyclicTree = _getLinePrefixes(sampleGraph, "CycleAndTree.obj")
    assert totalSegmentsCyclicTree == 4, "number of line prefixes in obj {} is not 4".format(totalSegmentsCyclicTree)


def test_treeNoCycle3d():
    # Test 4 Prefixes of l for a tree
    crosPairgraph = getDisjointTreesNoCycle3d()
    totalSegmentsCrosses = _getLinePrefixes(crosPairgraph, "Tree.obj")
    assert totalSegmentsCrosses == 8, "number of line prefixes in obj {} is not 8".format(totalSegmentsCrosses)

