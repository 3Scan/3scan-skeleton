from skeleton import skeleton_testlib

"""
Tests 3D thinning implemented as in
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999 is working as expected
from the program thinVolume.py
"""


def test_donut():
    # Test 1 donut should result in a single cycle
    image = skeleton_testlib.getDonut()
    yield skeleton_testlib.checkCycle, image


def test_randomImage():
    # Test 2 All random images should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.get3DRandImages()
    for image in testImages:
        yield skeleton_testlib.checkSameObjects, image


def test_rectangles():
    # Test 3 All Rectangles should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.getStationary3dRectangles(width=0)
    for image in testImages:
        yield skeleton_testlib.checkSameObjects, image


def test_singlePixelLines():
    # Test 4 single pixel lines should still be the same in an image
    testImages = skeleton_testlib.getStationary3DSinglePixelLines(width=0)
    for image in testImages:
        yield skeleton_testlib.checkAlgorithmPreservesImage, image


def test_tinyLoopWithBranches():
    # Test 5 tiny loop with  branches should still be the same
    skeleton_testlib.checkSameObjects(skeleton_testlib.getTinyLoopWithBranches())


def test_wideLines():
    # Test 6 All widelines should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.get3DRolledThickLines()
    for image in testImages:
        yield skeleton_testlib.checkSameObjects, image


def test_crosPair():
    # Test 7 tiny loop with  branches should still be the same
    skeleton_testlib.checkSameObjects(skeleton_testlib.getDisjointCrosses())


def test_singleVoxelLine():
    skeleton_testlib.checkSameObjects(skeleton_testlib.getSingleVoxelLine())
