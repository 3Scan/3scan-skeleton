import numpy as np

from scipy import ndimage
from skeleton.thinVolume import getThinned
from skeleton.thinning_testlib import (getDonut, get3DRandImages, getDisjointCrosses, get3DRolledThickLines, getTinyLoopWithBranches,
                                       getSingleVoxelLine, getStationary3dRectangles, getStationary3DSinglePixelLines, checkSameObjects)

"""
Tests 3D thinning implemented as in
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999 is working as expected
from the program thinVolume.py
"""


def checkAlgorithmPreservesImage(image):
    newImage = getThinned(image)
    assert np.array_equal(image, newImage)


def checkCycle(image):
    # check if number of cycles in the donut image after thinning is 1
    image = getThinned(image)
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=bool))
    assert countObjects == 1, "number of cycles in single donut is {}".format(countObjects)


def test_donut():
    # Test 1 donut should result in a single cycle
    image = getDonut()
    yield checkCycle, image


def test_randomImage():
    # Test 2 All random images should preserve topology and should have
    # same number of objects
    testImages = get3DRandImages()
    for image in testImages:
        yield checkSameObjects, image


def test_rectangles():
    # Test 3 All Rectangles should preserve topology and should have
    # same number of objects
    testImages = getStationary3dRectangles(width=0)
    for image in testImages:
        yield checkSameObjects, image


def test_singlePixelLines():
    # Test 4 single pixel lines should still be the same in an image
    testImages = getStationary3DSinglePixelLines(width=0)
    for image in testImages:
        yield checkAlgorithmPreservesImage, image


def test_tinyLoopWithBranches():
    # Test 5 tiny loop with  branches should still be the same
    checkSameObjects(getTinyLoopWithBranches())


def test_wideLines():
    # Test 6 All widelines should preserve topology and should have
    # same number of objects
    testImages = get3DRolledThickLines()
    for image in testImages:
        yield checkSameObjects, image


def test_crosPair():
    # Test 7 tiny loop with  branches should still be the same
    checkSameObjects(getDisjointCrosses())


def test_singleVoxelLine():
    checkSameObjects(getSingleVoxelLine())



