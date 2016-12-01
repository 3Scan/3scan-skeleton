from scipy import ndimage
import numpy as np

from skeleton.skeletonClass import Skeleton
from skeleton import skeleton_testlib


"""
Tests 3D thinning implemented as in
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999 is working as expected
from the program thinVolume.py
"""


def checkSameObjects(image):
    # check if number of objects are same in input and output of thinning
    dims = image.ndim
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones([3] * dims, dtype=bool))
    skel = Skeleton(image)
    skel.setThinningOutput()
    newImage = skel.thinnedStack
    label_img, countObjectsn = ndimage.measurements.label(newImage, structure=np.ones([3] * dims, dtype=bool))
    np.save("image.npy", image)
    assert (countObjectsn == countObjects), ("number of objects in input "
                                             "{} is different from output {}".format(countObjects, countObjectsn))
    return newImage


def checkAlgorithmPreservesImage(image):
    skel = Skeleton(image)
    skel.setThinningOutput()
    newImage = skel.thinnedStack
    assert np.array_equal(image, newImage)


def checkCycle(image):
    # check if number of cycles in the donut image after thinning is 1
    skel = Skeleton(image)
    skel.setThinningOutput()
    newImage = skel.thinnedStack
    label_img, countObjects = ndimage.measurements.label(newImage, structure=np.ones((3, 3, 3), dtype=bool))
    assert countObjects == 1, "number of cycles in single donut is {}".format(countObjects)


def test_donut():
    # Test 1 donut should result in a single cycle
    image = skeleton_testlib.getDonut()
    checkCycle(image)


def test_randomImage():
    # Test 2 All random images should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.get3DRandImages()
    for image in testImages:
        yield checkSameObjects, image


def test_rectangles():
    # Test 3 All Rectangles should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.getStationary3dRectangles()
    for image in testImages:
        print(image.sum())
        yield checkSameObjects, image


def test_singlePixelLines():
    # Test 4 single pixel lines should still be the same in an image
    testImages = skeleton_testlib.getStationary3DSinglePixelLines(width=0)
    for image in testImages:
        yield checkAlgorithmPreservesImage, image


def test_tinyLoopWithBranches():
    # Test 5 tiny loop with  branches should still be the same
    checkSameObjects(skeleton_testlib.getTinyLoopWithBranches())


def test_wideLines():
    # Test 6 All widelines should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.get3DRolledThickLines()
    for image in testImages:
        yield checkSameObjects, image


def test_crosPair():
    # Test 7 tiny loop with  branches should still be the same
    checkSameObjects(skeleton_testlib.getDisjointCrosses())


def test_singleVoxelLine():
    # Test 8 single voxel line should still be the same
    checkSameObjects(skeleton_testlib.getSingleVoxelLine())
