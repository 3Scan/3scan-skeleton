import numpy as np
from scipy import ndimage

from skeleton.thinning_testlib import getRandomBlob
from skeleton.unitWidthCurveSkeleton import getShortestPathSkeleton, outOfPixBounds
"""
tests for unit width curve skeleton
implemented according to the paper
Generation of Unit-Width Curve Skeletons Based on Valence Driven Spatial Median (VDSM)
Tao Wang, Irene Cheng
Advances in Visual Computing
Volume 5358 of the series Lecture Notes in Computer Science pp 1051-1060
in unitWidthCurveSkeleton.py
"""


def checkAlgorithmSameObjects(image):
    # single object is expected in convex blob it should remain the same in skeletonized blob
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    newImage = getShortestPathSkeleton(image)
    label_img, countObjectsn = ndimage.measurements.label(newImage, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert (countObjectsn == countObjects), ("number of objects in the skeletonized image"
                                             "{} is different from input {}".format(countObjectsn, countObjects))


def test_randomBlob():
    # Test 1 Random blob skeletonizes to topology preserving sinle object representing the blob
    testBlob = getRandomBlob()
    checkAlgorithmSameObjects(testBlob)


def test_outOfPixBounds():
    assert outOfPixBounds((51, 51), (50, 50)), "out of pix bounds should return 1, it is returning 0 instead"
    assert outOfPixBounds((49, 50), (50, 50)), "out of pix bounds should return 1, it is returning 0 instead"
    assert outOfPixBounds((50, 49), (50, 50)), "out of pix bounds should return 1, it is returning 0 instead"
    assert outOfPixBounds((50, 50), (50, 50)), "out of pix bounds should return 1, it is returning 0 instead"
    assert outOfPixBounds((50, 50, 50), (50, 50, 50)), "out of pix bounds should return 1, it is returning 0 instead"
    assert outOfPixBounds((49, 49), (50, 50)) == 0, "out of pix bounds should return 0, it is returning 1 instead"
    assert outOfPixBounds((51, 51, 51), (50, 50, 50)), "out of pix bounds should return 1, it is returning 0 instead"
    assert outOfPixBounds((49, 49, 49), (50, 50, 50)) == 0, "out of pix bounds should return 0, it is returning 1 instead"
