import numpy as np
from scipy import ndimage

from skeleton.thinVolume import getThinned
from skeleton.unitWidthCurveSkeleton import getShortestPathSkeleton

"""
tests for unit width curve skeleton
implemented according to the paper
Generation of Unit-Width Curve Skeletons Based on Valence Driven Spatial Median (VDSM)
Tao Wang, Irene Cheng
Advances in Visual Computing
Volume 5358 of the series Lecture Notes in Computer Science pp 1051-1060
in unitWidthCurveSkeleton.py
"""


def getRandomBlob():
    # get random convex blob
    from scipy.spatial import ConvexHull
    xs = np.random.uniform(-1, 1, size=50)
    ys = np.random.uniform(-1, 1, size=50)
    zs = np.random.uniform(-1, 1, size=50)

    xyzs = list(zip(xs, ys, zs))

    hullz = ConvexHull(xyzs)

    xf, yf, zf = np.mgrid[-1:1:100j, -1:1:100j, -1:1:100j]
    blob = np.ones(xf.shape, dtype=bool)
    for x, y, z, c in hullz.equations:
        mask = (xf * x) + (yf * y) + (zf * z) - c < 0
        blob[mask] = 0
    blob = blob.astype(bool)
    return blob


def checkAlgorithmSameObjects(image):
    # single object is expected in convex blob it should remain the same in skeletonized blob
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    newImage = getThinned(image)
    newImage = getShortestPathSkeleton(newImage)
    label_img, countObjectsn = ndimage.measurements.label(newImage, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if countObjects != countObjectsn:
        np.save("image.npy", image)
    assert (countObjectsn == countObjects), "number of objects in the skeletonized image {} is different from input {}".format(countObjectsn, countObjects)


def test_randomBlob():
    # Test 1 Random blob skeletonizes to topology preserving sinle object representing the blob
    testBlob = getRandomBlob()
    checkAlgorithmSameObjects(testBlob)
