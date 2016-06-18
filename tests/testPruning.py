import numpy as np
from scipy import ndimage
from skeleton.pruningByDistTransform import getPrunedSkeleton
from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton
from skeleton.thin3DVolume import getThinned3D


def getRectangle(w=10, h=6, d=6):
    return np.ones((w, h, d), dtype=bool)


def getRectangleNoise(w=10, h=6, d=6):
    rect = np.zeros((w + 2, h + 2, d + 2), dtype=bool)
    rect[1: w + 1, 1: h + 1, 1: d + 1] = 1
    rect[int((w / 2) + 1), int((h / 2) + 1), 1] = 1
    return rect


def getRandomBlob():
    from scipy.spatial import ConvexHull
    xs = np.random.uniform(-1, 1, size=5)
    ys = np.random.uniform(-1, 1, size=5)
    zs = np.random.uniform(-1, 1, size=5)

    xyzs = list(zip(xs, ys, zs))

    hullz = ConvexHull(xyzs)

    xf, yf, zf = np.mgrid[-1:1:50j, -1:1:50j, -1:1:50j]
    i = np.ones(xf.shape, dtype=bool)
    for x, y, z, c in hullz.equations:
        mask = (xf * x) + (yf * y) + (zf * z) - c < 0
        i[mask] = 0
    i = i.astype(bool)
    return i


def checkAlgorithmSinglePixeled(image):
    print("sum before", np.sum(image))
    newImage = getPrunedSkeleton(getShortestPathSkeleton(getThinned3D(image)))
    label_img, countObjectsn = ndimage.measurements.label(newImage, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    print(countObjects, countObjectsn, np.sum(newImage))
    assert (countObjectsn == countObjects)


def test_rectangle():
    checkAlgorithmSinglePixeled(getRectangle())


def test_rectangleNoise():
    checkAlgorithmSinglePixeled(getRectangleNoise())


def test_randomBlob():
    checkAlgorithmSinglePixeled(getRandomBlob())
