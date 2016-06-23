import numpy as np
from scipy import ndimage

from skeleton.thin3DVolume import getThinned3D
from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton

""" test 1 and 3 lines passed , test for rings and random images not defined
and not checked
"""


def getStationary3DSinglePixelLines(width=5):
    # Test #1:
    # the answer that skeletonize gives for hLine/vLine should be the same as
    # the input, as there is no pixel that can be removed without affecting
    # the topology.
    # The algorithim when run on any of the below feature sets
    # Should return a skeleton that is the same as the feature
    # A single horizontal/vertical line
    hLine = np.zeros((25, 25, 25), dtype=np.uint8)
    hLine[:, 8:8 + width, :] = 1
    vLine = hLine.T.copy()

    # A "comb" of lines
    hLines = np.zeros((25, 25, 25), dtype=np.uint8)
    hLines[0:width, ::3, :] = 1
    vLines = hLines.T.copy()
    # A grid made up of two perpendicular combs
    grid = hLines | vLines
    stationaryImages = [hLine, vLine, hLines, vLines, grid]
    return stationaryImages


def get3DRandImages():
    # Test #2:
    # The algorithim should be able to run on arbitrary input without crashing.
    # We are not interested in the answer, so much as that the algo has full
    # coverage over the possible inputs into it
    randomImages = [np.random.randint(2, size=(25, 25, 25)) for i in range(6)]
    return randomImages


def get3DRolledThickLines():
    # Test #3:
    # The algorithim should thing a single long contiguous segment to a line of
    # pixels
    hBar = np.zeros((25, 25, 25), dtype=np.uint8)
    hBar[1, 0:5, :] = 1
    barImages = [np.roll(hBar, 2 * n, axis=0) for n in range(10)]
    # 2,6 and 20 pixel wide lines
    # for i in [2, 6, 20]:
    #     hLine = np.zeros((25, 25, 25), dtype=np.uint8)
    #     hLine[1, 1:2 + i, :] = 1
    #     vLine = hLine.T.copy()
    #     barImages.append(hLine)
    #     barImages.append(vLine)
    # Result graph should have _no_ cycles
    return barImages


def getRing(ri, ro, size=(25, 25)):
    """
    Make a annular ring in 2d.
    The inner and outer radius are given as a
    percentage of the overall size.
    """
    n, m = size
    xs, ys = np.mgrid[-1:1:n * 1j, -1:1:m * 1j]
    r = np.sqrt(xs ** 2 + ys ** 2)

    torus = np.zeros(size, dtype=np.uint8)
    torus[(r < ro) & (r > ri)] = 1
    return torus


def getDonut(width=2, size=(25, 25, 25)):
    x, y, z = size
    assert width < z / 2

    # This is a single planr slice of ring
    ringPlane = getRing(0.25, 0.5, size=(x, y))

    # Stack up those slices starting form the center
    donutArray = np.zeros(size, dtype=np.uint8)
    zStart = z // 2
    for n in range(width):
        donutArray[zStart + n, :, :] = ringPlane

    return donutArray


def checkAlgorithmPreservesImage(image):
    newImage = getShortestPathSkeleton(getThinned3D(image))
    assert np.array_equal(image, newImage)


def test_SinglePixelLines():
    print("checking single pixel lines")
    testImages = getStationary3DSinglePixelLines(width=0)
    for image in testImages:
        yield checkAlgorithmPreservesImage, image


def test_RandomImage():
    print("checking random images")
    randomSample = np.random.randint(2, size=(10, 10, 10))
    checkAlgorithmSinglePixeled(randomSample)


def checkAlgorithmSinglePixeled(image):
    newImage = getShortestPathSkeleton(getThinned3D(image))
    label_img, countObjectsn = ndimage.measurements.label(newImage, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if countObjects != countObjectsn:
        np.save("newImage.npy", image)
    assert (countObjectsn == countObjects) or np.sum(label_img) == 1


def checkCycles(image):
    image = getShortestPathSkeleton(getThinned3D(image))
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert(countObjects == 1)


def test_rings():
    print("checking donut")
    image = getDonut()
    yield checkCycles, image


def test_WideLines():
    testImages = get3DRolledThickLines()
    print("checking widelines")
    for image in testImages:
        yield checkAlgorithmSinglePixeled, image
