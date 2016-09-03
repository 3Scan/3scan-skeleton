import numpy as np

from scipy import ndimage
from skeleton.thinVolume import getThinned

"""
Tests 3D thinning implemented as in
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999 is working as expected
from the program thinVolume.py
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


def getStationary3dRectangles(width=5):
    # cubes of different sizes
    hLine = np.zeros((25, 25, 25), dtype=bool)
    hLine[:, 8:8 + width, :] = 1
    vLine = hLine.T.copy()

    # A "comb" of lines
    hLines = np.zeros((25, 25, 25), dtype=bool)
    hLines[0:width, ::3, :] = 1
    vLines = hLines.T.copy()
    # A grid made up of two perpendicular combs
    grid = hLines | vLines
    stationaryImages = [hLine, vLine, hLines, vLines, grid]
    return stationaryImages


def get3DRandImages(width=4):
    # Random binary images
    randomImages = [np.random.randint(2, size=(25, 25, 25)) for i in range(6)]
    return randomImages


def get3DRolledThickLines():
    # grid of thick lines
    hBar = np.zeros((25, 25, 25), dtype=bool)
    hBar[1, 0:5, :] = 1
    barImages = [np.roll(hBar, 2 * n, axis=0) for n in range(10)]
    return barImages


def getRing(ri, ro, size=(25, 25)):
    # Make a annular ring in 2d. The inner and outer radius are given as a
    # percentage of the overall size.
    n, m = size
    xs, ys = np.mgrid[-1:1:n * 1j, -1:1:m * 1j]
    r = np.sqrt(xs ** 2 + ys ** 2)

    torus = np.zeros(size, dtype=bool)
    torus[(r < ro) & (r > ri)] = 1
    return torus


def getDonut(width=2, size=(25, 25, 25)):
    # Ring of width = Donut
    x, y, z = size
    assert width < z / 2, "width {} of the donut should be less than half the array size in z {}".format(width, z / 2)

    # This is a single planr slice of ring
    ringPlane = getRing(0.25, 0.5, size=(x, y))

    # Stack up those slices starting form the center
    donutArray = np.zeros(size, dtype=bool)
    zStart = z // 2
    for n in range(width):
        donutArray[zStart + n, :, :] = ringPlane

    return donutArray


def checkAlgorithmPreservesImage(image):
    newImage = getThinned(image)
    assert np.array_equal(image, newImage)


def checkCycles(image):
    # check if number of object in the donut image after thinning is 1
    image = getThinned(image)
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=bool))
    assert countObjects == 1, "number of cycles in single donut is {}".format(countObjects)


def checkSameObjects(image):
    # check if number of objects are same in input and output of thinning
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=bool))
    newImage = getThinned(image)
    label_img, countObjectsn = ndimage.measurements.label(newImage, structure=np.ones((3, 3, 3), dtype=bool))
    assert (countObjectsn == countObjects), "number of objects in input {} is different from output".format(countObjects, countObjectsn)
    return newImage


def test_donut():
    # Test 1 donut should result in a single cycle
    image = getDonut()
    yield checkCycles, image


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


def test_wideLines():
    # Test 5 All widelines should preserve topology and should have
    # same number of objects
    testImages = get3DRolledThickLines()
    for image in testImages:
        yield checkSameObjects, image


