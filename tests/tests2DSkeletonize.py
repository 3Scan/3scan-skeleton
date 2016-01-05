import numpy as np

from scipy import ndimage

from skimage.morphology import skeletonize as getskeletonize2d

"""
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.skeletonize
"""


def getStationarySinglePixelLines():
    # Test #1:
    # the answer that skeletonize gives for hLine/vLine should be the same as
    # the input, as there is no pixel that can be removed without affecting the
    # topology.
    # The algorithim when run on any of the below feature sets
    # Should return a skeleton that is the same as the feature
    # A single horizontal/vertical line
    hLine = np.zeros((255, 255), dtype=np.uint8)
    hLine[128, :] = 1
    vLine = hLine.T.copy()

    # A "comb" of lines
    hLines = np.zeros((255, 255), dtype=np.uint8)
    hLines[::3, :] = 1
    vLines = hLines.T.copy()
    # A grid made up of two perpendicular combs
    grid = hLines | vLines
    stationaryImages = [hLine, vLine, hLines, vLines, grid]
    return stationaryImages


def getRandImages():
    # Test #2:
    # The algorithim should be able to run on arbitrary input without crashing.
    # We are not interested in the answer, so much as that the algorithm has full
    # coverage over the possible inputs into it
    randomImages = [np.random.randint(2, size=(255, 255)) for i in range(6)]
    return randomImages


def getRolledThickLines():
    # Test #3:
    # The algorithim should thing a single long contiguous segment to a
    # line of pixels
    hBar = np.zeros((255, 255), dtype=np.uint8)
    hBar[0:5, :] = 1
    barImages = [np.roll(hBar, 20 * n, axis=0) for n in range(10)]
    # 2,6 and 20 pixel wide lines
    for i in [2, 6, 20]:
        hLine = np.zeros((255, 255), dtype=np.uint8)
        hLine[128:129 + i, :] = 1
        vLine = hLine.T.copy()
        barImages.append(hLine)
        barImages.append(vLine)
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


def getRings(size=(25, 25)):
    # Test #4
    # The algorithim should reduce a torus to a graph with a single cycle
    rings = []
    for ri, ro in [(0.1, 0.2), (0.4, 0.6), (0.9, 0.99)]:
        ring = getRing(ri, ro, size=size)
        rings.append(ring)
    return rings


def checkAlgorithmPreservesImage(image):
    newImage = getskeletonize2d(image)
    assert np.array_equal(image, newImage) or np.sum(newImage) <= np.sum(image)


def test_SinglePixelLines():
    testImages = getStationarySinglePixelLines()
    for image in testImages:
        yield checkAlgorithmPreservesImage, image


def getWidthOfLine(image):
    m, n = np.shape(image)
    k = np.count_nonzero(image)
    thickness = k / m
    return round(thickness)


def checkAlgorithmSinglePixeled(image):
    newImage = getskeletonize2d(image)
    width = getWidthOfLine(newImage)
    assert (width == 1)


def checkCycles(image):
    image = getskeletonize2d(image)
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3), dtype=np.uint8))
    assert(countObjects == 1)


def test_rings():
    images = getRings()
    for image in images:
        yield checkCycles, image


def test_WideLines():
    testImages = getRolledThickLines()
    for image in testImages:
        yield checkAlgorithmSinglePixeled, image
