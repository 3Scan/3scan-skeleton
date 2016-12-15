import itertools

import numpy as np
from numpy import random
from scipy import ndimage
from skimage.morphology import skeletonize as getskeletonize2d

from skeleton.thinVolume import get_thinned
from skeleton.skeleton_testlib import getRing

"""
Tests for 2D and 3D thinning algorithms testing strictly with change in directions, axis
2D thinning using inbuilt python as in the below link
http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.skeletonize
3D Thinning implemented using
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
"""


def embed2in3(arr):
    # Embed a 2D shape in a 3D array, along all possible testing directions.
    assert arr.dtype == bool, "input array type is {}. should be boolean".format(arr.dtype)
    assert arr.ndim == 2, "number of dimensions in input array is {}, should be 2".fomrat(arr.ndim)

    m, n = arr.shape
    embeddedInX = np.zeros((3, m, n), dtype=bool)
    embeddedInX[1, :, :] = arr

    embeddedInY = np.zeros((m, 3, n), dtype=bool)
    embeddedInY[:, 1, :] = arr

    embeddedInZ = np.zeros((m, n, 3), dtype=bool)
    embeddedInZ[:, :, 1] = arr

    return embeddedInX, embeddedInY, embeddedInZ


def reorders(arr):
    # reorder the slices, rows, columns of 3D stack
    for xf, yf, zf in itertools.combinations_with_replacement([1, -1], 3):
        yield arr[::xf, ::yf, ::zf]


def doEmbeddedTest(arr, expectedResult=None):
    # number of objects at different embeddings of the array
    # should return a result as in expectedResult
    twoResult = _getCountObjects(arr)

    if expectedResult is not None:
        assert twoResult == expectedResult, "twoResult {} is not same as expectedResult {}".format(twoResult,
                                                                                                   expectedResult)
    else:
        expectedResult = twoResult

    for embedding in embed2in3(arr):
        _allOrientationsTest(embedding, expectedResult)
    return twoResult


def _allOrientationsTest(arr, expectedResult=None):
    assert arr.ndim == 3, "number of dimensions {} in input array is not 3".fomrat(arr.ndim)
    i = 0
    for reoriented in reorders(arr):
        i += 1
        result = _getCountObjects(reoriented)
        assert result == expectedResult, "result {} is not same as expectedResult {}".format(result, expectedResult)


def _getCountObjects(image):
    # count number of 26 or 8 connected objects in the skeletonized image
    if image.ndim == 2:
        skel = getskeletonize2d(image)
        label_skel, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3), dtype=bool))
    else:
        skel = get_thinned(image)
        label_skel, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=bool))
    return countObjects

# One single tiny loop
tinyLoop = np.array([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]], dtype=bool)

# Frame
frame = np.zeros((20, 20), dtype=bool)
frame[2:-2, 2:-2] = 1
frame[4:-4, 4:-4] = 0


def test_simpleLoopEmbedded():
    # Test 1 a single loop embedded in different axis locations
    doEmbeddedTest(tinyLoop, 1)


def test_multiLoopEmbedded():
    # Test 2 Three independent loops embedded in different axis locations
    multiLoop = np.zeros((25, 25), dtype=bool)
    multiLoop[2:5, 2:5] = tinyLoop
    multiLoop[7:10, 7:10] = tinyLoop
    doEmbeddedTest(multiLoop, 2)


def test_crossEmbedded():
    # Test 3 cross embedded in different axis locations
    cros = np.zeros((25, 25), dtype=bool)
    cros[:, 12] = 1
    cros[12, :] = 1
    doEmbeddedTest(cros, 1)


def test_loop():
    # Test 4 Two joint loops embedded in different axis locations
    loopPair = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=bool)
    doEmbeddedTest(loopPair, 1)


def test_square():
    # Test 5 Square embedded in different axis locations
    squae = np.zeros((20, 20), dtype=bool)
    squae[2:-2, 2:-2] = 1
    doEmbeddedTest(squae, 1)


def test_frame():
    # Test 6 Frame (hollow square) embedded in different axis locations
    c = doEmbeddedTest(frame)
    assert c == 1, "number of objects in the frame should be 1, but it is {}".format(c)


def test_framedSquare():
    # Test 7 Square inside a Frame (hollow square) embedded in different axis locations
    framedSquare = frame.copy()
    framedSquare[6:-6, 6:-6] = 1
    d = doEmbeddedTest(framedSquare)
    assert d == 2, "number of objects in the framed square should be 2, but it is {}".format(d)


def test_circle():
    # Test 8 Circle embedded in different axis locations
    i = np.zeros((25, 25), dtype=bool)
    xs, ys = np.mgrid[-1:1:25j, -1:1:25j]

    for trial in range(5):
        i[:] = 0
        r = np.random.uniform(3, 10)
        xc, yc = np.random.uniform(-1, 1, size=2)
        mask = ((xs ** 2) + (ys ** 2)) < r ** 2
        i[mask] = 1

        c = doEmbeddedTest(i)
        assert c == 1, "number of objects in the circle should be 1, but it is {}".format(c)


def test_heaviside():
    # Test 9 Heaviside(comb) embedded in different axis locations
    heavi = np.zeros((20, 20), dtype=bool)
    heavi[10:, :] = 1
    doEmbeddedTest(heavi, 1)


def test_ellipse():
    # Test 10 Ellipse embedded in different axis locations
    i = np.zeros((25, 25), dtype=bool)
    xs, ys = np.mgrid[-1:1:25j, -1:1:25j]

    aspect = random.randint(1, 2) / 10

    for trial in range(5):
        i[:] = 0
        r = np.random.uniform(3, 10)
        mask = (aspect * ((xs ** 2) + (ys ** 2))) < r ** 2
        i[mask] = 1
        i = i.astype(bool)
        c = doEmbeddedTest(i)
        assert c == 1, "number of objects in the ellipse should be 1, but it is {}".format(c)


def test_concentric():
    # Test 11 Concentric circles embedded in different axis locations
    concentricCircles = getRing(0.1, 0.2) + getRing(0.4, 0.5) + getRing(0.7, 0.9)
    c = doEmbeddedTest(concentricCircles)
    assert c == 3, "number of objects in the concentric circles should be 3, but it is {}".format(c)


def test_banana():
    # Test 12 Banana embedded in different axis locations
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    xf, yf = np.mgrid[-1.5:3:50j, -1.5:2:50j]
    f = (1 - xf) ** 2 + 100 * (yf - xf ** 2) ** 2
    i = 1 * (f > 250)
    i = i.astype(bool)
    doEmbeddedTest(i)


def test_hillbert():
    # Test 13 Hillbert flipped in different orientations
    hillbert = np.array([[[1, 1, 1],
                          [1, 0, 1],
                          [1, 0, 1]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [1, 0, 1]],
                         [[1, 1, 1],
                          [1, 0, 1],
                          [1, 0, 1]]], dtype=bool)
    _allOrientationsTest(hillbert, expectedResult=1)


def test_parallelepiped():
    # Test 14 Parallelepiped flipped in different orientations
    parallelepiped = np.zeros((10, 10, 10), dtype=bool)
    parallelepiped[2:-2, 2:-2, 2:-2] = 1
    _allOrientationsTest(parallelepiped, expectedResult=1)


def test_frame3d():
    # Test 15 3D frame flipped in different orientations
    frame3d = np.zeros((10, 10, 10), dtype=bool)
    frame3d[2:-2, 2:-2, 2:-2] = 1
    frame3d[4:-4, 4:-4, 4:-4] = 0
    _allOrientationsTest(frame3d, 1)
