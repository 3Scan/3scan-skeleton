import itertools

import numpy as np
from numpy import random

from scipy import ndimage
from scipy.spatial import ConvexHull

from skeleton.convOptimize import getSkeletonize3D
from skimage.morphology import skeletonize as getskeletonize2d

"""
   2D skeletonization using inbuilt python as in the below link
   http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.skeletonize
   3D Thinning using parallel 12 subiteration curve thinning by Palagyi
   http://web.inf.u-szeged.hu/ipcg/publications/papers/PalagyiKuba_GMIP1999.pdf
"""


def embed2in3(arr):
    """
       Embed a 2d shape in a 3d array,
       along all possible testing directions.
    """

    assert arr.dtype == bool
    assert arr.ndim == 2

    m, n = arr.shape
    embeddedInX = np.zeros((3, m, n), dtype=bool)
    embeddedInX[1, :, :] = arr

    embeddedInY = np.zeros((m, 3, n), dtype=bool)
    embeddedInY[:, 1, :] = arr

    embeddedInZ = np.zeros((m, n, 3), dtype=bool)
    embeddedInZ[:, :, 1] = arr

    return embeddedInX, embeddedInY, embeddedInZ


def reorders(arr):
    for xf, yf, zf in itertools.combinations_with_replacement([1, -1], 3):
        yield arr[::xf, ::yf, ::zf]


def doEmbeddedTest(arr, expectedResult=None):
    twoResult = checkCycles(arr)

    if expectedResult is not None:
        assert twoResult == expectedResult
    else:
        expectedResult = twoResult

    for embedding in embed2in3(arr):
        allOrientationsTest(embedding, expectedResult)

    return twoResult


def allOrientationsTest(arr, expectedResult=None):
    assert arr.ndim == 3
    i = 0
    for reoriented in reorders(arr):
        i += 1
        result = checkCycles(reoriented)
        assert result == expectedResult


def checkCycles(image):
    # count number of 26 or 8 connected objects in the skeletonized image
    if image.ndim == 2:
        skel = getskeletonize2d(image)
        label_skel, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3), dtype=bool))
    else:
        skel = getSkeletonize3D(image)
        label_skel, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=bool))
    return countObjects


# One single tiny loop
tinyLoop = np.array([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]], dtype=bool)

# This is a simple suite of test cases for the
# Algo which takes in skeletonized images, and emits
# networkx graphs
# Some quick and dumb tests against the core algorithim


def test_simpleLoopEmbedded():
    doEmbeddedTest(tinyLoop, 1)


# Three independant loops
multiLoop = np.zeros((25, 25), dtype=bool)
multiLoop[2:5, 2:5] = tinyLoop
multiLoop[7:10, 7:10] = tinyLoop


def test_multiLoopEmbedded():
    doEmbeddedTest(multiLoop, 2)

cros = np.zeros((25, 25), dtype=bool)
cros[:, 12] = 1
cros[12, :] = 1


def test_crossEmbedded():
    doEmbeddedTest(cros, 1)


hillbert = np.array([[[1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 1]],
                     [[0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 1]],
                     [[1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 1]]], dtype=bool)


def test_hillbert():
    allOrientationsTest(hillbert, expectedResult=1)

loopPair = np.array([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]], dtype=bool)


def test_loop():
    doEmbeddedTest(loopPair, 1)


squae = np.zeros((20, 20), dtype=bool)
squae[2:-2, 2:-2] = 1


def test_square():
    doEmbeddedTest(squae, 1)


parallelepiped = np.zeros((10, 10, 10), dtype=bool)
parallelepiped[2:-2, 2:-2, 2:-2] = 1


def test_parallelepiped():
    allOrientationsTest(parallelepiped, expectedResult=1)

frame = np.zeros((20, 20), dtype=bool)

frame[2:-2, 2:-2] = 1
frame[4:-4, 4:-4] = 0

frame3d = np.zeros((10, 10, 10), dtype=bool)
frame3d[2:-2, 2:-2, 2:-2] = 1
frame3d[4:-4, 4:-4, 4:-4] = 0


def test_frame3d():
    allOrientationsTest(frame3d, 1)


def test_frame():
    c = doEmbeddedTest(frame)
    assert c == 1

framedSquare = frame.copy()
framedSquare[6:-6, 6:-6] = 1


def test_framedSquare():
    d = doEmbeddedTest(framedSquare)
    assert d == 2


def test_convex2DBlob():
    xs = np.random.uniform(-1, 1, size=5)
    ys = np.random.uniform(-1, 1, size=5)

    xys = list(zip(xs, ys))

    hull = ConvexHull(xys)

    xf, yf = np.mgrid[-1:1:50j, -1:1:50j]
    i = np.ones(xf.shape, dtype=bool)
    for x, y, c in hull.equations:
        mask = (xf * x) + (yf * y) - c < 0
        i[mask] = 0
    c = doEmbeddedTest(i)
    assert c == 1


def test_banana():
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    xf, yf = np.mgrid[-1.5:3:50j, -1.5:2:50j]
    f = (1 - xf) ** 2 + 100 * (yf - xf ** 2) ** 2
    i = np.zeros(xf.shape, dtype=bool)
    i = 1 * (f > 250)
    i = i.astype(bool)
    doEmbeddedTest(i)

hevi = np.zeros((20, 20), dtype=bool)
hevi[10:, :] = 1


def test_circle():
    i = np.zeros((25, 25), dtype=bool)
    xs, ys = np.mgrid[-1:1:25j, -1:1:25j]

    for trial in range(5):
        i[:] = 0
        r = np.random.uniform(3, 10)
        xc, yc = np.random.uniform(-1, 1, size=2)
        mask = ((xs ** 2) + (ys ** 2)) < r ** 2
        i[mask] = 1

        d = doEmbeddedTest(i)
        assert d == 1


def test_Heaviside():
    doEmbeddedTest(hevi, 1)


def getRing(ri, ro, size=(25, 25)):
    """
    Make a annular ring in 2d.
    The inner and outer radius are given as a
    percentage of the overall size.
    """
    n, m = size
    xs, ys = np.mgrid[-1:1:n * 1j, -1:1:m * 1j]
    r = np.sqrt(xs ** 2 + ys ** 2)

    torus = np.zeros(size, dtype=bool)
    torus[(r < ro) & (r > ri)] = 1
    return torus

concentricCircles = getRing(0.1, 0.2) + getRing(0.4, 0.5) + getRing(0.7, 0.9)


def test_ellipse():
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
        assert c == 1


def test_concentric():
    c = doEmbeddedTest(concentricCircles)
    assert c == 3


def test_convex3DBlob():
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
    allOrientationsTest(i, 1)
