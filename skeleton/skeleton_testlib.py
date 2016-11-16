import numpy as np
from scipy.spatial import ConvexHull

from skeleton.skeletonClass import Skeleton


def getThinnedRandomBlob():
    # get random convex blob
    xs = np.random.uniform(-1, 1, size=50)
    ys = np.random.uniform(-1, 1, size=50)
    zs = np.random.uniform(-1, 1, size=50)

    xyzs = list(zip(xs, ys, zs))

    hullz = ConvexHull(xyzs)

    xf, yf, zf = np.mgrid[-1:1:100j, -1:1:10j, -1:1:10j]
    blob = np.ones(xf.shape, dtype=bool)
    for x, y, z, c in hullz.equations:
        mask = (xf * x) + (yf * y) + (zf * z) - c < 0
        blob[mask] = 0
    blob = blob.astype(bool)
    skel = Skeleton(blob)
    skel.setThinningOutput()
    newImage = skel.thinnedStack
    return newImage


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


def getTinyLoopWithBranches(size=(10, 10)):
    from skimage.morphology import skeletonize as getSkeletonize2D
    # a loop and a branches coming at end of the cycle
    frame = np.zeros(size, dtype=np.uint8)
    frame[2:-2, 2:-2] = 1
    frame[4:-4, 4:-4] = 0
    frame = getSkeletonize2D(frame)
    frame[1, 5] = 1
    frame[7, 5] = 1
    sampleImage = np.zeros((3, 10, 10), dtype=np.uint8)
    sampleImage[1] = frame
    return sampleImage


def getDisjointCrosses(size=(10, 10, 10)):
    # two disjoint crosses
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    return crosPair


def getSingleVoxelLine(size=(5, 5, 5)):
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    return sampleLine


def getCycleNoTree():
    # graph of a cycle
    donut = getDonut()
    skel = Skeleton(donut)
    skel.setNetworkGraph(True)
    return skel.graph


def getCyclesWithBranchesProtrude():
    # graph of a cycle with branches
    sampleImage = getTinyLoopWithBranches()
    skel = Skeleton(sampleImage)
    skel.setNetworkGraph(False)
    return skel.graph


def getDisjointTreesNoCycle3d():
    # graph of two disjoint trees
    crosPair = getDisjointCrosses()
    skel = Skeleton(crosPair)
    skel.setNetworkGraph(False)
    return skel.graph


def getSingleVoxelLineNobranches():
    # graph of no branches single line
    sampleLine = getSingleVoxelLine()
    skel = Skeleton(sampleLine)
    skel.setNetworkGraph(False)
    return skel.graph
