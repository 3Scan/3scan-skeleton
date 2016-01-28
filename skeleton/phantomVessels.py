import numpy as np
import random
import skimage


def makeFakeVessels(imgsize=(2048, 1024), background=230):
    """
    create and save a matrix with whitish background and randomly selected vessel sizes and save matrices generated as images of format png
    """
    nVes = 9
    mu = 20
    sigma = 5
    minw = 5
    sx, sy = imgsize
    vasc = np.ones((sx, sy), dtype=np.uint8) * background

    for i in range(nVes):
        cx, cy = random.uniform(0, sx), random.uniform(0, sy)
        r1, r2 = 0, 0
        while (r1 < minw) or (r2 < minw):
            # np.random.seed(20)
            r1 = np.random.normal(mu, sigma)
            r2 = np.random.normal(mu, sigma)
        print(r1, r2)

        rr, cc = skimage.draw.ellipse(cy, cx, r1, r2)
        if np.any(rr >= sy):
            ix = rr < sy
            rr, cc = rr[ix], cc[ix]
        if np.any(cc >= sx):
            ix = cc < sx
            rr, cc = rr[ix], cc[ix]
        vasc[rr, cc] = 1  # make circle blackish
    return vasc


def getPhantom(slices):
    vessels = _getCrosssection()
    phantom = np.zeros((slices, vessels.shape[0], vessels.shape[1]), dtype=np.uint8)
    for i in range(0, slices):
        phantom[i, :, :] = vessels
    return phantom


def _getCrosssection():
    s = np.array((512, 512))
    vessels = makeFakeVessels(s, background=0)
    return vessels


def getPhantomLineToCheckOrientation(size=(25, 25, 25)):

    hLine = np.zeros(size, dtype=bool)
    hLine[3, :, 4] = 1
    vLine = hLine.T.copy()

    # A "comb" of lines
    hLines = np.zeros(size, dtype=bool)
    hLines[0, ::3, :] = 1
    vLines = hLines.T.copy()
    # A grid made up of two perpendicular combs
    grid = hLines | vLines
    stationaryImages = [hLine, vLine, hLines, vLines, grid]
    return stationaryImages


def rotate2D(coordinates, fixedAngle=-30):
    from math import cos, sin, radians
    print(coordinates)
    x = coordinates[0]
    y = coordinates[1]
    fixedAngle = radians(fixedAngle)
    xt = x * cos(fixedAngle) - y * sin(fixedAngle)
    yt = y * cos(fixedAngle) + x * sin(fixedAngle)
    return xt, yt


def getSyntheticVasculature(size=(512, 512, 512), ro=0.5):
    # Allocate the size of synthetic vesels
    vessels = np.zeros(size, dtype=bool)
    centerList = []  # initialize lists to store centers, radius and directions
    radiusList = []
    directionList = []
    m, n, k = size  # size of the image
    rr, cc = skimage.draw.circle(256, 256, 32)  # (256, 256) = center radius = 32, gives coordinates inside the circle
    circle = np.zeros((n, k), dtype=np.bool)  # draw a circle on the first plane
    circle[rr, cc] = True
    centerList.append((256, 256))  # keeping track of radius and centers
    centerList.append((256, 256))
    radiusList.append(32)
    radiusList.append((36, 28))
    vessels[0] = circle  # set the first plane to circle
    rr, cc = skimage.draw.ellipse(256, 256, 36, 28)  # Set the second plane to ellipse
    ellipse = np.zeros((n, k), dtype=np.bool)
    ellipse[rr, cc] = True
    vessels[1] = ellipse
    for i in range(2, size[0]):
        print("ith loop --", i)
        randomProbability = np.random.ranf()
        if randomProbability < 0.07:
            if type(centerList[i - 1][0]) != int:
                print(centerList[i - 1])
                print("subscript")
                print(centerList[i - 1][0])
                for previousCenters in centerList[i - 1]:
                    numberOfBifurcations = np.random.randint(1, 6)
                    transformedCenterj = []
                    radiusj = []
                    anglej = []
                    for j in range(0, numberOfBifurcations):
                        angle = (30 / (j + 1))
                        anglej.append(angle)
                        transformedCenter = rotate2D(previousCenters, angle)
                        transformedCenterj.append(transformedCenter)
                        prevRadius = radiusList[i - 1]
                        if type(prevRadius) != int:
                            print(prevRadius)
                            radius = 0.7 * (prevRadius[0])
                        else:
                            radius = 0.7 * (prevRadius)
                        radiusj.append(radius)
                        rr, cc = skimage.draw.circle(transformedCenter[0], transformedCenter[1], radius)
                        circle1 = np.zeros((n, k), dtype=np.bool)
                        circle1[rr, cc] = True
                        vessels[i] = np.logical_or(circle1, vessels[i])
                directionList.append(anglej)
                centerList.append(transformedCenterj)
                radiusList.append(radiusj)
            else:
                transformedCenter1 = rotate2D(centerList[i - 1], 15)
                transformedCenter2 = rotate2D(centerList[i - 1], -15)
                prevRadius = radiusList[i - 1]
                if type(prevRadius) != int:
                    print(prevRadius)
                    radius = 0.7 * (prevRadius[0])
                else:
                    radius = 0.7 * (prevRadius)
                radiusList.append(tuple((radius, radius)))
                rr, cc = skimage.draw.circle(transformedCenter1[0], transformedCenter1[1], radius)
                circle1 = np.zeros((n, k), dtype=np.bool)
                circle1[rr, cc] = True
                rr, cc = skimage.draw.circle(transformedCenter2[0], transformedCenter2[1], radius)
                circle2 = np.zeros((n, k), dtype=np.bool)
                circle2[rr, cc] = True
                vessels[i] = np.logical_or(circle1, circle2)
                centerList.append((transformedCenter1, transformedCenter2))
                directionList.append((15, -15))
        else:
            print(len(radiusList))
            vessels[i] = vessels[i - 1]
            radiusList.append(radiusList[i - 1])
            centerList.append(centerList[i - 1])
            directionList.append(radiusList[i - 1])
    return vessels, centerList, radiusList, directionList

if __name__ == '__main__':
    from skeleton.convOptimize import getSkeletonize3D
    from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline, _getBouondariesOfimage
    from skeleton.unitwidthcurveskeleton import getShortestPathskeleton
    from skeleton.orientationStatisticsSpline import getStatistics
    # from density import splineInterpolateStatistics
    phantom = getPhantom(424)
    np.save('/home/pranathi/Downloads/phantom.npy', phantom)
    phantomSkel = getSkeletonize3D(phantom)
    phantomShort = getShortestPathskeleton(phantomSkel)
    np.save('/home/pranathi/Downloads/phantomShort.npy', phantomShort)
    phantomBound = _getBouondariesOfimage(phantom)
    np.save('/home/pranathi/Downloads/phantomBound.npy', phantomBound)
    dict1, dist = getRadiusByPointsOnCenterline(phantomShort, phantomBound, phantom)
    getStatistics(dict1, dist)
    # linesHandV = getPhantomLineToCheckOrientation((5, 5, 5))
    # x_knots, y_knots, z_knots, tangentVectors, normalVectors, binormalVectors, orientationPhi, orientationTheta, curvature, radiusoFCurvature = splineInterpolateStatistics(linesHandV[1])
    # assert np.unique(orientationPhi).tolist() == [90]
