import numpy as np
import skimage


def rotate2D(coordinates, fixedAngle=-30):
    from math import cos, sin, radians
    x = coordinates[0]
    y = coordinates[1]
    fixedAngle = radians(fixedAngle)
    xt = x * cos(fixedAngle) - y * sin(fixedAngle)
    yt = y * cos(fixedAngle) + x * sin(fixedAngle)
    return xt, yt


def getSyntheticVasculature(size=(512, 512, 512), ro=0.07):
    # Allocate the size of synthetic vesels
    vessels = np.zeros(size, dtype=bool)
    centerList = []  # initialize lists to store centers, radius and directions
    radiusList = []
    directionList = []
    m, n, k = size  # size of the image
    rr, cc = skimage.draw.circle(256, 256, 16)  # (256, 256) = center radius = 32, returns all of coordinates within the circle
    centerList.append((256, 256))  # keeping track of radius and centers
    radiusList.append(32)
    vessels[0, rr, cc] = True  # set the first plane to circle - z plane in python
    # for the rest of the planes do the following
    for i in range(1, size[0]):
        print("ith loop --", i)
        randomProbability = np.random.ranf()  # if a selected random number is less than 0.07,bifurcation folows in the plane
        if randomProbability < ro:
            if type(centerList[i - 1][0]) != int:  # if there was a single center in the previous plane
                for previousCenters in centerList[i - 1]:  # for each of the centers of ellipses in center list of previous plane
                    numberOfBifurcations = np.random.randint(1, 6)  # more bifurcations = random number between 1 and 6
                    transformedCenterj = []  # the centers are roated by an angle with reference to origin of the 2D plane
                    radiusj = []  # initialize list to store radius and angle for each of the bifurcations
                    anglej = []
                    for j in range(0, numberOfBifurcations):  # for each of the random bifurcations
                        angle = (40 / (j + 1))  # angle randomized based on bifurcation
                        anglej.append(angle)  # store angle
                        transformedCenter = rotate2D(previousCenters, angle)  # rotate center by the given angle
                        print("tc", transformedCenter, "angle", angle)
                        transformedCenterj.append(transformedCenter)   # store the transformed center
                        assert np.sum([item for item in transformedCenter if item < 1]) == 0  # assert the coordinates are not negative
                        prevRadius = radiusList[i - 1]   # change the radius of the points here based on previous radius
                        if type(prevRadius) != int:  # if the previous radius is not a single number i.e there was one circle in the previous plane
                            radius = 0.7 * (prevRadius[0])  # 0.7 is a random number choosen
                        else:
                            radius = 0.7 * (prevRadius)
                        radiusj.append(radius)  # store the radius
                        rr, cc = skimage.draw.circle(transformedCenter[0], transformedCenter[1], radius)  # get the coordinates inside the circle of radius found
                        assert np.sum([item for item in rr.tolist() if item < 1]) == 0  # assert the coordinates are not negative
                        assert np.sum([item for item in cc.tolist() if item < 1]) == 0  # assert the coordinates are not negative
                        vessels[i, rr, cc] = np.logical_or(circle1, vessels[i, rr, cc])  # vessels so far is the logical or of all the bifurcations
                directionList.append(anglej)  # keeping track of radius and direcions
                centerList.append(transformedCenterj)
                radiusList.append(radiusj)
            else:  # for the second plane that has single center , create two bifurcations by default (dichtomous tree)
                transformedCenter1 = rotate2D(centerList[i - 1], 40)  # bifurcations default rotation = 40 degrees
                print("tc-bi1------", transformedCenter1)
                transformedCenter2 = rotate2D(centerList[i - 1], 320)  # bifurcations default rotation = -40 degrees
                print("tc-bi2------", transformedCenter1)
                assert np.sum([item for item in transformedCenter1 if item < 1]) == 0  # assert the coordinates are not negative
                assert np.sum([item for item in transformedCenter2 if item < 1]) == 0  # assert the coordinates are not negative
                prevRadius = radiusList[i - 1]
                if type(prevRadius) != int:
                    radius = 0.7 * (prevRadius[0])
                else:
                    radius = 0.7 * (prevRadius)
                radiusList.append(tuple((radius, radius)))
                rr, cc = skimage.draw.circle(transformedCenter1[0], transformedCenter1[1], radius)  # two circles at an angle to the previous plane
                assert np.sum([item for item in rr.tolist() if item < 1]) == 0  # assert the coordinates are not negative
                assert np.sum([item for item in cc.tolist() if item < 1]) == 0  # assert the coordinates are not negative
                circle1 = np.zeros((n, k), dtype=np.bool)
                circle1[rr, cc] = True
                rr, cc = skimage.draw.circle(transformedCenter2[0], transformedCenter2[1], radius)
                assert np.sum([item for item in rr.tolist() if item < 1]) == 0  # assert the coordinates are not negative
                assert np.sum([item for item in cc.tolist() if item < 1]) == 0  # assert the coordinates are not negative
                circle2 = np.zeros((n, k), dtype=np.bool)
                circle2[rr, cc] = True
                vessels[i] = np.logical_or(circle1, circle2)
                centerList.append((transformedCenter1, transformedCenter2))
                directionList.append((15, -15))
        else:
            vessels[i] = vessels[i - 1]
            radiusList.append(radiusList[i - 1])
            centerList.append(centerList[i - 1])
            directionList.append(radiusList[i - 1])
    return vessels, centerList, radiusList, directionList

if __name__ == '__main__':
    vessels, centerList, radiusList, directionList = getSyntheticVasculature()
    # from skeleton.convOptimize import getSkeletonize3D
    # from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline, _getBouondariesOfimage
    # from skeleton.unitwidthcurveskeleton import getShortestPathskeleton
    # from skeleton.orientationStatisticsSpline import getStatistics
    # # from density import splineInterpolateStatistics
    # phantom = getPhantom(424)
    # np.save('/home/pranathi/Downloads/phantom.npy', phantom)
    # phantomSkel = getSkeletonize3D(phantom)
    # phantomShort = getShortestPathskeleton(phantomSkel)
    # np.save('/home/pranathi/Downloads/phantomShort.npy', phantomShort)
    # phantomBound = _getBouondariesOfimage(phantom)
    # np.save('/home/pranathi/Downloads/phantomBound.npy', phantomBound)
    # dict1, dist = getRadiusByPointsOnCenterline(phantomShort, phantomBound, phantom)
    # getStatistics(dict1, dist)
    # linesHandV = getPhantomLineToCheckOrientation((5, 5, 5))
    # x_knots, y_knots, z_knots, tangentVectors, normalVectors, binormalVectors, orientationPhi, orientationTheta, curvature, radiusoFCurvature = splineInterpolateStatistics(linesHandV[1])
    # assert np.unique(orientationPhi).tolist() == [90]
