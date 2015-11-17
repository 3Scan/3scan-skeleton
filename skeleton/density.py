import copy
import numpy as np
import time
from radiusOfNodes import getRadiusByPointsOnCenterline


thresholdIm = np.load('/Users/3scan_editing/records/scratch/threshold3dtestoptimize.npy')
inputIm = np.load('/Users/3scan_editing/records/scratch/input3dtestoptimizeWoBoundary.npy')
shskel = np.load("/Users/3scan_editing/records/scratch/shortestPathSkel.npy")
boundaryIm = np.load('/Users/3scan_editing/records/scratch/boundaries.npy')
skeletonIm = np.load('/Users/3scan_editing/records/skeleton3d.npy')

# skelAndBound = shskel + boundaryIm
# ball1=skimage.morphology.ball(4, bool)
# label, countObjectsInput = ndimage.measurements.label(inputIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
# countObjectsInput = 1
# label1, countObjectsThreshold = ndimage.measurements.label(thresholdIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
# countObjectsThreshold = 1069

# vesselVoxelsInGreyscaleImage = len(list(np.transpose(np.nonzero(inputIm))))
# vesselVoxelsInGreyscaleImage =

vesselVoxelsInThresholdedImage = len(list(np.transpose(np.nonzero(thresholdIm))))
# vesselVoxelsInThresholdedImage =
from collections import Counter

resolution = 0.7 * 0.7 * 10
boxVolume = vesselVoxelsInThresholdedImage * resolution
dictOfNodesAndRadius, distTransformedIm = getRadiusByPointsOnCenterline(shskel, boundaryIm)
listRadius = list(dictOfNodesAndRadius.values())
from statistics import median, mode, pvariance, mean
meanRadius = mean(listRadius)
varianceRadius = (pvariance(listRadius, meanRadius))
radiusCounts = Counter(listRadius)
from math import sqrt
import json
dictStats = {'isolatedVoxels': list(dictOfNodesAndRadius.values()).count(0.0), 'maxRadius': max(listRadius), 'minRadius': min(listRadius),
        'meanRadius': mean(listRadius),
        'medianRadius': median(listRadius), 'modeRadius': mode(listRadius),
        'varianceRadius': varianceRadius, 'stddevRadius': sqrt(varianceRadius),
        'uniquevessels': len(radiusCounts)}
with open("statistics", "w") as outfile:
    json.dump(dictStats, outfile, indent=4)


def outOfPixBOunds(neighborCoordinate, aShape):
    zMax, yMax, xMax = aShape
    isAtZBoundary = neighborCoordinate[0] == zMax
    isAtYBoundary = neighborCoordinate[1] == yMax
    isAtXBoundary = neighborCoordinate[2] == xMax
    if isAtXBoundary or isAtYBoundary or isAtZBoundary:
        return 0
    else:
        return 1


def getDensityOfRegion(regionBounds, shskel):
    aShape = shskel.shape
    neighborCoordinate = (regionBounds[1], regionBounds[3], regionBounds[5])
    neighborCoordinate2 = (regionBounds[0], regionBounds[2], regionBounds[4])
    if len(set(regionBounds)) >= 3 and outOfPixBOunds(neighborCoordinate, aShape) and outOfPixBOunds(neighborCoordinate2, aShape):
        thresholdImRegion = thresholdIm[regionBounds[0]:regionBounds[1], regionBounds[2]:regionBounds[3], regionBounds[4]:regionBounds[5]]
        regionSize = len(list(np.transpose(np.nonzero(thresholdImRegion))))
        regionVolume = regionSize * resolution
        densityOfRegion = regionVolume / boxVolume
        return densityOfRegion
    else:
        print("not a valid region, doesn't occupy a volume, so density is zero")
        return 0

densityOfRegion = getDensityOfRegion([200, 300, 256, 511, 256, 511], thresholdIm)

radiusThreshold = max(dictOfNodesAndRadius.values())
radiusArray = distTransformedIm


def filterTreeStructureByRadius(radiusThreshold, shskel, skeletonIm, radiusArray):
    shSkelTree = copy.deepcopy(shskel)
    shSkelNet = copy.deepcopy(shskel)
    skeletonImTree = copy.deepcopy(skeletonIm)
    skeletonImTree[radiusArray < radiusThreshold - 3] = 0
    shSkelTree[radiusArray < radiusThreshold - 3] = 0
    shSkelNet[radiusArray > radiusThreshold - 3] = 0
    return radiusArray, shSkelTree, shSkelNet, skeletonImTree

rad, shskelTree, shSkelNet, skeletonImTree = filterTreeStructureByRadius(radiusThreshold, shskel, skeletonIm, radiusArray)


def transformSphericalToCartesian(r, q, p):
    """Converts from spherical to cartesian co-ordinates."""
    from numpy import sin, cos
    x = r * sin(q) * cos(p)
    y = r * sin(q) * sin(p)
    z = r * cos(q)
    return x, y, z


def setCartesianToSpherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 1] ** 2 + xyz[:, 2] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 0] ** 2)
    ptsnew[:, 4] = np.arctan2(np.sqrt(xy), xyz[:, 0])  # for elevation angle defined from Z-axis down
    #  ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 2])
    return ptsnew


def sphere(x, y, z, a, b, c, r):
    """Returns zero if (x,y,z) lies on a sphere centred at (a,b,c) with radius r."""
    return (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - r ** 2


# # Create some points approximately spherical distribution
xyz = np.array(np.where(shskel != 0))
ptsnew = setCartesianToSpherical(xyz)
x = ptsnew[2]; y = ptsnew[1]; z = ptsnew[0]
radiuscCoordinate = x[len(xyz[0]):]
elevationCoordinate = y[len(xyz[0]):]
azimuthCoordinate = z[len(xyz[0]):]
from statistics import mean
elevationMean = mean(elevationCoordinate)
azimuthMean = mean(azimuthCoordinate)


"""
spline curve fitting
"""

from scipy import interpolate
from math import degrees, atan2
z_sample, y_sample, x_sample = np.array(np.where(shskel != 0))
num_true_pts = len(x_sample)
starttInterp = time.time()
tck, u = interpolate.splprep([x_sample, y_sample, z_sample], s=0)
print("interpolate.splprep done and took %i seconds", time.time() - starttInterp)
starttKnots = time.time()
# x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
firstDerX, firstDerY, firstDerZ = interpolate.splev(tck[0], tck, der=1)
secondDerX, secondDerY, secondDerZ = interpolate.splev(tck[0], tck, der=2)
import json
# u_fine = np.linspace(0, 1, num_true_pts)
# z_fine, y_fine, x_fine = interpolate.splev(u_fine, tck)
print("interpolate.splev done and took %i seconds", time.time() - starttKnots)


orientationTheta = np.empty(len(firstDerX))
orientationPhi = np.empty(len(firstDerX))
for i in range(0, len(firstDerX)):
    orientationTheta[i] = degrees(atan2(firstDerZ[i], firstDerX[i]))

    orientationPhi[i] = degrees(atan2(firstDerY[i], firstDerX[i]))

from statistics import median, mode, pvariance
with open("statistics", "w") as outfile:
    json.dump({'orientationPhiMean': orientationPhi.mean(), 'orientationPhiMax': orientationPhi.max(), 'orientationPhiMin': orientationPhi.min(),
        'orientationThetaMean': orientationTheta.mean(), 'orientationThetaMax': orientationTheta.max(), 'orientationThetaMin': orientationTheta.min(),
        'orientationThetaMedian': median(orientationTheta), 'orientationPhiMedian': median(orientationPhi), 'orientationThetaMode': mode(orientationTheta),
        'orientationPhiVariance': pvariance(orientationPhi), 'orientationThetaVariance': pvariance(orientationTheta)}, outfile, indent=4)
# PLOTTING
import seaborn as sns
radArray = np.array(list(dictOfNodesAndRadius.values()))
# KDE
sns.distplot(radArray, kde=True, bins=20)
# Histogram
sns.distplot(radArray, kde=False, rug=True)
sns.distplot(orientationTheta, kde=True)
sns.distplot(orientationPhi, kde=True)
# import matplotlib.pyplot as plt
# fig2 = plt.figure(2)
# ax3d = fig2.add_subplot(111, projection='rectilinear')
# ax3d.plot(z_sample, y_sample, x_sample, 'r*')
# ax3d.plot(z_knots, y_knots, x_knots, 'go')
# ax3d.plot(z_fine, y_fine, x_fine, 'g')
# fig2.show()
