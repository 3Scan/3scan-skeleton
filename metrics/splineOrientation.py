
import time

from scipy import interpolate

import numpy as np

from math import sqrt, pow, atan2, degrees
from statistics import median, pvariance
from skeleton.plotStats import saveDictAsJson


def splineInterpolateStatistics(shskel, path=None):
    """
       assumes an array with z as it's first dimension
       spline curve fitting and orientation finding from
       the derivatives
    """
    z_sample, y_sample, x_sample = np.array(np.where(shskel != 0))
    starttInterp = time.time()
    tck, u = interpolate.splprep([z_sample, y_sample, x_sample])
    print("interpolate.splprep done and took %0.3f seconds" % (time.time() - starttInterp))
    starttKnots = time.time()
    z_knots, y_knots, x_knots = interpolate.splev(tck[0], tck)
    firstDerZ, firstDerY, firstDerX = interpolate.splev(tck[0], tck, der=1)
    secondDerZ, secondDerY, secondDerX = interpolate.splev(tck[0], tck, der=2)
    print("interpolate.splev done and took %0.3f seconds" % (time.time() - starttKnots))
    orientationTheta = np.empty(len(firstDerX))
    orientationPhi = np.empty(len(firstDerX))
    for i in range(0, len(firstDerX)):
        orientationTheta[i] = degrees(atan2(firstDerZ[i], firstDerX[i]))
        orientationPhi[i] = degrees(atan2(firstDerY[i], firstDerX[i]))
    lenFirstDer = firstDerX.size
    tangentVectorsX = np.empty(lenFirstDer)
    tangentVectorsY = np.empty(lenFirstDer)
    tangentVectorsZ = np.empty(lenFirstDer)
    tangentVectors = []
    for i in range(0, lenFirstDer):
        modOfVect = (sqrt(pow(firstDerX[i], 2) + pow(firstDerY[i], 2) + pow(firstDerZ[i], 2)))
        tangentVectorsX[i] = firstDerX[i] / modOfVect
        tangentVectorsY[i] = firstDerY[i] / modOfVect
        tangentVectorsZ[i] = firstDerZ[i] / modOfVect
        tangentVectors.append(np.array((tangentVectorsX[i], tangentVectorsY[i], tangentVectorsZ[i])))

    normalVectorsX = np.empty(lenFirstDer)
    normalVectorsY = np.empty(lenFirstDer)
    normalVectorsZ = np.empty(lenFirstDer)
    normalVectors = []
    for i in range(0, lenFirstDer):
        dotProduct = ((secondDerX[i] * tangentVectorsX[i]) + (secondDerY[i] * tangentVectorsY[i]) + (secondDerZ[i] * tangentVectorsZ[i]))
        numeratorNormX = secondDerX[i] - dotProduct * tangentVectorsX[i]
        numeratorNormY = secondDerY[i] - dotProduct * tangentVectorsY[i]
        numeratorNormZ = secondDerZ[i] - dotProduct * tangentVectorsZ[i]
        modOfVect = (sqrt(pow(numeratorNormX, 2) + pow(numeratorNormY, 2) + pow(numeratorNormZ, 2)))
        normalVectorsX[i] = numeratorNormX / modOfVect
        normalVectorsY[i] = numeratorNormY / modOfVect
        normalVectorsZ[i] = numeratorNormZ / modOfVect
        normalVectors.append(np.array((normalVectorsX[i], normalVectorsY[i], normalVectorsZ[i])))

    binormalVectors = []
    for (a, b) in zip(tangentVectors, normalVectors):
        binormalVectors.append(np.cross(a.T, b.T))

    curvature = np.empty(lenFirstDer)
    radiusoFCurvature = np.empty(lenFirstDer)
    for i in range(0, lenFirstDer):
        numeratorDotProduct = ((secondDerX[i] * normalVectorsX[i]) + (secondDerY[i] * normalVectorsY[i]) + (secondDerZ[i] * normalVectorsZ[i]))
        denomMod = pow((sqrt(pow(firstDerX[i], 2) + pow(firstDerY[i], 2) + pow(firstDerZ[i], 2))), 2)
        curvature[i] = numeratorDotProduct / denomMod
        radiusoFCurvature[i] = 1 / curvature[i]
    dictStats = {'orientationPhiMean': orientationPhi.mean(), 'orientationPhiMax': orientationPhi.max(), 'orientationPhiMin': orientationPhi.min(),
                 'orientationThetaMean': orientationTheta.mean(), 'orientationThetaMax': orientationTheta.max(), 'orientationThetaMin': orientationTheta.min(),
                 'orientationThetaMedian': median(orientationTheta), 'orientationPhiMedian': median(orientationPhi),
                 'orientationPhiVariance': pvariance(orientationPhi), 'orientationThetaVariance': pvariance(orientationTheta)}
    saveDictAsJson(dictStats, path)
    return x_knots, y_knots, z_knots, tangentVectors, normalVectors, binormalVectors, orientationPhi, orientationTheta, curvature, radiusoFCurvature


def ravel_index(x, dims):
    """
       function that converts a 3D coordinate to a
       one dimensional number similar to reshaping
       indices
    """
    i = 0
    for dim, j in zip(dims, x):
        i = i * dim
        i += j
    return i


def list_to_dict(listNZI, skeletonLabelled):
    """
       converts an array and their value to a dictionary with keys as the
       coordinate and the values as the value of the voxel/pixel
    """
    dictOfIndicesAndlabels = {item: skeletonLabelled[index] for index, item in enumerate(listNZI)}
    return dictOfIndicesAndlabels


if __name__ == '__main__':
    shskel = np.load(input("please enter a path to your unit width voxelized skeleton"))
    (x_knots, y_knots, z_knots, tangentVectors, normalVectors, binormalVectors,
     orientationPhi, orientationTheta, curvature, radiusoFCurvature) = splineInterpolateStatistics(shskel)
