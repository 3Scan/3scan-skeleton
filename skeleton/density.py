import json
import time

from scipy import interpolate

import numpy as np
import scipy
import seaborn as sns

from collections import Counter
from math import sqrt, pow, atan2, degrees
from statistics import median, mode, pvariance, mean

from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline

"""
   The follwing program quantifies a skeletonized structure to meaningful
   statistics like radius, orienation at each node, mean radiuses of the
   nodes in a volume and other statistically significant values
"""


def saveVolumeFeatures(image, nameOfTheImage):
    """
       count number of objects and number of nonzero voxels
       and save them to a json
    """
    date = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
    t = 'countObjects%' + date
    t = t.replace('%', nameOfTheImage)
    label, countObjectsInput = scipy.ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    dictStats = {t: countObjectsInput}
    with open("statistics.json", "a") as feedsjson:
        feedsjson.write("{}\n".format(json.dumps(dictStats)))
    feedsjson.close()


def saveVolumeMetadata(inputIm):
    """
       save 3D volume meta data like resolution
       and volume of the dataset
    """
    resolution = 0.7 * 0.7 * 10
    volumeOfDataset = inputIm.size * resolution
    dictStats = {'resolution': resolution,
                 'volumeOfDataset': volumeOfDataset}
    with open("statistics.json", "a") as feedsjson:
        feedsjson.write("{}\n".format(json.dumps(dictStats)))
    feedsjson.close()


def getRadisuStatistics(dictOfNodesAndRadius, distTransformedIm):
    listRadius = [dictOfNodesAndRadius[k] for k in dictOfNodesAndRadius]
    meanRadius = mean(listRadius)
    varianceRadius = (pvariance(listRadius, meanRadius))
    radiusCounts = Counter(listRadius)
    dictStats = {'isolatedVoxels': listRadius.count(0.0), 'maxRadius': max(listRadius), 'minRadius': min(listRadius),
                 'meanRadius': mean(listRadius),
                 'medianRadius': median(listRadius), 'modeRadius': mode(listRadius),
                 'varianceRadius': varianceRadius, 'stddevRadius': sqrt(varianceRadius),
                 'uniquevessels': len(radiusCounts)}
    with open("statistics.json", "a") as feedsjson:
        feedsjson.write("{}\n".format(json.dumps(dictStats)))
    feedsjson.close()


def plotKDEAndHistogram(ndimarray, bins):
    sns.distplot(ndimarray, kde=True, bins=bins)


def splineInterpolateStatistics(shskel, aspectRatio=[1, 1, 1]):
    """
       spline curve fitting and orientation finding from
       the derivatives
    """
    interpolatedSkeleton = scipy.ndimage.interpolation.zoom(shskel, zoom=aspectRatio, order=0)
    z_sample, y_sample, x_sample = np.array(np.where(interpolatedSkeleton != 0))
    num_true_pts = len(x_sample)
    starttInterp = time.time()
    tck, u = interpolate.splprep([z_sample, y_sample, x_sample])
    print("interpolate.splprep done and took %i seconds", time.time() - starttInterp)
    starttKnots = time.time()
    z_knots, y_knots, x_knots = interpolate.splev(tck[0], tck)
    firstDerZ, firstDerY, firstDerX = interpolate.splev(tck[0], tck, der=1)
    secondDerZ, secondDerY, secondDerX = interpolate.splev(tck[0], tck, der=2)
    u_fine = np.linspace(0, 1, num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    print("interpolate.splev done and took %i seconds", time.time() - starttKnots)
    orientationTheta = np.empty(len(firstDerX))
    orientationPhi = np.empty(len(firstDerX))
    for i in range(0, len(firstDerX)):
        orientationTheta[i] = degrees(atan2(firstDerZ[i], firstDerX[i]))
        orientationPhi[i] = degrees(atan2(firstDerY[i], firstDerX[i]))
    binsTheta = np.unique(np.round(orientationTheta, 0))
    print("number of unique orientationThetas are", len(binsTheta))
    binsPhi = np.unique(np.round(orientationPhi, 0))
    print("number of unique orientationPhi are", len(binsPhi))

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
        numeratorNormX = secondDerX[i] - dotProduct * tangentVectorsX[i]; numeratorNormY = secondDerY[i] - dotProduct * tangentVectorsY[i];
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
    with open("statistics.json", "a") as feedsjson:
        feedsjson.write("{}\n".format(json.dumps(dictStats)))
    feedsjson.close()
    return x_knots, y_knots, z_knots, tangentVectors, normalVectors, binormalVectors, orientationPhi, orientationTheta, curvature, radiusoFCurvature


def getBranches(sorteddictOfNodesAndRadius, curvature):
    """
       a trial function written to find branches based on if the
       curvature at next node is different from the previous node
    """
    dictOfNodesAndBranches = {}
    i = 0
    keys = list(sorteddictOfNodesAndRadius.keys())
    keys = np.array(np.unravel_index(keys, (424, 512, 512), order='C')).tolist()
    for coordinate, radius, curvature in zip(keys, list(sorteddictOfNodesAndRadius.values()), curvature):
        branch = 0
        if radius[i] > radius[i + 1]:
            dictOfNodesAndBranches[tuple(coordinate)] = branch + 1
        else:
            if curvature[i] < curvature[i + 1]:
                pass
        i += 1
    return dictOfNodesAndBranches


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


def plot3Dfigure(inrerpolatedImage):
    """
       plots a 3D volume
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from skimage import measure
    verts, faces = measure.marching_cubes(inrerpolatedImage, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, inrerpolatedImage.shape[0])
    ax.set_ylim(0, inrerpolatedImage.shape[1])
    ax.set_zlim(0, inrerpolatedImage.shape[2])
    ax.set_aspect('equal')


if __name__ == '__main__':
    from KESMAnalysis.skeleton.radiusOfNodes import _getBouondariesOfimage
    inputIm = np.load('/home/pranathi/Downloads/mouseBrainGreyscale.npy')
    # saveVolumeFeatures(inputIm, ' Grey scale image')

    thresholdIm = np.load('/home/pranathi/Downloads/mouseBrainBinary.npy')
    # saveVolumeFeatures(thresholdIm, ' Binary image')

    boundaryIm = _getBouondariesOfimage(thresholdIm)
    # saveVolumeMetadata(inputIm)
    skeletonIm = np.load('/home/pranathi/Downloads/shortestPathSkel.npy')
    # saveVolumeFeatures(skeletonIm, ' Thinned image')

    shskel = np.load("/home/pranathi/Downloads/shortestPathSkel.npy")
    # interpolatedImage = np.load('/Users/3scan_editing/records/interpolatedSkeleton.npy')
    dictOfNodesAndRadius, distTransformedIm = getRadiusByPointsOnCenterline(shskel, boundaryIm, inputIm)
    # getRadisuStatistics(dictOfNodesAndRadius, distTransformedIm)
    # # saveVolumeFeatures(shskel, ' Unit Width Voxel image')
    # aspectRatio = [1, 1, 10]
    # x_knots, y_knots, z_knots, tangentVectors, normalVectors, binormalVectors, orientationPhi, orientationTheta, curvature, radiusoFCurvature = splineInterpolateStatistics(shskel, aspectRatio)
