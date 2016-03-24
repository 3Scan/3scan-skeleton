
import json
import time

from scipy import interpolate
from scipy import ndimage

import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from collections import Counter
from math import sqrt, pow, atan2, degrees
from statistics import median, mode, pvariance, mean


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


def getStatistics(dictOfNodesAndRadius, featureName):
    """
       function to obtain common statistics like mean, median. featureName is the string variable
       that takes in the name of the feature being evaluated
    """
    listRadius = [float(dictOfNodesAndRadius[k]) for k in dictOfNodesAndRadius]
    meanRadius = mean(listRadius)
    varianceRadius = (pvariance(listRadius, meanRadius))
    radiusCounts = Counter(listRadius)
    dictStats = {'Zerocount' + featureName: listRadius.count(0.0), 'max' + featureName: max(listRadius), 'min' + featureName: min(listRadius),
                 'mean' + featureName: mean(listRadius),
                 'median' + featureName: median(listRadius), 'mode' + featureName: mode(listRadius),
                 'variance' + featureName: varianceRadius, 'stddev' + featureName: sqrt(varianceRadius),
                 'uniqueCounts' + featureName: len(radiusCounts)}
    with open("statistics.json", "a") as feedsjson:
        feedsjson.write("{}\n".format(json.dumps(dictStats)))
    feedsjson.close()


def plotKDEAndHistogram(ndimarray):
    """
       function to obtain a KDE and histogram of the distribution.
       input ndimarray must be array. if it isn't then it is converted
       to one and histogram is plotted
    """
    if type(ndimarray) == list:
        ndimarray = np.array(ndimarray)
    bins = np.unique(np.round(ndimarray, 0))
    sns.distplot(ndimarray, kde=False, bins=bins)
    plt.show()


def plotKde(dictionary):
    arr = np.array(list(dictionary.values()))
    X = np.reshape(arr, (arr.size, 1))
    X_plot = np.linspace(X.min(), X.max(), 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
    log_dens = kde.score_samples(X_plot)
    plt.plot(X_plot[:, 0], np.exp(log_dens))
    plt.show()


def splineInterpolateStatistics(shskel, aspectRatio=[1, 1, 1]):
    """
       spline curve fitting and orientation finding from
       the derivatives
    """
    interpolatedSkeleton = ndimage.interpolation.zoom(shskel, zoom=aspectRatio, order=0)
    z_sample, y_sample, x_sample = np.array(np.where(interpolatedSkeleton != 0))
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
    shskel = np.load(input("please enter a path to your unit width voxelized skeleton"))
    # interpolatedImage = np.load('/Users/3scan_editing/records/interpolatedSkeleton.npy')
    # saveVolumeFeatures(shskel, ' Unit Width Voxel image')
    aspectRatio = input("please enter a sequence resolution of a voxel in 3D with resolution in z followed by y and x")
    x_knots, y_knots, z_knots, tangentVectors, normalVectors, binormalVectors, orientationPhi, orientationTheta, curvature, radiusoFCurvature = splineInterpolateStatistics(shskel, aspectRatio)
    plotKDEAndHistogram(orientationPhi)
    plotKDEAndHistogram(orientationTheta)
