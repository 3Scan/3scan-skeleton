import numpy as np

import copy
import operator
import scipy
import time

from scipy import ndimage
from skimage import morphology


"""
   calculates radius as distance of node to nearest zero co-ordinate(edge)
   if radius is zero it is a single isolated voxel which may be
   due to noise, itself forms an edge
   these voxels can be removed through some sanity checks with
   their presence in the grey scale iage
"""


def _getBouondariesOfimage(image):
    """
       function to find boundaries/border/edges of the array/image
    """
    assert image.shape[0] != 1
    if image.shape[0] == 1:
        print("Single slice, not a 3d image")
    sElement = ndimage.generate_binary_structure(3, 1)
    erode_im = ndimage.morphology.binary_erosion(image, sElement)
    boundaryIm = image - erode_im
    assert np.sum(boundaryIm) <= np.sum(image)
    return boundaryIm


def getRadiusByPointsOnCenterline(skeletonIm, boundaryIm, inputIm, aspectRatio=[1, 1, 1]):
    """
       removes voxels with radius 0.0
    """
    skeletonImCopy = copy.deepcopy(skeletonIm)
    inputImCopy = copy.deepcopy(inputIm)
    startt = time.time()
    skeletonImCopy[skeletonIm == 0] = 255
    skeletonImCopy[boundaryIm == 1] = 0
    distTransformedIm = ndimage.distance_transform_edt(skeletonImCopy, aspectRatio)
    listNZI = list(set(map(tuple, np.transpose(np.nonzero(skeletonIm)))))
    dictOfNodesAndRadius = list_to_dict(listNZI, distTransformedIm)
    label, countBefore = scipy.ndimage.measurements.label(inputIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    inputImCopy[distTransformedIm == 0.0] = 0
    label, countAfter = scipy.ndimage.measurements.label(inputIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if countAfter == countBefore:
        print("voxels with radius 0.00 can be removed")
    else:
        print("voxels with radius 0.00 cannot be removed")

    print("time taken to find the nodes and their radius is ", time.time() - startt, "seconds")
    return dictOfNodesAndRadius, distTransformedIm


def getRadiusByPointsOnCenterlineslicewise(skeletonIm, boundaryIm, inputIm, aspectRatio=[1, 1], plane=0):
    """
       (z, y, x) removes voxels with radius 0.0 and get radius by looking in the plane = 0 look in x, y
       plane = 1 look in z, x and plane = 2 look in z, y
    """
    skeletonImCopy = copy.deepcopy(skeletonIm)
    inputImCopy = copy.deepcopy(inputIm)
    startt = time.time()
    skeletonImCopy[skeletonIm == 0] = 255
    skeletonImCopy[boundaryIm == 1] = 0
    distTransformedIm = np.zeros((skeletonImCopy.shape))
    for i in range(0, skeletonImCopy.shape[0]):
        distTransformedIm[i] = ndimage.distance_transform_edt(skeletonImCopy[i], sampling=aspectRatio)
    listNZI = list(set(map(tuple, np.transpose(np.nonzero(skeletonIm)))))
    dictOfNodesAndRadius = list_to_dict(listNZI, distTransformedIm)
    label, countBefore = scipy.ndimage.measurements.label(inputIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    inputImCopy[distTransformedIm == 0.0] = 0
    label, countAfter = scipy.ndimage.measurements.label(inputIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if countAfter == countBefore:
        print("voxels with radius 0.00 can be removed")
    else:
        print("voxels with radius 0.00 cannot be removed")

    print("time taken to find the nodes and their radius is ", time.time() - startt, "seconds")
    return dictOfNodesAndRadius, distTransformedIm


def list_to_dict(listNZI, skeletonLabelled):
    dictOfIndicesAndlabels = {item: skeletonLabelled[item] for item in listNZI}
    return dictOfIndicesAndlabels


def colorCodeByRadius(dictOfNodesAndRadius, distTransformedIm):
    z, y, x = np.shape(distTransformedIm)
    shapeC = z, y, x, 3
    colorCodedImage = np.zeros(shapeC, dtype=np.uint8)
    sorted_x = sorted(dictOfNodesAndRadius.items(), key=operator.itemgetter(1), reverse=True)
    for index, (key, value) in enumerate(sorted_x):
        x, y, z = key
        colorCodedImage[z, y, x, 0] = (index + 1) * 1
        colorCodedImage[z, y, x, 1] = (index + 1) * 2
        colorCodedImage[z, y, x, 2] = (index + 1) * 3
    return colorCodedImage


def averageShortestdistance(d1, d2, d3):
    assert len(d1) == len(d2) == len(d3)
    radiusz = [d1[k] for k in d1]
    radiusy = [d2[k] for k in d2]
    radiusx = [d3[k] for k in d3]
    averageShortestdistance = []
    for i in range(0, len(d1)):
        averageShortestdistance[i] = (radiusz[i] + radiusy[i] + radiusx[i]) / 3
    return averageShortestdistance


def getReconstructedVasculature(dictOfNodesAndRadius, distTransformedIm, skeletonIm):
    zDim, yDim, xDim = np.shape(distTransformedIm)
    shapeC = zDim, yDim, xDim
    reconstructedImage = np.zeros(shapeC, dtype=np.uint8)
    se = np.ones((3, 3, 3), dtype=np.uint8)
    label, numDisjointRegions = ndimage.measurements.label(skeletonIm, structure=se)
    mask = np.zeros(shapeC, dtype=np.uint8)
    print(numDisjointRegions)
    objectify = ndimage.find_objects(label)
    for i in range(0, numDisjointRegions):
        loc = objectify[i]
        zcoords = loc[0]; ycoords = loc[1]; xcoords = loc[2]
        regionLowerBoundZ = zcoords.start - 1; regionLowerBoundY = ycoords.start - 1; regionLowerBoundX = xcoords.start - 1
        regionUpperBoundZ = zcoords.stop + 1; regionUpperBoundY = ycoords.stop + 1; regionUpperBoundX = xcoords.stop + 1
        dilatedMask = mask[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
        dilatedSkeletonIm = skeletonIm[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
        dests = map(tuple, np.transpose(np.nonzero(dilatedSkeletonIm)))
        for dest in dests:
            radius = dictOfNodesAndRadius[dest]
            selemBall = morphology.ball(radius)
            dilatedMask[dest] = 1
            reconstructIthImage = ndimage.morphology.binary_dilation(dilatedSkeletonIm, structure=selemBall, iterations=1, mask=dilatedMask)
            reconstructedImage[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX] = np.logical_or(reconstructedImage[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX], reconstructIthImage)
    # skeletonImNew[np.logical_and(valencearray == 0, skeletonIm == 1)] = 1  # see if isolated voxels can be removed (answer: yes)
    stopp = time.time()
    print("time taken to reconstruct the skeleton is", (stopp - startt), "seconds")
    return reconstructedImage


if __name__ == '__main__':
    from skeleton.orientationStatisticsSpline import getStatistics, plotKDEAndHistogram
    startt = time.time()
    # load the skeletonized image
    skeletonIm = np.load('/home/pranathi/Downloads/shortestPathSkel.npy')
    thresholdIm = np.load('/home/pranathi/Downloads/mouseBrainBinary.npy')
    inputIm = np.load('/home/pranathi/Downloads/mouseBrainGreyscale.npy')
    # finding edges of the microvasculature
    boundaryIm = _getBouondariesOfimage(thresholdIm)
    dictOfNodesAndRadius, distTransformedIm = getRadiusByPointsOnCenterline(skeletonIm, boundaryIm, inputIm, aspectRatio=[1, 1, 1])
    dictOfNodesAndRadiusz, distTransformedImz = getRadiusByPointsOnCenterlineslicewise(skeletonIm, boundaryIm, inputIm, aspectRatio=[0.7, 0.7], plane=0)
    dictOfNodesAndRadiusy, distTransformedImz = getRadiusByPointsOnCenterlineslicewise(skeletonIm, boundaryIm, inputIm, aspectRatio=[0.7, 0.7], plane=1)
    dictOfNodesAndRadiusx, distTransformedImz = getRadiusByPointsOnCenterlineslicewise(skeletonIm, boundaryIm, inputIm, aspectRatio=[0.7, 0.7], plane=2)
    averageShortestdistance = averageShortestdistance(dictOfNodesAndRadiusy, dictOfNodesAndRadiusx, dictOfNodesAndRadiusz)
    getStatistics(dictOfNodesAndRadius, 'Radius')
    plotKDEAndHistogram(list(dictOfNodesAndRadius.values()))
