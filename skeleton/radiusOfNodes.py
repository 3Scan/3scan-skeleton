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
    if len(image.shape) == 3:
        sElement = ndimage.generate_binary_structure(3, 1)
    elif len(image.shape) == 2:
        sElement = ndimage.generate_binary_structure(2, 1)
    erode_im = ndimage.morphology.binary_erosion(image, sElement)
    boundaryIm = image - erode_im
    assert np.sum(boundaryIm) <= np.sum(image)
    return boundaryIm


def getRadiusByPointsOnCenterline(skeletonIm, boundaryIm, aspectRatio=[1, 1, 1]):
    """
       find radius
    """
    skeletonImCopy = copy.deepcopy(skeletonIm)
    startt = time.time()
    skeletonImCopy[skeletonIm == 0] = 255
    skeletonImCopy[boundaryIm == 1] = 0
    distTransformedIm = ndimage.distance_transform_edt(skeletonImCopy, aspectRatio)
    listNZI = list(set(map(tuple, np.transpose(np.nonzero(skeletonIm)))))
    dictOfNodesAndRadius = list_to_dict(listNZI, distTransformedIm)
    print("time taken to reconstruct the skeleton is %0.3f seconds" % (time.time() - startt))
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
    print("time taken to reconstruct the skeleton is %0.3f seconds" % (time.time() - startt))
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


def getReconstructedVasculature(distTransformedIm, skeletonIm):
    startt = time.time()
    zDim, yDim, xDim = np.shape(distTransformedIm)
    shapeC = zDim, yDim, xDim
    reconstructedImage = np.zeros(shapeC, dtype=np.uint8)
    mask = np.zeros(shapeC, dtype=np.uint8)
    dests = map(tuple, np.transpose(np.nonzero(skeletonIm)))
    for dest in dests:
        radius = distTransformedIm[dest]
        selemBall = morphology.ball(radius)
        mask[dest] = 1
        reconstructIthImage = ndimage.morphology.binary_dilation(skeletonIm, structure=selemBall, iterations=-1, mask=mask)
        reconstructedImage = np.logical_or(reconstructedImage, reconstructIthImage)
    print("time taken to reconstruct the skeleton is %0.3f seconds" % (time.time() - startt))
    return reconstructedImage


if __name__ == '__main__':
    # from skeleton.orientationStatisticsSpline import getStatistics, plotKDEAndHistogram
    startt = time.time()
    # load the skeletonized image
    skeletonIm = np.load('/home/pranathi/Downloads/mouseBrainSkeleton.npy')
    # finding edges of the microvasculature
    boundaryIm = np.load('/media/pranathi/DATA/twodimageslices/twodkskeletonslices/boundaries.npy')
    dictOfNodesAndRadius, distTransformedIm = getRadiusByPointsOnCenterline(skeletonIm, boundaryIm, aspectRatio=[5, 0.7, 0.7])
    reconstructedImage = getReconstructedVasculature(distTransformedIm, skeletonIm)
    # dictOfNodesAndRadiusz, distTransformedImz = getRadiusByPointsOnCenterlineslicewise(skeletonIm, boundaryIm, inputIm, aspectRatio=[0.7, 0.7], plane=0)
    # dictOfNodesAndRadiusy, distTransformedImz = getRadiusByPointsOnCenterlineslicewise(skeletonIm, boundaryIm, inputIm, aspectRatio=[0.7, 0.7], plane=1)
    # dictOfNodesAndRadiusx, distTransformedImz = getRadiusByPointsOnCenterlineslicewise(skeletonIm, boundaryIm, inputIm, aspectRatio=[0.7, 0.7], plane=2)
    # averageShortestdistance = averageShortestdistance(dictOfNodesAndRadiusy, dictOfNodesAndRadiusx, dictOfNodesAndRadiusz)
    # getStatistics(dictOfNodesAndRadius, 'Radius')
    # plotKDEAndHistogram(list(dictOfNodesAndRadius.values()))
