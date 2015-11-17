import copy
import itertools
import numpy as np
import time

from scipy import ndimage
from scipy.ndimage.filters import convolve
from skimage.graph import route_through_array

"""
   The goal of this algorithm is to generate a topologically and geometrically preserved
   unit pixel width skeleton. The skeleton/ centerline
   of an image obtained by using various structuring elements does not
   necessarily be a 1 pixel wide although it ensures topological connectedness.
   end point = 1 middle point = 2 joint point = 3 crowded point = 4 crowded region = 5
   implemented according to the paper in this link
   https://drive.google.com/a/3scan.com/file/d/0BwQueSW2_nsOWGRzb2s4dlZmRlE/view?usp=sharing

"""


def __getPaddedimage(image):
    """

       pads array in all dimensions with 0s

    """
    z, m, n = np.shape(image)
    paddedShape = z + 2, m + 2, n + 2
    padImage = np.zeros((paddedShape), dtype=np.uint8)
    padImage[1:z + 1, 1:m + 1, 1:n + 1] = image
    return padImage


def _setValenceOfarray(arr):
    assert arr.ndim == 3

    template = np.ones((3, 3, 3), dtype=np.uint8)
    template[1, 1, 1] = 0

    result = convolve(arr, template, mode='constant', cval=0)
    result[arr == 0] = 0
    dictOfIndicesAndvalencies = {tuple(item): result[tuple(item)] for item in list(np.transpose(np.nonzero(result)))}
    return result, dictOfIndicesAndvalencies


def __intersect(a, b):
    """

       return the intersection of two lists

    """
    return set(a) == set(b)


def __intersectAssert(a, b):
    """

       return the intersection of two lists

    """
    if len(set(a) - set(b)) == 0:
        return 1
    else:
        return 0


def _setLabelEndMiddlepoints(connNeighbors, nzi):
    """

       set an object point with label 2 if non zero
       neighbors in the 26 neighbors
       as middle point,
       if the degree is 2 and the points
       are not 26 connected, otherwise it is an end
       point
       To check the 26 connectivity, square of distance
       between the non zero coordinates is considered


    """
    cNeighbors = int(np.sum(connNeighbors))
    assert cNeighbors in [1, 2]

    if cNeighbors == 2:
        if np.sum((nzi[0] - nzi[1]) ** 2) > 3:
            return 2
        else:
            return 4
    else:
        return 4


def _setLabelJointCrowdedpoints(connNeighbors):
    """

       set an object point as joint point if the
       point has degree greater than 2 and the
       26 connected points are either middle points
       or end points i.e have labels 1 or 2 in the
       valence array else it is a crowded joint point

    """
    if __intersect([1, 2], connNeighbors) or __intersect([1], connNeighbors) or __intersect([2], connNeighbors):
        return 3
    else:
        return 4


def __intersectCrowded(a, b):
    """

       return the intersection of two lists

    """
    if list(set(a) & set(b)) == list(set(a)):
        return 1
    else:
        return 0


def _setLabelCrowdedregion(connNeighborsList):
    # d = np.diff(points, axis=0)
    # segdists = ((d ** 2).sum(axis=0))
    # condition = segdists <= 3
    if __intersectCrowded([4], connNeighborsList) == 1:
        return 5
    else:
        return 4


def outOfPixBOunds(nearByCoordinate, aShape):
    zMax, yMax, xMax = aShape
    isAtXBoundary = nearByCoordinate[0] in [0, xMax]
    isAtYBoundary = nearByCoordinate[1] in [0, yMax]
    isAtZBoundary = nearByCoordinate[2] in [0, zMax]
    if isAtXBoundary or isAtYBoundary or isAtZBoundary:
        return 1
    else:
        return 0


stepDirect = itertools.product((-1, 0, 1), repeat=3)
listStepDirect = list(stepDirect)
listStepDirect.remove((0, 0, 0))


def _getAllLabelledarray(skeletonIm, valencearray):
    """

       label end, middle, joint, crowded points and a
       crowded region as 1, 2, 3, 4 and 5 respectively

    """
    skeletonImLabel = np.zeros_like(skeletonIm)
    skeletonImLabel[valencearray == 1] = 1
    startt = time.time()
    aShape = np.shape(valencearray)

    # The point we start from must be either a point with
    # degree 1 or 2 to label end and middle points
    iterationMaskEndAndMiddle = np.zeros_like(skeletonIm)
    iterationMaskEndAndMiddle[valencearray == 2] = 1
    listIterateEndAndMiddle = list(np.transpose(np.nonzero(iterationMaskEndAndMiddle)))
    for k in listIterateEndAndMiddle:
        connNeighborsList = []
        connNeighborsIndices = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + np.array(d))
            if outOfPixBOunds(nearByCoordinate, aShape):
                continue
            if skeletonIm[nearByCoordinate] == 0:
                continue
            connNeighbors = skeletonIm[nearByCoordinate]
            connNeighborsList.append(connNeighbors)
            connNeighborsIndices.append(np.array(nearByCoordinate))
        # call labels function
        if connNeighborsList == []:
            continue
        skeletonImLabel[tuple(k)] = _setLabelEndMiddlepoints(connNeighborsList, connNeighborsIndices)
    stoppM = time.time()
    print("time taken to label end and middle points is", (stoppM - startt), "seconds")
    skeletonImLabel[valencearray > 2] = 3
    skeletonImLabel2 = np.copy(skeletonImLabel)
    iterationMaskJointAndCrowded = np.zeros_like(skeletonIm)
    iterationMaskJointAndCrowded[valencearray > 2] = 1
    listJointAndcrowded = list(np.transpose(np.nonzero(iterationMaskJointAndCrowded)))
    for k in listJointAndcrowded:
        # The point we start from must be with valency greater than 2
        # have to be decided if they are joint points
        # or crowded joint points
        connNeighborsList = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + np.array(d))
            if outOfPixBOunds(nearByCoordinate, aShape):
                continue
            if skeletonImLabel[nearByCoordinate] != 0:
                connNeighbors = skeletonImLabel[nearByCoordinate]
                connNeighborsList.append(connNeighbors)
        # call labels function
        skeletonImLabel2[tuple(k)] = _setLabelJointCrowdedpoints(connNeighborsList)
    stoppJ = time.time()
    print("time taken to label joint and crowded points is", (stoppJ - stoppM), "seconds")
    skeletonImLabel3 = np.copy(skeletonImLabel2)
    iterationMaskCrowdedRegion = np.zeros_like(skeletonImLabel3)
    iterationMaskCrowdedRegion[skeletonImLabel2 == 4] = 1
    listCrowdedRegion = list(np.transpose(np.nonzero(iterationMaskCrowdedRegion)))
    for index, k in enumerate(listCrowdedRegion):
        # The point we start from must be "crowded point(label = 4)"
        # to label crowded regions
        connNeighborsList = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + np.array(d))
            if outOfPixBOunds(nearByCoordinate, aShape):
                continue
            if skeletonImLabel2[nearByCoordinate] != 0:
                connNeighbors = skeletonImLabel2[nearByCoordinate]
                connNeighborsList.append(connNeighbors)
        # call labels function
        skeletonImLabel3[tuple(k)] = _setLabelCrowdedregion(connNeighborsList)
    stoppR = time.time()
    print("time taken to label crowded regions is", (stoppR - stoppJ), "seconds")
    listOfLabelledArrays = [skeletonImLabel, skeletonImLabel2, skeletonImLabel3]
    assert __intersectAssert(list(set(map(tuple, list(np.transpose(np.nonzero(skeletonImLabel)))))), list(set(map(tuple, list(np.transpose(np.nonzero(skeletonIm)))))))
    assert __intersectAssert(list(set(map(tuple, list(np.transpose(np.nonzero(skeletonImLabel2)))))), list(set(map(tuple, list(np.transpose(np.nonzero(skeletonIm)))))))
    assert __intersectAssert(list(set(map(tuple, list(np.transpose(np.nonzero(skeletonImLabel3)))))), list(set(map(tuple, list(np.transpose(np.nonzero(skeletonIm)))))))
    return np.uint8(skeletonImLabel3), listOfLabelledArrays


def _getExitsOfShortestpaths(listExitIndices, listSourceIndices):
    """

       exit is a end or middle point

    """
    # listOfExits = []
    for items in listExitIndices:
        a = np.array(items)
        for item in listSourceIndices:
            b = np.array(item)
            dist = np.sum((a - b) ** 2)
            if dist <= 3:
                # listOfExits.append(tuple(a))
                yield tuple(a)


def _getSourcesOfShortestpaths(listNZI, dictOfIndicesAndlabels):
    # print("in search of sources")
    # print(dictOfIndicesAndlabels)
    listIndex = [(coord, dictOfIndicesAndlabels[coord]) for coord in listNZI]
    # print(listIndex)
    summationList = []
    for i, (value, valence) in enumerate(listIndex):
        # print("in search of listIndex {}/{}".format(i, len(listIndex)))
        distList = []
        for item in listNZI:
            dist = np.sum((np.array(value) - np.array(item)) ** 2)
            distList.append(dist)
        # print("in search of summationList")
        summation = sum(distList) / valence
        summationList.append(summation)
    src = listNZI[np.argmin(summationList)]
    # print(src)
    # print("src is returned")
    return src


def _findShortestPathFromCRcenterToexit(valencearray, source, dest):
    """
       dijkstra's shortest path, route through the array across the
       minimum cost path
    """
    indices, weight = route_through_array(valencearray, source, dest)
    indices = np.array(indices).T
    path = np.zeros_like(valencearray)
    path[indices[0], indices[1], indices[2]] = 1
    return path


def getShortestPathskeleton(skeletonIm):
    # structuring elements to find the number of crowded regions
    # dimensions of the original skeleton before padding
    z, m, n = np.shape(skeletonIm)
    startt = time.time()
    # pad the image
    skeletonIm = __getPaddedimage(skeletonIm)
    skeletonImNew = np.zeros_like(skeletonIm)
    listOfLabelledArrays = []
    # obtain the valence array and dictionary with non zero coordinates and their degrees
    valencearray, dict1 = _setValenceOfarray(skeletonIm)
    print("time taken to calculate the valence array is", (time.time() - startt), "seconds")
    if [1] not in list(dict1.values()) and [2] not in list(dict1.values()):
        print("all elements belong to crowded region! weird")
        return skeletonIm[1:z + 1, 1:m + 1, 1:n + 1]
    else:
        # label the skeleton with the crowded regions, joint points, crowded points, end points and middle
        # points as 5, 3, 4 1, 2
        # dimensions of the original skeleton before padding
        skeletonLabelled, listOfLabelledArrays = _getAllLabelledarray(skeletonIm, valencearray)
        # the background points i.e 0s are changed to 255
        # because they are points a skeleton never has to
        # pass through
        skeletonLabelled1 = copy.deepcopy(skeletonLabelled)
        skeletonLabelled1[skeletonLabelled1 == 0] = 255
        if np.max(skeletonLabelled) < 4:
            print("there are no crowded joint points in the image")
            # if they are no crowded joint points in the image, then
            # there are no points that should be removed, as there are
            # no crowded regions
            return skeletonIm[1:z + 1, 1:m + 1, 1:n + 1]
        else:
            print("crowded joint points exist")
        # find and copy the coordinates with labels 1 and 2
        # i.e end and middle points
        # to find the exits to a crowded region
        # crowded region are the pixels with label 5 in labelled
        # skeleton image
        exits = np.uint8(np.logical_or(skeletonLabelled == 1, skeletonLabelled == 2))
        crowdedRegion = np.zeros_like(skeletonLabelled)
        crowdedRegion[skeletonLabelled == 5] = 1
        se = np.ones((3, 3, 3), dtype=np.uint8)
        label, noOfCrowdedregions = ndimage.measurements.label(crowdedRegion, structure=se)
        # print("found # of crowded regions: ", noOfCrowdedregions)
        objectify = ndimage.find_objects(label)
        # noOfCrowdedregions2 = len(objectify)
        # print("found # of objects: ", noOfCrowdedregions2)
        for i in range(0, noOfCrowdedregions):
            print("in ith loop", i)
            loc = objectify[i]
            zcoords = loc[0]; ycoords = loc[1]; xcoords = loc[2]
            regionLowerBoundZ = zcoords.start - 1; regionLowerBoundY = ycoords.start - 1; regionLowerBoundX = xcoords.start - 1
            regionUpperBoundZ = zcoords.stop + 1; regionUpperBoundY = ycoords.stop + 1; regionUpperBoundX = xcoords.stop + 1
            # print("region bounds found: ({}, {}, {}) to ({}, {}, {})".format(regionLowerBoundX, regionLowerBoundY, regionLowerBoundZ, regionUpperBoundX, regionUpperBoundY, regionUpperBoundZ))
            starttSrc = time.time()
            dilatedValenceObjectLoc = valencearray[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
            # print("dilatedValenceObjectLoc found")
            dilatedLabelledObjectLoc = skeletonLabelled1[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
            # print("dilatedLabelledObjectLoc found")
            # dilatedRegionExits = exits[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
            listNZI = list(set(map(tuple, np.transpose(np.nonzero(dilatedValenceObjectLoc)))))
            dictOfIndicesAndlabels = list_to_dict(listNZI, dilatedValenceObjectLoc)
            src = _getSourcesOfShortestpaths(listNZI, dictOfIndicesAndlabels)
            # print("src found: ", src)
            exits = np.uint8(np.logical_or(dilatedLabelledObjectLoc == 1, dilatedLabelledObjectLoc == 2))
            listExitIndices = list(set(map(tuple, np.transpose(np.nonzero(exits)))))
            dests = _getExitsOfShortestpaths(listExitIndices, listNZI)
            print("src and dest finding took ", (time.time() - starttSrc), "seconds")
            for dest in dests:
                # print("dests found: ", dest)
                dilatedLabelledObjectLoc1 = _findShortestPathFromCRcenterToexit(dilatedLabelledObjectLoc, src, dest)
                skeletonImNew[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX] = np.logical_or(skeletonImNew[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX], dilatedLabelledObjectLoc1)
        skeletonImNew[skeletonLabelled < 5] = 1
        skeletonImNew[skeletonLabelled == 0] = 0
        stopp = time.time()
        print("time taken to find the shortest path skeleton is", (stopp - startt), "seconds")
        return np.uint8(skeletonImNew[1:z + 1, 1:m + 1, 1:n + 1])


def list_to_dict(listNZI, skeletonLabelled):
    dictOfIndicesAndlabels = {item: skeletonLabelled[item] for item in listNZI}
    return dictOfIndicesAndlabels


# def main():
skeletonIm = np.load('/Users/3scan_editing/records/scratch/skeleton3dtestpotimizeWoBoundary3.npy')
# image = np.ones((3, 3, 3), dtype=np.uint8)
# sample = np.zeros((5, 5, 5), dtype=np.uint8)
# sample2 = np.zeros((7, 5, 5), dtype=np.uint8)
# sample[1:4, 1:4, 1:4] = image
# sample[4][2][2] = 1
# sample[0][2][2] = 1
# sample2[0][2][2] = 1
# sample2[6, ...] = sample[0, ...]
# sample2[1:6, ...] = sample
shortestPathSkel = getShortestPathskeleton(skeletonIm)
# print(shortestPathSkel)
# test if number of objects in the skeletonized image without crowded regions and
# with crowded regions are equal
label_img1, countObjects = ndimage.measurements.label(skeletonIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
label_img2, countObjectsShorty = ndimage.measurements.label(shortestPathSkel, structure=np.ones((3, 3, 3), dtype=np.uint8))
assert countObjects == countObjectsShorty
np.save("/Users/3scan_editing/records/scratch/shortestPathSkeldoa.npy", shortestPathSkel)


# if __name__ == '__main__':
#     main()
