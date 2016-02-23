import itertools
import numpy as np
# import time

from scipy import ndimage
from scipy.ndimage.filters import convolve
from skimage.graph import route_through_array

"""
   The goal of this algorithm is to generate a topologically and geometrically preserved
   unit voxel width skeleton. The skeleton/ centerline
   of an image obtained by using various structuring elements does not
   necessarily be a 1 voxel wide although it ensures topological connectedness.
   end point = 1 middle point = 2 joint point = 3 crowded point = 4 crowded region = 5
   implemented according to the paper in this link
   https://drive.google.com/a/3scan.com/file/d/0BwQueSW2_nsOWGRzb2s4dlZmRlE/view?usp=sharing
"""


def _setValenceOfarray(arr):
    template = np.ones((3, 3, 3), dtype=np.uint8)
    template[1, 1, 1] = 0

    result = convolve(arr, template, mode='constant', cval=0)
    result[arr == 0] = 0
    dictOfIndicesAndvalencies = {tuple(item): result[tuple(item)] for item in list(np.transpose(np.nonzero(result)))}
    return result, dictOfIndicesAndvalencies


def _intersect(a, b):
    """
       return the intersection of two lists
    """
    return set(a) == set(b)


def _intersectAssert(a, b):
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
    if _intersect([1, 2], connNeighbors) or _intersect([1], connNeighbors) or _intersect([2], connNeighbors):
        return 3
    else:
        return 4


def _intersectCrowded(a, b):
    """
       return the intersection of two lists
    """
    if list(set(a) & set(b)) == list(set(a)):
        return 1
    else:
        return 0


def _setLabelCrowdedregion(connNeighborsList):
    if _intersectCrowded([4], connNeighborsList) == 1:
        return 5
    else:
        return 4


def outOfPixBOunds(nearByCoordinate, aShape):
    onbound = 0
    for index, maxVal in enumerate(aShape):
        isAtBoundary = nearByCoordinate[index] in [0, maxVal] or nearByCoordinate[index] > maxVal
        if isAtBoundary:
            onbound = 1
            break
        else:
            continue
    return onbound


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
    aShape = np.shape(valencearray)

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
        if connNeighborsList == []:
            continue
        skeletonImLabel[tuple(k)] = _setLabelEndMiddlepoints(connNeighborsList, connNeighborsIndices)
    skeletonImLabel[valencearray > 2] = 3
    skeletonImLabel2 = np.copy(skeletonImLabel)
    iterationMaskJointAndCrowded = np.zeros_like(skeletonIm)
    iterationMaskJointAndCrowded[valencearray > 2] = 1
    listJointAndcrowded = list(np.transpose(np.nonzero(iterationMaskJointAndCrowded)))
    for k in listJointAndcrowded:
        connNeighborsList = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + np.array(d))
            if outOfPixBOunds(nearByCoordinate, aShape):
                continue
            if skeletonImLabel[nearByCoordinate] != 0:
                connNeighbors = skeletonImLabel[nearByCoordinate]
                connNeighborsList.append(connNeighbors)
        skeletonImLabel2[tuple(k)] = _setLabelJointCrowdedpoints(connNeighborsList)
    skeletonImLabel3 = np.copy(skeletonImLabel2)
    iterationMaskCrowdedRegion = np.zeros_like(skeletonImLabel3)
    iterationMaskCrowdedRegion[skeletonImLabel2 == 4] = 1
    listCrowdedRegion = list(np.transpose(np.nonzero(iterationMaskCrowdedRegion)))
    for index, k in enumerate(listCrowdedRegion):
        connNeighborsList = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + np.array(d))
            if outOfPixBOunds(nearByCoordinate, aShape):
                continue
            if skeletonImLabel2[nearByCoordinate] != 0:
                connNeighbors = skeletonImLabel2[nearByCoordinate]
                connNeighborsList.append(connNeighbors)
        skeletonImLabel3[tuple(k)] = _setLabelCrowdedregion(connNeighborsList)
    listOfLabelledArrays = [skeletonImLabel, skeletonImLabel2, skeletonImLabel3]
    assert _intersectAssert(list(set(map(tuple, list(np.transpose(np.nonzero(skeletonImLabel)))))), list(set(map(tuple, list(np.transpose(np.nonzero(skeletonIm)))))))
    assert _intersectAssert(list(set(map(tuple, list(np.transpose(np.nonzero(skeletonImLabel2)))))), list(set(map(tuple, list(np.transpose(np.nonzero(skeletonIm)))))))
    assert _intersectAssert(list(set(map(tuple, list(np.transpose(np.nonzero(skeletonImLabel3)))))), list(set(map(tuple, list(np.transpose(np.nonzero(skeletonIm)))))))
    return np.uint8(skeletonImLabel3), listOfLabelledArrays


def _getSourcesOfShortestpaths(dilatedValenceObjectLoc):
    listNZI = list(set(map(tuple, np.transpose(np.nonzero(dilatedValenceObjectLoc)))))
    listIndex = [(coord, dilatedValenceObjectLoc[coord]) for coord in listNZI]
    summationList = []
    for value, valence in listIndex:
        distList = []
        for item in listNZI:
            dist = np.sum((np.array(value) - np.array(item)) ** 2)
            distList.append(dist)
        summation = sum(distList) / valence
        summationList.append(summation)
    src = listNZI[np.argmin(summationList)]
    return src


def _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc):
    """
       exit is a end or middle point
    """
    a = set(map(tuple, list(np.transpose(np.nonzero(dilatedRegionExits)))))
    dilatedRegionExits[dilatedRegionExits == 0] = 2
    dilatedRegionExits[dilatedLabelledObjectLoc == 5] = 0
    distTransformedIm = ndimage.distance_transform_edt(dilatedRegionExits)
    b = np.array(np.where(distTransformedIm <= 3)).T
    b = set(map(tuple, b))
    return list(a & b)


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
    se = np.ones((3, 3, 3), dtype=np.uint8)
    z, m, n = np.shape(skeletonIm)
    # startt = time.time()
    labelInput, noOfObjects = ndimage.measurements.label(skeletonIm, structure=se)
    skeletonIm = np.lib.pad(skeletonIm, 1, 'constant', constant_values=0)
    skeletonImNew = np.zeros_like(skeletonIm)
    listOfLabelledArrays = []
    valencearray, dict1 = _setValenceOfarray(skeletonIm)
    if np.sum(valencearray) == 0:
        return skeletonIm[1:z + 1, 1:m + 1, 1:n + 1]
    else:
        skeletonLabelled, listOfLabelledArrays = _getAllLabelledarray(skeletonIm, valencearray)
        crowdedRegion = np.zeros_like(skeletonLabelled)
        crowdedRegion[skeletonLabelled == 5] = 1
        label, noOfCrowdedregions = ndimage.measurements.label(crowdedRegion, structure=se)
        if noOfObjects == 1 and np.sum(skeletonIm) > 50 and noOfCrowdedregions == 1:
            src = _getSourcesOfShortestpaths(valencearray)
            skeletonImNew[src] = 1
            # print(" a large number of elements belong to one region")
            return np.uint8(skeletonImNew[1:z + 1, 1:m + 1, 1:n + 1])
        if np.max(skeletonLabelled) <= 4:
            # print("there are no crowded joint points in the image")
            return skeletonIm[1:z + 1, 1:m + 1, 1:n + 1]
        else:
            # print("crowded joint points exist")
            skeletonImNew = np.zeros_like(skeletonIm)
            objectify = ndimage.find_objects(label)
            exits = np.uint8(np.logical_or(skeletonLabelled == 1, skeletonLabelled == 2))
            # print(noOfCrowdedregions)
            for i in range(0, noOfCrowdedregions):
                loc = objectify[i]
                zcoords = loc[0]; ycoords = loc[1]; xcoords = loc[2]
                regionLowerBoundZ = zcoords.start - 1; regionLowerBoundY = ycoords.start - 1; regionLowerBoundX = xcoords.start - 1
                regionUpperBoundZ = zcoords.stop + 1; regionUpperBoundY = ycoords.stop + 1; regionUpperBoundX = xcoords.stop + 1
                dilatedValenceObjectLoc = valencearray[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
                dilatedLabelledObjectLoc = skeletonLabelled[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
                dilatedLabelledObjectLoc[dilatedLabelledObjectLoc == 0] = 255
                dilatedRegionExits = exits[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
                src = _getSourcesOfShortestpaths(dilatedValenceObjectLoc)
                dests = _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc)
                for dest in dests:
                    dilatedLabelledObjectLoc1 = _findShortestPathFromCRcenterToexit(dilatedLabelledObjectLoc, src, dest)
                    skeletonImNew[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX] = np.logical_or(skeletonImNew[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX], dilatedLabelledObjectLoc1)
            skeletonImNew[skeletonLabelled < 5] = 1
            skeletonImNew[skeletonLabelled == 0] = 0
            # skeletonImNew[np.logical_and(valencearray == 0, skeletonIm == 1)] = 1  # see if isolated voxels can be removed (answer: yes)
            # stopp = time.time()
            # print("time taken to find the shortest path skeleton is", (stopp - startt), "seconds")
            label_img1, countObjects = ndimage.measurements.label(skeletonIm, structure=np.ones((3, 3, 3), dtype=np.uint8))
            label_img2, countObjectsShorty = ndimage.measurements.label(skeletonImNew, structure=np.ones((3, 3, 3), dtype=np.uint8))
            assert countObjects >= countObjectsShorty
            return np.uint8(skeletonImNew[1:z + 1, 1:m + 1, 1:n + 1])


def list_to_dict(skeletonLabelled):
    listNZI = map(tuple, np.transpose(np.nonzero(skeletonLabelled)))
    dictOfIndicesAndlabels = {item: skeletonLabelled[item] for item in listNZI}
    return dictOfIndicesAndlabels


if __name__ == '__main__':
    skeletonIm = np.load('/home/pranathi/Downloads/twodimageslices/Skeleton.npy')
    shortestPathSkel = getShortestPathskeleton(skeletonIm)
    np.save("/home/pranathi/Downloads/twodimageslices/shortestPathSkel.npy", shortestPathSkel)
