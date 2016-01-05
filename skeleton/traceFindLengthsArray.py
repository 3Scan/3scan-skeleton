import itertools
import numpy as np
import time

from scipy import ndimage
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


def _setOutValenceOfarray(arr):
    from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
    from skeleton.cliqueRemovig import removeCliqueEdges
    import networkx as nx
    sampleGraph = getNetworkxGraphFromarray(arr, True)
    removeCliqueEdges(sampleGraph)
    dictOfIndicesAndvalencies = nx.degree(sampleGraph)
    result = dict_to_array(dictOfIndicesAndvalencies, arr.shape)
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
    print(cNeighbors)
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
    print(len(listNZI))
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


def _getSourcesOfShortestpathsArr(listNZI, dictOfIndicesAndvalencies):
    print(len(listNZI))
    listIndex = [(tuple(coord), dictOfIndicesAndvalencies[tuple(coord)]) for coord in listNZI]
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
    return path, indices


def _getDistanceBetweenPointsInpath(cyclePath, cycle=0):
    distList = []
    print(" in _getDistanceBetweenPointsInpath function")
    if cycle:
        for index, item in enumerate(cyclePath):
            if index + 1 < len(cyclePath):
                item2 = cyclePath[index + 1]
            elif index + 1 == len(cyclePath):
                item2 = cyclePath[0]
            dist = np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2))
            distList.append(dist)
    else:
        for index, item in enumerate(cyclePath):
            if index + 1 != len(cyclePath):
                item2 = cyclePath[index + 1]
                dist = np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2))
                distList.append(dist)
    return sum(distList)


def getTrace(skeletonIm):
    se = np.ones((3, 3, 3), dtype=np.uint8)
    z, m, n = np.shape(skeletonIm)
    startt = time.time()
    skeletonIm = np.lib.pad(skeletonIm, 1, 'constant', constant_values=0)
    labelInput, noOfObjects = ndimage.measurements.label(skeletonIm, structure=se)
    skeletonImNew = np.zeros_like(skeletonIm)
    skeletonImP = np.zeros_like(skeletonIm)
    valencearray, dictOfIndicesAndvalencies = _setOutValenceOfarray(skeletonIm)
    skeletonLabelled, listOfLabelledArrays = _getAllLabelledarray(skeletonIm, valencearray)
    print(skeletonLabelled)
    print("va", valencearray)
    objectify = ndimage.find_objects(labelInput)
    exits = np.uint8(np.logical_or(skeletonLabelled == 1, skeletonLabelled == 2))
    segmentCountdict = {}
    segmentLengthdict = {}
    segmentTortuositydict = {}
    print(noOfObjects)
    for i in range(0, noOfObjects):
        print(labelInput)
        loc = objectify[i]
        print(loc)
        zcoords = loc[0]; ycoords = loc[1]; xcoords = loc[2]
        regionLowerBoundZ = zcoords.start - 1; regionLowerBoundY = ycoords.start - 1; regionLowerBoundX = xcoords.start - 1
        regionUpperBoundZ = zcoords.stop + 1; regionUpperBoundY = ycoords.stop + 1; regionUpperBoundX = xcoords.stop + 1
        dilatedValenceObjectLoc = valencearray[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
        print(dilatedValenceObjectLoc)
        if _intersectCrowded(np.unique(dilatedValenceObjectLoc).tolist(), [0, 1, 2]) == 1:
            skeletonImNew[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundX: regionUpperBoundX, regionLowerBoundY: regionUpperBoundY] = skeletonIm[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundX: regionUpperBoundX, regionLowerBoundY: regionUpperBoundY]
            indices = np.transpose(np.nonzero(skeletonImNew)).tolist()
            print(indices)
            curveLength = _getDistanceBetweenPointsInpath(indices, 0)
            src = tuple(indices[0] - np.array((1, 1, 1))); dest = tuple(indices[-1] - np.array((1, 1, 1)))
            segmentTortuositydict[i, src, dest] = 1
            segmentCountdict[src] = 1
            segmentLengthdict[(src, dest)] = curveLength
        else:
            dilatedLabelledObjectLoc = skeletonLabelled[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
            dilatedLabelledObjectLoc[dilatedLabelledObjectLoc == 0] = 255
            dilatedRegionExits = exits[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
            print(dilatedValenceObjectLoc)
            skeletonImP[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX] = skeletonIm[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX]
            indices = np.transpose(np.nonzero(skeletonImP))
            srcArr = tuple(np.array(_getSourcesOfShortestpathsArr(indices, dictOfIndicesAndvalencies)) - np.array((1, 1, 1)))
            src = tuple(np.array(_getSourcesOfShortestpaths(dilatedValenceObjectLoc)) - np.array((1, 1, 1)))
            dests = _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc)
            segmentCountdict[srcArr] = len(dests)
            for i, dest in enumerate(dests):
                dilatedLabelledObjectLoc1, indices = _findShortestPathFromCRcenterToexit(dilatedLabelledObjectLoc, src, dest)
                curveDisplacement = np.sqrt(np.sum((np.array(src) - np.array(dest)) ** 2))
                curveLength = _getDistanceBetweenPointsInpath(indices, 0)
                dest = tuple(np.array(dest) - np.array((1, 1, 1)))
                segmentLengthdict[(srcArr, dest)] = curveLength
                segmentTortuositydict[i, srcArr, dest] = curveLength / curveDisplacement
                skeletonImNew[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX] = np.logical_or(skeletonImNew[regionLowerBoundZ: regionUpperBoundZ, regionLowerBoundY: regionUpperBoundY, regionLowerBoundX: regionUpperBoundX], dilatedLabelledObjectLoc1)
    skeletonImNew[skeletonLabelled < 5] = 1
    skeletonImNew[skeletonLabelled == 0] = 0
    stopp = time.time()
    print("time taken to find the shortest path skeleton segments and trace their lengths is", (stopp - startt), "seconds")
    return np.uint8(skeletonImNew[1: z + 1, 1: m + 1, 1: n + 1]), segmentCountdict, segmentLengthdict, segmentTortuositydict


def list_to_dict(listNZI, skeletonLabelled):
    dictOfIndicesAndlabels = {item: skeletonLabelled[item] for item in listNZI}
    return dictOfIndicesAndlabels


def dict_to_array(dictOfIndicesAndvalencies, aShape):
    npArray = np.zeros(aShape, dtype=np.uint8)
    for key, value in dictOfIndicesAndvalencies.items():
        npArray[key] = value
    return npArray


if __name__ == '__main__':
    # skeletonIm = np.load('/home/pranathi/Downloads/shortestPathSkel.npy')
    sampleLine = np.zeros((5, 5, 5), dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    shortestPathSkelNoc, segmentCountdict, segmentLengthdict, segmentTortuositydict = getTrace(sampleLine)
    # np.save("/home/pranathi/Downloads/shortestPathSkeldiff.npy", shortestPathSkelNoc)
