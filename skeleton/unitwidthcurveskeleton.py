import itertools
import numpy as np
import time

from scipy import ndimage
from scipy.ndimage.filters import convolve
from skimage.graph import route_through_array

"""
   The goal of this algorithm is to generate a topologically and geometrically preserved
   unit voxel width skeleton for both 3D and 2D thinned arrays with crowded regions. The skeleton/ centerline
   of an image obtained by using various structuring elements does not
   necessarily be a 1 voxel wide although it ensures topological connectedness.
   end point = 1 middle point = 2 joint point = 3 crowded point = 4 crowded region = 5
   implemented according to the paper in this link
   https://drive.google.com/a/3scan.com/file/d/0BwQueSW2_nsOWGRzb2s4dlZmRlE/view?usp=sharing
"""


def _setValenceOfarray(arr):
    template = np.ones([3] * arr.ndim, dtype=np.uint8)
    if arr.ndim == 3:
        template[1, 1, 1] = 0
    else:
        template[1, 1] = 0
    arr = np.uint8(arr)
    result = convolve(arr, template, mode='constant', cval=0)
    result[arr == 0] = 0
    return result


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
    assert cNeighbors == 2

    if np.sum((nzi[0] - nzi[1]) ** 2) > 3:
        return 2
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


def outOfPixBounds(nearByCoordinate, aShape):
    onbound = 0
    for index, maxVal in enumerate(aShape):
        isAtBoundary = nearByCoordinate[index] >= maxVal or nearByCoordinate[index] < 0
        if isAtBoundary:
            onbound = 1
            break
        else:
            continue
    return onbound


def _getAllLabelledarray(skeletonIm, valencearray):
    """
       label end, middle, joint, crowded points and a
       crowded region as 1, 2, 3, 4 and 5 respectively
    """
    stepDirect = itertools.product((-1, 0, 1), repeat=skeletonIm.ndim)
    listStepDirect = list(stepDirect)
    if skeletonIm.ndim == 3:
        listStepDirect.remove((0, 0, 0))
    else:
        listStepDirect.remove((0, 0))
    listStepDirect = list(map(np.array, listStepDirect))
    skeletonImLabel = np.zeros(skeletonIm.shape, dtype=np.uint8)
    skeletonImLabel[valencearray == 1] = 1
    aShape = np.shape(valencearray)

    listIterateMiddle = list(np.transpose(np.array(np.where(valencearray == 2))))
    for k in listIterateMiddle:
        connNeighborsList = []
        connNeighborsIndices = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + d)
            if outOfPixBounds(nearByCoordinate, aShape):
                continue
            if skeletonIm[nearByCoordinate] == 0:
                continue
            connNeighbors = skeletonIm[nearByCoordinate]
            connNeighborsList.append(connNeighbors)
            connNeighborsIndices.append(np.array(nearByCoordinate))
        skeletonImLabel[tuple(k)] = _setLabelEndMiddlepoints(connNeighborsList, connNeighborsIndices)

    listJointAndcrowded = list(np.transpose(np.array(np.where(valencearray > 2))))
    for k in listJointAndcrowded:
        connNeighborsList = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + d)
            if outOfPixBounds(nearByCoordinate, aShape):
                continue
            if skeletonIm[nearByCoordinate] == 0:
                continue
            connNeighbors = skeletonImLabel[nearByCoordinate]
            connNeighborsList.append(connNeighbors)
        skeletonImLabel[tuple(k)] = _setLabelJointCrowdedpoints(connNeighborsList)
    assert _intersectAssert(list(set(map(tuple, list(np.transpose(np.nonzero(skeletonImLabel)))))), list(set(map(tuple, list(np.transpose(np.nonzero(skeletonIm)))))))
    return skeletonImLabel


def _getSourcesOfShortestpaths(dilatedValenceObjectLoc, dilatedLabelledObjectLoc):
    # print("in _getSourcesOfShortestpaths")
    listNZI = list(np.transpose(np.array(np.where(dilatedLabelledObjectLoc == 4))))
    listIndex = [(coord, dilatedValenceObjectLoc[tuple(coord)]) for coord in listNZI]
    if len(listNZI) == 1:
        src = listNZI[0]
    else:
        summationList = [sum([np.sum(np.square(value - pt)) for pt in listNZI]) / valence for value, valence in listIndex]
    src = listNZI[np.argmin(summationList)]
    return tuple(src)


def _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc):
    """
       exit is a end or middle point
    """
    # print("in _getExitsOfShortestpaths")
    a = set(map(tuple, list(np.transpose(np.nonzero(dilatedRegionExits)))))
    dilatedRegionExits[dilatedRegionExits == 0] = 2
    dilatedRegionExits[dilatedLabelledObjectLoc == 4] = 0
    distTransformedIm = np.square(ndimage.distance_transform_edt(dilatedRegionExits))
    b = np.array(np.where(distTransformedIm <= 3)).T
    b = set(map(tuple, b))
    return list(a & b)


def _findShortestPathFromCRcenterToexit(valencearray, source, dest):
    """
       dijkstra's shortest path, route through the array across the
       minimum cost path
    """
    # print("in _findShortestPathFromCRcenterToexit")
    assert type(source) == tuple
    assert type(dest) == tuple
    indices, weight = route_through_array(valencearray, source, dest, fully_connected=True)
    indices = np.array(indices).T
    path = np.zeros_like(valencearray)
    path[indices[0], indices[1], indices[2]] = 1
    return path


def getShortestPathskeleton(skeletonIm):
    se = np.ones([3] * skeletonIm.ndim, dtype=np.uint8)
    startt = time.time()
    aShape = skeletonIm.shape
    labelInput, noOfObjects = ndimage.measurements.label(skeletonIm, structure=se)
    skeletonImNew = np.zeros_like(skeletonIm)
    valencearray = _setValenceOfarray(skeletonIm)
    if np.sum(valencearray) == 0:
        return skeletonIm
    else:
        skeletonLabelled = _getAllLabelledarray(skeletonIm, valencearray)
        crowdedRegion = np.zeros_like(skeletonLabelled)
        crowdedRegion[skeletonLabelled == 4] = 1
        label, noOfCrowdedregions = ndimage.measurements.label(crowdedRegion, structure=se)
        if np.max(skeletonLabelled) < 4:
            return skeletonIm
        elif np.array_equal(crowdedRegion, skeletonIm):
            srcs = _getSourcesOfShortestpaths(valencearray, skeletonLabelled)
            for i in srcs:
                skeletonImNew[i] = True
            return skeletonImNew
        else:
            objectify = ndimage.find_objects(label)
            exits = np.logical_or(skeletonLabelled == 1, skeletonLabelled == 2)
            for i in range(0, noOfCrowdedregions):
                loc = objectify[i]
                if skeletonIm.ndim == 3:
                    zcoords = loc[0]; ycoords = loc[1]; xcoords = loc[2]
                    regionLowerBoundZ = zcoords.start - 1; regionLowerBoundY = ycoords.start - 1; regionLowerBoundX = xcoords.start - 1
                    regionUpperBoundZ = zcoords.stop + 1; regionUpperBoundY = ycoords.stop + 1; regionUpperBoundX = xcoords.stop + 1
                    bounds = [regionLowerBoundZ, regionLowerBoundY, regionLowerBoundX, regionUpperBoundZ, regionUpperBoundY, regionUpperBoundX]
                    if outOfPixBounds(tuple((bounds[0], bounds[1], bounds[2])), aShape) or outOfPixBounds(tuple((bounds[3], bounds[4], bounds[5])), aShape):
                        for count, i in enumerate(bounds):
                            if i < 0:
                                bounds[count] = 0
                    dilatedValenceObjectLoc = valencearray[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
                    dilatedLabelledObjectLoc = skeletonLabelled[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
                    dilatedLabelledObjectLoc[dilatedLabelledObjectLoc == 0] = 255
                    dilatedRegionExits = exits[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
                    src = _getSourcesOfShortestpaths(dilatedValenceObjectLoc, dilatedLabelledObjectLoc)
                    dests = _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc)
                    for dest in dests:
                        # print(dest)
                        dilatedLabelledObjectLoc1 = _findShortestPathFromCRcenterToexit(dilatedLabelledObjectLoc, src, dest)
                        skeletonImNew[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]] = np.logical_or(skeletonImNew[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]], dilatedLabelledObjectLoc1)
                else:
                    ycoords = loc[0]; xcoords = loc[1]
                    regionLowerBoundY = ycoords.start - 1; regionLowerBoundX = xcoords.start - 1
                    regionUpperBoundY = ycoords.stop + 1; regionUpperBoundX = xcoords.stop + 1
                    bounds = [regionLowerBoundY, regionLowerBoundX, regionUpperBoundY, regionUpperBoundX]
                    if outOfPixBounds(tuple((bounds[0], bounds[1])), aShape) or outOfPixBounds(tuple((bounds[2], bounds[3])), aShape):
                        for count, i in enumerate(bounds):
                            if i < 0:
                                bounds[count] = 0
                    dilatedValenceObjectLoc = valencearray[bounds[0]: bounds[2], bounds[1]: bounds[3]]
                    dilatedLabelledObjectLoc = skeletonLabelled[bounds[0]: bounds[1], bounds[1]: bounds[3]]
                    dilatedLabelledObjectLoc[dilatedLabelledObjectLoc == 0] = 255
                    dilatedRegionExits = exits[bounds[0]: bounds[2], bounds[1]: bounds[3]]
                    srcs = _getSourcesOfShortestpaths(dilatedValenceObjectLoc, dilatedLabelledObjectLoc)
                    dests = _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc)
                    for src, dest in itertools.product(srcs, dests):
                        dilatedLabelledObjectLoc1 = _findShortestPathFromCRcenterToexit(dilatedLabelledObjectLoc, src, dest)
                        skeletonImNew[bounds[0]: bounds[2], bounds[1]: bounds[3]] = np.logical_or(skeletonImNew[bounds[0]: bounds[2], bounds[1]: bounds[3]], dilatedLabelledObjectLoc1)
            skeletonImNew[skeletonLabelled < 4] = True
            skeletonImNew[skeletonLabelled == 0] = False
            skeletonImNew[np.logical_and(valencearray == 0, skeletonIm == 1)] = 1  # see if isolated voxels can be removed (answer: yes)
            label_img1, countObjects = ndimage.measurements.label(skeletonIm, structure=se)
            label_img2, countObjectsShorty = ndimage.measurements.label(skeletonImNew, structure=se)
            assert countObjects == countObjectsShorty
            print("time taken is %0.3f seconds" % (time.time() - startt))
            return skeletonImNew


if __name__ == '__main__':
    skeletonIm = np.load(input("enter a path to your skeleton volume with crowded regions------"))
    shortestPathSkel = getShortestPathskeleton(skeletonIm)
