import itertools
import numpy as np
# import time
# import copy

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

template = np.ones((3, 3, 3), dtype=np.uint8)
template[1, 1, 1] = 0


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
    if set((1, 2)) == set(connNeighbors) or {1} == set(connNeighbors) or {2} == set(connNeighbors):
        return 3
    else:
        return 4


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

stepDirect = itertools.product((-1, 0, 1), repeat=3)
listStepDirect = list(stepDirect)
listStepDirect.remove((0, 0, 0))
listStepDirect = list(map(np.array, listStepDirect))


def _getAllLabelledarray(skeletonIm, valencearray):
    """
       label end, middle, joint, crowded points and a
       crowded region as 1, 2, 3, 4 and 5 respectively
    """
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
    return skeletonImLabel


def _getSourcesOfShortestpaths(dilatedValenceObjectLoc, dilatedLabelledObjectLoc):
    listNZI = list(np.transpose(np.array(np.where(dilatedLabelledObjectLoc == 4))))
    listIndex = [(coord, dilatedValenceObjectLoc[tuple(coord)]) for coord in listNZI]
    if len(listNZI) == 1:
        srcs = listNZI[0]
    else:
        summationList = [sum([np.sum(np.square(value - pt)) for pt in listNZI]) / valence for value, valence in listIndex]
    srcs = [tuple(item2) for item1, item2 in zip(summationList, listNZI) if item1 == min(summationList)]
    return srcs


def _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc):
    """
       exit is a end or middle point
    """
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
    indices, weight = route_through_array(valencearray, source, dest, fully_connected=True)
    indices = np.array(indices).T
    path = np.zeros_like(valencearray)
    path[indices[0], indices[1], indices[2]] = 1
    return path


def getShortestPathskeleton(skeletonIm):
    se = np.ones([3] * 3, dtype=np.uint8)
    skeletonImNew = np.zeros_like(skeletonIm)
    valencearray = convolve(np.uint8(skeletonIm), template, mode='constant', cval=0)
    valencearray[skeletonIm == 0] = 0
    skeletonLabelled = _getAllLabelledarray(skeletonIm, valencearray)
    crowdedRegion = np.zeros_like(skeletonLabelled)
    crowdedRegion[skeletonLabelled == 4] = 1
    label, noOfCrowdedregions = ndimage.measurements.label(crowdedRegion, structure=se)
    if np.max(skeletonLabelled) < 4:
        return skeletonIm
    else:
        objectify = ndimage.find_objects(label)
        exits = np.logical_or(skeletonLabelled == 1, skeletonLabelled == 2)
        for i in range(0, noOfCrowdedregions):
            loc = objectify[i]
            zcoords = loc[0]; ycoords = loc[1]; xcoords = loc[2]
            regionLowerBoundZ = zcoords.start - 1; regionLowerBoundY = ycoords.start - 1; regionLowerBoundX = xcoords.start - 1
            regionUpperBoundZ = zcoords.stop + 1; regionUpperBoundY = ycoords.stop + 1; regionUpperBoundX = xcoords.stop + 1
            bounds = [regionLowerBoundZ, regionLowerBoundY, regionLowerBoundX, regionUpperBoundZ, regionUpperBoundY, regionUpperBoundX]
            bounds = [0 if i < 0 else i for i in bounds]
            dilatedValenceObjectLoc = valencearray[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
            dilatedLabelledObjectLoc = skeletonLabelled[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
            dilatedLabelledObjectLoc[dilatedLabelledObjectLoc == 0] = 255
            dilatedRegionExits = exits[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
            srcs = _getSourcesOfShortestpaths(dilatedValenceObjectLoc, dilatedLabelledObjectLoc)
            dests = _getExitsOfShortestpaths(dilatedRegionExits, dilatedLabelledObjectLoc)
            for src, dest in itertools.product(srcs, dests):
                dilatedLabelledObjectLoc1 = _findShortestPathFromCRcenterToexit(dilatedLabelledObjectLoc, src, dest)
                skeletonImNew[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]] = np.logical_or(skeletonImNew[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]], dilatedLabelledObjectLoc1)
        skeletonImNew[skeletonLabelled < 4] = True
        skeletonImNew[skeletonLabelled == 0] = False
        skeletonImNew[np.logical_and(valencearray == 0, skeletonIm == 1)] = 1  # see if isolated voxels can be removed (answer: yes)
        return skeletonImNew
