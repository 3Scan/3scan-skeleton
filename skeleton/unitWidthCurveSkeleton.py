import itertools
import time

import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
from skimage.graph import route_through_array

from skeleton.networkxGraphFromArray import LIST_STEP_DIRECTIONS3D

"""
The goal of this algorithm is to generate a topologically and geometrically preserved
unit voxel width skeleton implemented according to the paper
Generation of Unit-Width Curve Skeletons Based on Valence Driven Spatial Median (VDSM)
Tao Wang, Irene Cheng
Advances in Visual Computing
Volume 5358 of the series Lecture Notes in Computer Science pp 1051-1060
"""

TEMPLATE = np.ones((3, 3, 3), dtype=np.uint8)
TEMPLATE[1, 1, 1] = 0
ARRAYSTEPDIRECTIONS3D = np.array(LIST_STEP_DIRECTIONS3D).copy(order='C')


def outOfPixBounds(nearByCoordinate, aShape):
    """
    Return a boolean saying if any of x, y, z of nearByCoordinate is outside of aShape
    Parameters
    ----------
    nearByCoordinate : tuple
        tuple of shape 2D or 3D

    aShape : tuple
        tuple of shape 2D or 3D

    Returns
    -------
    onBound : Bpolean
        1 if its on or over boundary, 0 if not
    """
    onBound = 0
    for index, maxVal in enumerate(aShape):
        isAtBoundary = nearByCoordinate[index] >= maxVal or nearByCoordinate[index] < 0
        if isAtBoundary:
            onBound = 1
            break
    return onBound


def _getAllLabelledArray(skeletonIm, valenceArray):
    """
    Return labelled array
    Parameters
    ----------
    skeletonIm : Numpy array
        3D binary numpy array

    valenceArray : Numpy array
        3D binary numpy array

    Returns
    -------
    skeletonImLabel : Numpy array
        3D labeled array of same shape and type uint8

    Notes
    -----
    Labels are as follows
    end point = 1 middle point = 2 joint point = 3 crowded point = 4 crowded region = 5
    Definitions of this points are in the reference paper
    """
    # allocate an array of same shape for the labels
    skeletonImLabel = np.zeros(skeletonIm.shape, dtype=np.uint8)
    # label end points: degree = 1
    skeletonImLabel[valenceArray == 1] = 1
    aShape = np.shape(valenceArray)
    # label middle points: degree = 2
    listIterateMiddle = list(np.transpose(np.array(np.where(valenceArray == 2))))
    for k in listIterateMiddle:
        connNeighborsIndices = []
        for d in LIST_STEP_DIRECTIONS3D:
            nearByCoordinate = tuple(k + d)
            if outOfPixBounds(nearByCoordinate, aShape) or skeletonIm[nearByCoordinate] == 0:
                continue
            connNeighbors = skeletonIm[nearByCoordinate]
            connNeighborsIndices.append(np.array(nearByCoordinate))
        # middle point only if squared distance between their coordinates is greater than 3
        # i.e they are not 26 connected
        if np.sum((connNeighborsIndices[0] - connNeighborsIndices[1]) ** 2) > 3:
            skeletonImLabel[tuple(k)] = 2
        else:
            skeletonImLabel[tuple(k)] = 4
    # label joint and crowded points
    listJointAndcrowded = list(np.transpose(np.array(np.where(valenceArray > 2))))
    for k in listJointAndcrowded:
        connNeighborsList = []
        for d in ARRAYSTEPDIRECTIONS3D:
            nearByCoordinate = tuple(k + d)
            if outOfPixBounds(nearByCoordinate, aShape) or skeletonIm[nearByCoordinate] == 0:
                continue
            connNeighbors = skeletonImLabel[nearByCoordinate]
            connNeighborsList.append(connNeighbors)
        # joint point degree > 2 1nd order neighbors are either end points or middle points
        if set((1, 2)) == set(connNeighborsList) or {1} == set(connNeighborsList) or {2} == set(connNeighborsList):
            skeletonImLabel[tuple(k)] = 3
        else:  # crowded points
            skeletonImLabel[tuple(k)] = 4
    return skeletonImLabel


def getShortestPathSkeleton(skeletonIm):
    """
    Return unit width curve/shortest path skeleton
    Parameters
    ----------
    skeletonIm : Numpy array
        3D binary numpy array of boolean dtype

    Returns
    -------
    Numpy array
        3D binary crowded region removed numpy array of the same shape and dtype
    """
    assert (skeletonIm.dtype is bool or np.unique(skeletonIm).tolist() == [0, 1] or
            np.unique(skeletonIm).tolist() == [0] or
            np.unique(skeletonIm).tolist() == [1]), "skeletonIm is not boolean it is {}".format(skeletonIm.dtype)
    if len(skeletonIm.shape) == 2:
        return skeletonIm
    else:
        # initialize
        start = time.time()
        se = np.ones([3] * 3, dtype=np.uint8)
        skeletonImNew = np.zeros_like(skeletonIm, dtype=bool)
        # compute degrees
        valencearray = convolve(np.uint8(skeletonIm), TEMPLATE, mode='constant', cval=0)
        valencearray[skeletonIm == 0] = 0
        skeletonLabelled = _getAllLabelledArray(skeletonIm, valencearray)
        # find and locate crowded regions exist
        # crowded region - 26 connected crowded joint point
        crowdedRegion = np.zeros_like(skeletonLabelled)
        crowdedRegion[skeletonLabelled == 4] = 1
        label, countCrowdedRegions = ndimage.measurements.label(crowdedRegion, structure=se)
        if np.max(skeletonLabelled) < 4:  # no crowded regions
            print("no crowded regions")
            return skeletonIm
        else:
            objectify = ndimage.find_objects(label)
            # detect exits
            exits = np.logical_or(skeletonLabelled == 1, skeletonLabelled == 2)
            for crowdRegion in range(countCrowdedRegions):
                loc = objectify[crowdRegion]
                # find boundaries of the crowded region
                bounds = [(max(coords.start - 1, 0), max(coords.stop + 1, 0)) for coords in loc]
                dilatedValenceObjectLoc = valencearray[bounds[0][0]: bounds[0][1],
                                                       bounds[1][0]: bounds[1][1],
                                                       bounds[2][0]: bounds[2][1]]
                dilatedRegionExits = exits[bounds[0][0]: bounds[0][1],
                                           bounds[1][0]: bounds[1][1],
                                           bounds[2][0]: bounds[2][1]]
                dilatedLabelledObjectLoc = skeletonLabelled[bounds[0][0]: bounds[0][1],
                                                            bounds[1][0]: bounds[1][1],
                                                            bounds[2][0]: bounds[2][1]]
                listSourceIndices = list(np.transpose(np.array(np.where(dilatedLabelledObjectLoc == 4))))
                listExitIndices = list(np.transpose(np.array(np.where(dilatedRegionExits != 0))))
                listOfExits = []
                # detect exits in the subregion
                for items in listExitIndices:
                    for item in listSourceIndices:
                        dist = np.sum(np.square(items - item))
                        if dist > 3:
                            continue
                        listOfExits.append(tuple(items))
                dests = list(set(listOfExits))
                listIndex = [(coord, dilatedValenceObjectLoc[tuple(coord)]) for coord in listSourceIndices]
                # determiine the centroid of the crowded region
                if len(listSourceIndices) == 1:
                    srcs = listSourceIndices[0]
                else:
                    summationList = [sum([np.sum(np.square(value - pt)) for pt in listSourceIndices]) / valence
                                     for value, valence in listIndex]
                    srcs = [tuple(item2) for item1, item2 in zip(summationList, listSourceIndices)
                            if item1 == min(summationList)]
                dilatedLabelledObjectLoc[dilatedLabelledObjectLoc == 0] = 255
                # find shortest paths from centroid to exits
                for src, dest in itertools.product(srcs, dests):
                    indices, weight = route_through_array(dilatedLabelledObjectLoc, src, dest, fully_connected=True)
                    indices = np.array(indices).T
                    dilatedLabelledObjectLoc1 = np.zeros_like(dilatedLabelledObjectLoc)
                    dilatedLabelledObjectLoc1[indices[0], indices[1], indices[2]] = 1
                    skeletonImNew[bounds[0][0]: bounds[0][1],
                                  bounds[1][0]: bounds[1][1],
                                  bounds[2][0]: bounds[2][1]] = np.logical_or(skeletonImNew[bounds[0][0]: bounds[0][1],
                                                                                            bounds[1][0]: bounds[1][1],
                                                                                            bounds[2][0]: bounds[2][1]],
                                                                              dilatedLabelledObjectLoc1)
                progress = int((100 * crowdRegion) / countCrowdedRegions)
                print("cleaning crowded regions in progress {}% \r".format(progress), end="", flush=True)
            # output the unit width curve skeleton
            skeletonImNew[skeletonLabelled < 4] = True
            skeletonImNew[skeletonLabelled == 0] = False
            skeletonImNew[np.logical_and(valencearray == 0, skeletonIm == 1)] = 0  # remove isolated single voxels
            print("time taken to find unit width curve skeleton is %0.3f seconds" % (time.time() - start))
            return skeletonImNew

if __name__ == '__main__':
    skeletonIm = np.load(input("enter a path to your skeleton volume with crowded regions------"))
    shortestPathSkel = getShortestPathSkeleton(skeletonIm)
