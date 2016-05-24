import itertools
import numpy as np

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
        connNeighborsIndices = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + d)
            if outOfPixBounds(nearByCoordinate, aShape) or skeletonIm[nearByCoordinate] == 0:
                continue
            connNeighbors = skeletonIm[nearByCoordinate]
            connNeighborsIndices.append(np.array(nearByCoordinate))
        if np.sum((connNeighborsIndices[0] - connNeighborsIndices[1]) ** 2) > 3:
            skeletonImLabel[tuple(k)] = 2
        else:
            skeletonImLabel[tuple(k)] = 4

    listJointAndcrowded = list(np.transpose(np.array(np.where(valencearray > 2))))
    for k in listJointAndcrowded:
        connNeighborsList = []
        for d in listStepDirect:
            nearByCoordinate = tuple(k + d)
            if outOfPixBounds(nearByCoordinate, aShape) or skeletonIm[nearByCoordinate] == 0:
                continue
            connNeighbors = skeletonImLabel[nearByCoordinate]
            connNeighborsList.append(connNeighbors)
        if set((1, 2)) == set(connNeighborsList) or {1} == set(connNeighborsList) or {2} == set(connNeighborsList):
            skeletonImLabel[tuple(k)] = 3
        else:
            skeletonImLabel[tuple(k)] = 4
    return skeletonImLabel


def getShortestPathSkeleton(skeletonIm):
    se = np.ones([3] * 3, dtype=np.uint8)
    skeletonImNew = np.zeros_like(skeletonIm, dtype=bool)
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
            print("cleaning crowded region # {} / {}".format(i, noOfCrowdedregions))
            loc = objectify[i]
            zcoords = loc[0]
            ycoords = loc[1]
            xcoords = loc[2]
            regionLowerBoundZ = zcoords.start - 1
            regionLowerBoundY = ycoords.start - 1
            regionLowerBoundX = xcoords.start - 1
            regionUpperBoundZ = zcoords.stop + 1
            regionUpperBoundY = ycoords.stop + 1
            regionUpperBoundX = xcoords.stop + 1
            bounds = [regionLowerBoundZ, regionLowerBoundY, regionLowerBoundX, regionUpperBoundZ, regionUpperBoundY, regionUpperBoundX]
            bounds = [0 if i < 0 else i for i in bounds]
            dilatedValenceObjectLoc = valencearray[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
            dilatedRegionExits = exits[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
            dilatedLabelledObjectLoc = skeletonLabelled[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]]
            listSourceIndices = list(np.transpose(np.array(np.where(dilatedLabelledObjectLoc == 4))))
            listExitIndices = list(np.transpose(np.array(np.where(dilatedRegionExits != 0))))
            listOfExits = []
            for items in listExitIndices:
                for item in listSourceIndices:
                    dist = np.sum(np.square(items - item))
                    if dist > 3:
                        continue
                    listOfExits.append(tuple(items))
            dests = list(set(listOfExits))
            listIndex = [(coord, dilatedValenceObjectLoc[tuple(coord)]) for coord in listSourceIndices]
            if len(listSourceIndices) == 1:
                srcs = listSourceIndices[0]
            else:
                summationList = [sum([np.sum(np.square(value - pt)) for pt in listSourceIndices]) / valence for value, valence in listIndex]
            srcs = [tuple(item2) for item1, item2 in zip(summationList, listSourceIndices) if item1 == min(summationList)]
            dilatedLabelledObjectLoc[dilatedLabelledObjectLoc == 0] = 255
            for src, dest in itertools.product(srcs, dests):
                indices, weight = route_through_array(dilatedLabelledObjectLoc, src, dest, fully_connected=True)
                indices = np.array(indices).T
                dilatedLabelledObjectLoc1 = np.zeros_like(dilatedLabelledObjectLoc)
                dilatedLabelledObjectLoc1[indices[0], indices[1], indices[2]] = 1
                skeletonImNew[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]] = np.logical_or(skeletonImNew[bounds[0]: bounds[3], bounds[1]: bounds[4], bounds[2]: bounds[5]], dilatedLabelledObjectLoc1)
        skeletonImNew[skeletonLabelled < 4] = True
        skeletonImNew[skeletonLabelled == 0] = False
        skeletonImNew[np.logical_and(valencearray == 0, skeletonIm == 1)] = 1  # see if isolated voxels can be removed (answer: yes)
        return skeletonImNew
