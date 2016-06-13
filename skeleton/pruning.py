import time
import numpy as np
from scipy.ndimage.filters import convolve
from scipy import ndimage
# from skeleton.thin3DVolume import getThinned3D
# from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton

"""
    code to prune the spurious edges on skeleton
    References:
    https://en.wikipedia.org/wiki/Pruning_(morphology)
    http://www.mathworks.com/matlabcentral/answers/88284-remove-the-spurious-edge-of-skeleton?requestedDomain=www.mathworks.com
"""
template = np.ones((3, 3, 3), dtype=np.uint8)
template[1, 1, 1] = 0


def bwmorph(skel, str):
    valencearray = convolve(np.uint8(skel), template, mode='constant', cval=0)
    valencearray[skel == 0] = 0
    mask = np.zeros_like(skel)
    if str is 'endpoints':
        mask[valencearray == 1] = 1
    else:
        mask[valencearray >= 3] = 1
    return mask

start_prune = time.time()
filePath = input("please enter a root directory where your 3D input is---")
# skel = getShortestPathSkeleton(getThinned3D(np.load(filePath)))
skel = np.load(filePath)
label_img1, countObjects = ndimage.measurements.label(skel, structure=np.ones((3, 3, 3), dtype=np.uint8))
B = bwmorph(skel, 'branchpoints')
E = bwmorph(skel, 'endpoints')
listEndIndices = list(np.transpose(np.array(np.where(E != 0))))
listBranchIndices = list(set(map(tuple, list(np.transpose(np.nonzero(B))))))
listIndices = list(np.transpose(np.array(np.where(skel != 0))))
skelD = np.copy(skel)
for endPoint in listEndIndices:
    listOfBranchDists = []
    D = np.zeros(skel.shape)
    for item in listIndices:
        dist = np.sum(np.square(endPoint - item))
        tupItem = tuple(item)
        D[tupItem] = dist
        if tupItem in listBranchIndices:
            listOfBranchDists.append(dist)
    skelD[D < min(listOfBranchDists)] = 0
E = bwmorph(skelD, 'endpoints')
template = np.ones((3, 3, 3), dtype=np.uint8)
skelNew = ndimage.morphology.binary_dilation(skelD, structure=template, iterations=1, mask=E)
label_img2, countObjectsPruned = ndimage.measurements.label(skelNew, structure=np.ones((3, 3, 3), dtype=np.uint8))
assert countObjects == countObjectsPruned, "Number of disconnected objects in pruned skeleton {} is greater than input objects {}".format(countObjectsPruned, countObjects)
print("time taken is %0.3f seconds" % (time.time() - start_prune))
