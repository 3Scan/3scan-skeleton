import os
import numpy as np
from scipy.misc import imread
from scipy import ndimage
import itertools
from collections import Counter


def outOfPixBounds(nearByCoordinate, aShape):
    onbound = 1
    for index, maxVal in enumerate(aShape):
        isAtBoundary = nearByCoordinate[index] >= maxVal or nearByCoordinate[index] < 0
        if isAtBoundary:
            onbound = 0
            break
        else:
            continue
    return onbound

centers = []
root = '/media/pranathi/DATA/ii-5016-15-ms-brain_1920/downsampledslices/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
k = 2767
count = 0
transverseSlice = np.zeros((799, 17480 / 7), dtype=np.uint8)
for i in range(0, len(listOfJpgs)):
    image = imread(root + 'downsampledslice%i.png' % i)
    transverseSlice[count, :] = image[:, int(k / 7.0)]
    count += 1
for coords in centers:
    transverseSlice[coords[0] - 1: coords[0] + 2, int((coords[1] + 6) / 7.0) - 1: int((coords[1] + 6) / 7.0) + 2] = 128

maskBrain = np.load('/media/pranathi/DATA/NPYS/maskDownsampled10.npy')
maskArtVein = np.load('/media/pranathi/DATA/NPYS/maskArtVein.npy')
iskpx = 900
iskpz = 10
iskpy = 71
ilist = list(range(60, 799 - 10, iskpz))
klist = [3667, 6367, 7267]
klist = [2767]
# klist = [k for k in range(67, 8026 - 68, iskpx) if (k > 2420 and k < 4000) or (k > 5500 and k < 8000)]
it = list(itertools.product(ilist, range(67, 17480 - 68, iskpy), klist))
listElements = list(map(list, it))
subsubVolShape = (135, 135, 135)
cornersOfCube = list(map(list, itertools.product((-1, 1), repeat=3)))
aShape = (2497, 1147)
cornersOfCube = [list((element[0] * 9, element[1] * 130, element[2] * 130)) for element in cornersOfCube]
validit = [element for element, elementList in zip(it, listElements) for i in cornersOfCube if outOfPixBounds((int((elementList[1] + i[1] + 6) / 7.0), int((elementList[2] + i[2] + 6) / 7.0)), aShape) and maskBrain[(int((elementList[0] + i[0] + 9) / 10.0), int((elementList[1] + i[1] + 6) / 7.0), int((elementList[2] + i[2] + 6) / 7.0))]]
c = Counter(validit)
validCenters = [element for element in c if c[element] == 8]
# maskArtVein = ndimage.interpolation.zoom(maskArtVein, zoom=[5 / 0.7037037, 7, 7], order=0)
for i, j, k in validCenters:
    jm = int((j + 6) / 7.0)
    km = int((k + 6) / 7.0)
    imf = int((i - 9 - 9) / 10.0)
    iml = int((i + 9 + 9) / 10.0)
    maskSub = maskArtVein[imf: iml + 1, jm - 9: jm + 10, km - 9, km + 10]
    if np.sum(maskSub) == 0:
        centers.append(tuple((i, j, k)))
mask = np.zeros((80, 2497, 19), dtype=np.uint8)
maskArt = np.zeros((80, 2497, 19), dtype=np.uint8)
for i in range(mask.shape[0]):
    mask[i, :] = maskBrain[i, :, int(k / 7.0) - 9: int(k / 7.0) + 10]
    maskArt[i, :] = maskArtVein[i, :, int(k / 7.0) - 9: int(k / 7.0) + 10]
maskArt = np.amax(maskArt, 2)
mask = np.amax(mask, 2)
maskArt = ndimage.interpolation.zoom(maskArt, zoom=[9.9875, 1], order=0)
mask = ndimage.interpolation.zoom(mask, zoom=[9.9875, 1], order=0)
maskArt = 255 * maskArt
mask = 255 * mask
for coords in validCenters:
    print(coords)
    maskArt[coords[0] - 1: coords[0] + 2, int((coords[1] + 6) / 7.0) - 1: int((coords[1] + 6) / 7.0) + 2] = 128
    mask[coords[0] - 1: coords[0] + 2, int((coords[1] + 6) / 7.0) - 1: int((coords[1] + 6) / 7.0) + 2] = 128
