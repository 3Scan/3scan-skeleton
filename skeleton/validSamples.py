import itertools
from collections import Counter
from cv2 import imread
import numpy as np
import os

root = '/home/pranathi/mask/maskBrain/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
shape = (len(listOfJpgs), 2497, 1147)
mask = np.zeros(shape, dtype=np.uint8)
count = 0
prevSum = 1
for i in range(0, 800, 10):
    image = imread(root + 'maskBrainSlice%i.png' % i)
    # plt.imshow(image[:, :, 0], cmap='gray')
    # plt.show()
    # o = np.flipud(np.rot90(image, 3))
    # imsave(root + 'rot/' 'maskBrainSliceRot%i.png' % i, o)
    mask[count, :, :] = image[:, :, 0]
    prevSum += np.sum(mask)
    assert np.sum(mask) < prevSum
    # assert [mask[count].flatten()] != 0
    count = count + 1
maskdsN = np.zeros(mask.shape, dtype=bool)
maskdsN[mask == 255] = 1
maskdsN[mask != 255] = 0

root = '/home/pranathi/mask/rightOrderedMaskArtVein/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
shape = (len(listOfJpgs), 2497, 1147)
maskArtVein = np.zeros(shape, dtype=np.uint8)
count = 0
for i in range(0, 800, 10):
    image = imread(root + 'maskedSliceSub%i.png' % i)
    # o = np.flipud(np.rot90(image, 3))
    # imsave(root + 'rot/' 'maskBrainSliceRot%i.png' % i, o)
    maskArtVein[count, :, :] = image[:, :, 0]
    # assert [mask[count].flatten()] != 0
    count = count + 1
# maskArtVein = np.load('/media/pranathi/DATA/NPYS/maskArtVein.npy')
iskpy = iskpx = 280; iskpz = int(0.5 + 560 * 0.7 / 5.0)
ilist = list(range(30, 799 - 10, iskpz))
klist = [k for k in range(67, 8026 - 68, iskpx) if k < 1600 or (k > 2350 and k < 4800) or (k > 5520 and k < 8020)]
jlist = range(67, 17480 - 68, iskpy)
it = list(itertools.product(ilist, jlist, klist))
listElements = list(map(list, it))
subsubVolShape = (135, 135, 135)
cornersOfCube = list(map(list, itertools.product((-1, 1), repeat=3)))
cornersOfCube = [list((element[0] * 9, element[1] * 67, element[2] * 67)) for element in cornersOfCube]
validit = [element for element, elementList in zip(it, listElements) for i in cornersOfCube if maskdsN[(int((elementList[0] + i[0]) / 10), int((elementList[1] + i[1]) / 7), int((elementList[2] + i[2]) / 7))] and maskArtVein[(int((elementList[0] + i[0]) / 10), int((elementList[1] + i[1]) / 7), int((elementList[2] + i[2]) / 7))] != 255]
validSamp = list(set(validit))
c = Counter(validit)
validCenters = [element for element in c if c[element] == 8]
centers = []
for i, j, k in validCenters:
    jm = int(j / 7); km = int(k / 7)
    imf = int((i - 9) / 10)
    iml = int((i + 9) / 10)
    maskSub = maskArtVein[imf: iml + 1, jm - 9:jm + 10, km - 9: km + 10]
    if np.sum(maskSub) != 0:
        centers.append(tuple((i, j, k)))
