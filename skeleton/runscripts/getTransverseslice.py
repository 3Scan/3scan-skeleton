import os
from scipy.misc import imread, imsave
import numpy as np
import cv2
from scipy import ndimage

root = '/media/pranathi/DATA/ii-5016-15-ms-brain_1920/downsampledslices/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
transverseSlice = np.zeros((799, 17480 / 7), dtype=np.uint8)
count = 0
klist = [2767, 3667, 6367, 7267]
for k in klist:
    for i in range(0, len(listOfJpgs)):
        image = imread(root + 'downsampledslice%i.png' % i)
        transverseSlice[count, :] = image[:, 395]
        print(count)
        count += 1
    imsave("transverseSlice%i.png" % k, transverseSlice)

maskBrain = np.load('/media/pranathi/DATA/NPYS/maskDownsampled10.npy')
maskArtVein = np.load('/media/pranathi/DATA/NPYS/maskArtVein.npy')
mask = np.zeros((80, 2497), dtype=np.uint8)
maskArt = np.zeros((80, 2497), dtype=np.uint8)
for i in range(mask.shape[0]):
    mask[i, :] = maskBrain[i, :, 395]
    maskArt[i, :] = maskArtVein[i, :, 395]
maskArt[maskArt != 255] = 1
maskArt[maskArt == 255] = 0
maskArt = ndimage.interpolation.zoom(maskArt, zoom=[9.9875, 1], order=0)
mask = ndimage.interpolation.zoom(mask, zoom=[9.9875, 1], order=0)
transverseSlice = transverseSlice * mask * maskArt
paramImage = imread("/home/pranathi/src/3scan-skeleton/transverseSlicepercentVasc2767.png")
color_img = cv2.cvtColor(transverseSlice, cv2.COLOR_GRAY2RGB)
nonZeros = list(set(map(tuple, np.transpose(np.nonzero(paramImage)))))
for index in nonZeros:
    color_img[index[0], index[1], 0] = paramImage[index]
    color_img[index[0], index[1], 1] = 0
    color_img[index[0], index[1], 2] = 0
imsave("transverseSliceColorOverlapPercentVasc.png", color_img)
# mask = np.zeros_like(paramImage)
# mask[paramImage == 0] = 255
# color_img[:, :, 1] = mask
