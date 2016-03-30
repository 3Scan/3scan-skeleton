import os
from scipy.misc import imread, imsave
import numpy as np
import cv2
from scipy import ndimage

root = '/media/pranathi/KINGSTON/RESULTS_1sttransverseslice/subsubVolumestat/'
formatOfFiles = 'txt'
listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfNpys.sort()
dictPercetages = {}
dictLength = {}; dictTortuosity = {}
for f in listOfNpys:
    file_contents = open(f, 'r')
    npyIndex = f.replace(".txt", "")
    fileC = [_ for _ in file_contents]
    file_contents.close()
    strs = npyIndex.split('/')[-1].split('_')
    strs[2] = strs[2].replace('~', '')
    i, j, k = [int(s) for s in strs if s.isdigit()]
    dictPercetages[(i, j, k)] = float(fileC[0].replace('\n', ''))
    dictLength[(i, j, k)] = float(fileC[2].replace('\n', ''))
    dictTortuosity[(i, j, k)] = float(fileC[3].replace('\n', ''))

dictList = [dictPercetages, dictLength, dictTortuosity]
imNames = ['percentVasc', 'length', 'tortuosity']
badKeys = [(260, 564, 2767), (260, 2055, 2767), (270, 16042, 2767), (480, 5818, 3667), (380, 2623, 6367), (320, 8303, 7267), (410, 7877, 7267), (350, 9013, 7267)]
klist = [2767, 3667, 6367, 7267]
for k in klist:
    for index, dictStat in enumerate(dictList):
        dictStat = {key: value for key, value in dictStat.items() if key[2] == k}
        maxVal = max(list(dictStat.values()))
        if index < 1:
            dictPercetagesFiltz = {((key[0] - 160), key[1] / 7): (255 * dictStat[key] / maxVal) for key, value in dictStat.items() if key not in badKeys}
            samplePoints = np.zeros((799, 17480 / 7), dtype=np.uint8)
            for coords, values in dictPercetagesFiltz.items():
                samplePoints[int(coords[0]) - 1: int(coords[0]) + 2, int(coords[1]) - 1: int(coords[1]) + 2] = values
        else:
            minVal = min(list(dictStat.values()))
            dictPercetagesFiltz = {((key[0] - 160), key[1] / 7): ((dictStat[key] - minVal) / (maxVal - minVal)) for key, value in dictStat.items() if key not in badKeys}
            samplePoints = np.zeros((799, 17480 / 7), dtype=np.uint8)
            maxVal = max(list(dictPercetagesFiltz.values()))
            for coords, values in dictPercetagesFiltz.items():
                samplePoints[int(coords[0]) - 1: int(coords[0]) + 2, int(coords[1]) - 1: int(coords[1]) + 2] = (255 * values / maxVal)
        imsave("transverseSlice" + imNames[index] + "%i.png" % k, samplePoints)


root = '/media/pranathi/DATA/ii-5016-15-ms-brain_1920/downsampledslices/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
maskBrain = np.load('/media/pranathi/DATA/NPYS/maskDownsampled10.npy')
maskArtVein = np.load('/media/pranathi/DATA/NPYS/maskArtVein.npy')
imNames = ['percentVasc', 'length', 'tortuosity']
channels = [0, 1, 2]
for k in klist:
    count = 0
    transverseSlice = np.zeros((799, 17480 / 7), dtype=np.uint8)
    for i in range(0, len(listOfJpgs)):
        image = imread(root + 'downsampledslice%i.png' % i)
        transverseSlice[count, :] = image[:, int(k / 7)]
        count += 1
    imsave("transverseSlice%i.png" % k, transverseSlice)
    mask = np.zeros((80, 2497), dtype=np.uint8)
    maskArt = np.zeros((80, 2497), dtype=np.uint8)
    for i in range(mask.shape[0]):
        mask[i, :] = maskBrain[i, :, int(k / 7)]
        maskArt[i, :] = maskArtVein[i, :, int(k / 7)]
    maskArt = ndimage.interpolation.zoom(maskArt, zoom=[9.9875, 1], order=0)
    mask = ndimage.interpolation.zoom(mask, zoom=[9.9875, 1], order=0)
    transverseSlice = transverseSlice * mask * maskArt
    for index, values in enumerate(imNames):
        paramImage = imread("/home/pranathi/transverseSlice" + values + "%i.png" % k)
        color_img = cv2.cvtColor(transverseSlice, cv2.COLOR_GRAY2RGB)
        nonZeros = list(set(map(tuple, np.transpose(np.nonzero(paramImage)))))
        for nz in nonZeros:
            color_img[nz[0], nz[1], index] = paramImage[nz]
            remChannels = [channel for channel in channels if channel != index]
            for j in remChannels:
                color_img[nz[0], nz[1], j] = 0
        imsave("transverseSliceColorOverlap" + imNames[index] + "%i.png" % k, color_img)

