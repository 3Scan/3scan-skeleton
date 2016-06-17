import numpy as np
import os
from scipy.misc import imread
from scipy import ndimage
from skimage.filters import threshold_otsu
from scipy.signal import fftconvolve
from skeleton.runscripts.thin3DVolumeRV import getSkeleton3D
from skeleton.runscripts.segmentLengthsRV import getSegmentStats
import matplotlib.pyplot as plt

root = '/media/pranathi/User Data/mouseBrain-CS/'
formatOfFiles = 'jpg'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
i = 3707
subVolume = np.zeros((95, 6050, 7790), dtype=np.uint8)
subSubvolume = np.zeros((95, 135, 135), dtype=np.uint8)
subList = listOfJpgs[i - 47: i + 48]
count = 0
for fileName in subList:
    subVolume[count][:][:] = imread(fileName)
    count += 1
j = 2567
k = 3357
subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
subSubvolume = 255 - subSubvolume
subSubvolumefl = np.float64(subSubvolume)
subSubvolumeavg = np.zeros((19, 135, 135))
count = 0
for i in range(0, subSubvolume.shape[0], 5):
    subSubvolumeavg[count, :, :] = (subSubvolumefl[i, :, :] + subSubvolumefl[i + 1, :, :] + subSubvolumefl[i + 2, :, :] + subSubvolumefl[i + 3, :, :] + subSubvolumefl[i + 4, :, :]) / 5
    count += 1
npy = "coronalgreyscaleSubSubVolume_{}_{}_{}.npy".format(3707 + 13, j, k)
npyAvg = "coronalgreyscaleSubSubVolumeavged_{}_{}_{}.npy".format(3707 + 13, j, k)
subSubvolume = ndimage.interpolation.zoom(subSubvolume, [135 / 95, 1, 1], order=2, prefilter=False)
subSubvolumeavg = np.uint8(subSubvolumeavg)
subSubvolumeavg2 = ndimage.interpolation.zoom(subSubvolumeavg, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
np.save(npy, subSubvolume)
np.save(npyAvg, subSubvolumeavg)
histData, bins = np.histogram(subSubvolume.flatten(), 256, [0, 256])
histDataavg, bins = np.histogram(subSubvolumeavg.flatten(), 256, [0, 256])
vsum = 0
totalSize = 2460375.0
tsum = 0.8 * 95 * 135 * 135
t1sum = 0.9 * 95 * 135 * 135
t2sum = 0.95 * 95 * 135 * 135
tsumAvg = 0.8 * 19 * 135 * 135
t1sumAvg = 0.9 * 19 * 135 * 135
t2sumAvg = 0.95 * 19 * 135 * 135
selem = np.zeros((31, 31, 31), dtype=np.uint16)
xs, ys, zs = np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
selem[(r < 1)] = 1
selem2 = np.zeros((29, 29, 29), dtype=np.uint16)
xs, ys, zs = np.mgrid[-1:1:29j, -1:1:29j, -1:1:29j]
r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
selem2[(r < 1)] = 1
selem3 = np.zeros((27, 27, 27), dtype=np.uint16)
xs, ys, zs = np.mgrid[-1:1:27j, -1:1:27j, -1:1:27j]
r = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
selem3[(r < 1)] = 1
for i in range(0, 255):
    if vsum < tsum:
        vsum += histData[i]
        t = i
subSubvolume[subSubvolume < t] = 0
l1 = np.array([y for y in subSubvolume.ravel() if y != 0])
threshold = threshold_otsu(l1)
interpolatedIm = subSubvolume > threshold
i = np.ascontiguousarray(interpolatedIm, dtype=np.uint16)
erode_im = fftconvolve(i, selem, mode='same')
percentVasc = np.sum(interpolatedIm) / totalSize
threshPath = npy.replace('greyscale', 'threshold')
if np.max(erode_im) <= 8000 and percentVasc <= 0.1:
    np.save(threshPath, interpolatedIm)
else:
    t1 = t
    for i in range(t + 1, 255):
        if vsum < t1sum:
            vsum += histData[i]
            t1 = i
    subSubvolume[subSubvolume < t1] = 0
    l1 = np.array([y for y in subSubvolume.ravel() if y != 0])
    threshold = threshold_otsu(l1)
    interpolatedIm = subSubvolume > (0.85 * threshold)
    i = np.ascontiguousarray(interpolatedIm, dtype=np.uint16)
    erode_im = fftconvolve(i, selem2, mode='same')
    percentVasc = np.sum(interpolatedIm) / totalSize
    if np.max(erode_im) <= 6500 and percentVasc <= 0.07:
        np.save(threshPath, interpolatedIm)
    else:
        t2 = t1
        for i in range(t1 + 1, 255):
            if vsum < t2sum:
                vsum += histData[i]
                t2 = i
        subSubvolume[subSubvolume < t2] = 0
        l1 = np.array([y for y in subSubvolume.ravel() if y != 0])
        threshold = threshold_otsu(l1)
        interpolatedIm = subSubvolume > (0.85 * threshold)
        i = np.ascontiguousarray(interpolatedIm, dtype=np.uint16)
        erode_im = fftconvolve(i, selem3, mode='same')
        percentVasc = np.sum(interpolatedIm) / totalSize
        if np.max(erode_im) <= 5000 and percentVasc <= 0.04:
            np.save(threshPath, interpolatedIm)
print("percent vasc original", percentVasc)


vsum = 0
for i in range(0, 255):
    if vsum < tsumAvg:
        vsum += histDataavg[i]
        tAvg = i
subSubvolumeavg[subSubvolumeavg < tAvg] = 0
l1 = np.array([y for y in subSubvolumeavg.ravel() if y != 0])
thresholdavg = threshold_otsu(l1)
interpolatedImavg = ndimage.interpolation.zoom(subSubvolumeavg, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
interpolatedImavg = interpolatedImavg > thresholdavg
i = np.ascontiguousarray(interpolatedImavg, dtype=np.uint16)
erode_im = fftconvolve(i, selem, mode='same')
percentVasc = np.sum(interpolatedImavg) / totalSize
threshPathavg = npyAvg.replace('greyscale', 'threshold')

if np.max(erode_im) <= 8000 and percentVasc <= 0.1:
    np.save(threshPathavg, interpolatedImavg)
else:
    t1avg = tAvg
    for i in range(tAvg + 1, 255):
        if vsum < t1sumAvg:
            vsum += histDataavg[i]
            t1avg = i
    subSubvolumeavg[subSubvolumeavg < t1avg] = 0
    l1 = np.array([y for y in subSubvolumeavg.ravel() if y != 0])
    thresholdavg = threshold_otsu(l1)
    interpolatedImavg = ndimage.interpolation.zoom(subSubvolumeavg, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
    interpolatedImavg = interpolatedImavg > (0.85 * thresholdavg)
    i = np.ascontiguousarray(interpolatedImavg, dtype=np.uint16)
    erode_im = fftconvolve(i, selem2, mode='same')
    percentVasc = np.sum(interpolatedImavg) / totalSize
    if np.max(erode_im) <= 6500 and percentVasc <= 0.07:
        np.save(threshPathavg, interpolatedImavg)
    else:
        t2avg = t1avg
        for i in range(t1 + 1, 255):
            if vsum < t2sumAvg:
                vsum += histDataavg[i]
                t2avg = i
        subSubvolumeavg[subSubvolumeavg < t2avg] = 0
        l1 = np.array([y for y in subSubvolumeavg.ravel() if y != 0])
        threshold = threshold_otsu(l1)
        interpolatedImavg = ndimage.interpolation.zoom(subSubvolumeavg, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
        interpolatedImavg = interpolatedImavg > (0.85 * threshold)
        i = np.ascontiguousarray(interpolatedImavg, dtype=np.uint16)
        erode_im = fftconvolve(i, selem3, mode='same')
        percentVasc = np.sum(interpolatedImavg) / totalSize
        if np.max(erode_im) <= 5000 and percentVasc <= 0.04:
            np.save(threshPathavg, interpolatedImavg)
print("percent vasc artificial", percentVasc)

skeletonPath = threshPath.replace('threshold', 'skeleton')
skeleton = getSkeleton3D(np.load(threshPath))
np.save(skeletonPath, skeleton)
path = (skeletonPath.replace('skeleton', 'stat')).replace('npy', 'txt')
f = open(path, 'a')
dist, T1, T2 = getSegmentStats(skeleton)
d = [str(dist) + "\n", str(T1) + "\n", str(T2) + "\n"]
f.writelines(d)
f.close()
skeletonPathavg = threshPathavg.replace('threshold', 'skeleton')
skeletonAvg = getSkeleton3D(np.load(threshPathavg))
np.save(skeletonPathavg, skeletonAvg)
path = (skeletonPathavg.replace('skeleton', 'stat')).replace('npy', 'txt')
f = open(path, 'a')
dist, T1, T2 = getSegmentStats(skeletonAvg)
d = [str(dist) + "\n", str(T1) + "\n", str(T2) + "\n"]
f.writelines(d)
f.close()


plt.subplot(3, 2, 1)
I = np.load("coronalgreyscaleSubSubVolume_3720_2567_3357.npy")
plt.imshow(np.amax(I, 1), cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(np.amax(subSubvolumeavg2, 1), cmap='gray')
plt.subplot(3, 2, 3)
plt.imshow(np.amax(interpolatedIm, 1), cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(np.amax(interpolatedImavg, 1), cmap='gray')
plt.subplot(3, 2, 5)
plt.imshow(np.amax(skeleton, 1), cmap='gray')
plt.subplot(3, 2, 6)
plt.imshow(np.amax(skeletonAvg, 1), cmap='gray')
