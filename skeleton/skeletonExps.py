import copy
import time
import os
import numpy as np
from scipy import ndimage
from scipy.misc import imsave
from skeleton.radiusOfNodes import _getBouondariesOfimage
from skimage.filters import threshold_otsu
from skeleton.unitwidthcurveskeleton import getShortestPathskeleton
from skeleton.convOptimize import getSkeletonize3D

root = '/media/pranathi/A336-5F43/'

f = open("/media/pranathi/A336-5F43/skeletonStatsSubVolume.txt", 'w')
strLists = []
volume = np.load('/media/pranathi/A336-5F43/stackUnSmoothedQuadratic.npy')
volume = 255 - volume
volume1 = volume[0:256, 0:256, 0:256]
volume2 = volume[0:128, 0:128, 0:128]
subVolumes = [volume1, volume2]
j = 1
for grey in subVolumes:
    maskOut = copy.deepcopy(grey)
    np.save("/media/pranathi/A336-5F43/grey%i.npy" % j, grey)
    maxip = np.amax(grey, 0)
    imsave('/media/pranathi/A336-5F43/greyImMaxip%i.png' % j, maxip)
    t = threshold_otsu(grey)
    thresh = grey > t
    maskOut[thresh == 1] = 200
    skeleton = getSkeletonize3D(thresh)
    strLists.append("threshold of {} subvolume is {}". format(j, t))
    np.save("/media/pranathi/A336-5F43/thresholded%i.npy" % j, np.uint8(thresh))
    bound = _getBouondariesOfimage(thresh)
    np.save("/media/pranathi/A336-5F43/boundary%i.npy" % j, np.uint8(bound))
    volumeThresh = np.sum(thresh) / thresh.size
    l, count = ndimage.measurements.label(thresh, structure=np.ones((3, 3, 3), dtype=np.uint8))
    maxip = np.amax(thresh, 0)
    imsave('/media/pranathi/A336-5F43/threshImMaxip%i.png' % j, maxip * 255)
    st = time.time()
    sh = time.time()
    strLists.append("skeletonized %i number of pixels in %0.2f seconds" % (np.sum(thresh), sh - st))
    shortest = getShortestPathskeleton(skeleton)
    strLists.append("shortest path skeleton in %i number of pixels in %0.2f seconds" % (np.sum(skeleton), time.time() - sh))
    np.save("/media/pranathi/A336-5F43/thresholded%i.npy" % j, np.uint8(thresh))
    np.save("/media/pranathi/A336-5F43/skeleton%i.npy" % j, np.uint8(skeleton))
    np.save("/media/pranathi/A336-5F43/shortestPathskeleton%i.npy" % j, np.uint8(shortest))
    strLists.append("disjoint objects of {} thresholded subvolume is {}". format(j, count))
    l, count = ndimage.measurements.label(skeleton, structure=np.ones((3, 3, 3), dtype=np.uint8))
    strLists.append("disjoint objects of {} skeletonized subvolume is {}". format(j, count))
    l, count = ndimage.measurements.label(shortest, structure=np.ones((3, 3, 3), dtype=np.uint8))
    strLists.append("disjoint objects of {} shortest path skeleton subvolume is {}". format(j, count))
    maxip = np.amax(skeleton, 0)
    imsave('/media/pranathi/A336-5`F43/skeletonImMaxip%i.png' % j, maxip * 255)
    maxip = np.amax(shortest, 0)
    imsave('/media/pranathi/A336-5F43/shortestSkeletonImMaxip%i.png' % j, maxip * 255)
    os.mkdir(root + 'twodmaskslicesvolume%i' % j); os.mkdir(root + 'twodgreyslicesvolume%i' % j); os.mkdir(root + 'twodthresholdedslicesvolume%i' % j)
    os.mkdir(root + 'twodskeletonslicesvolume%i' % j); os.mkdir(root + 'twodshortestskeletonslicesvolume%i' % j)
    for i in range(maskOut.shape[0]):
        imsave(root + 'twodmaskslicesvolume%i' % j + '/' + 'maskSlice%i.png' % i, maskOut[i])
        imsave(root + 'twodgreyslicesvolume%i' % j + '/' + 'greySlice%i.png' % i, grey[i])
        imsave(root + 'twodthresholdedslicesvolume%i' % j + '/' + 'threshSlice%i.png' % i, thresh[i])
        imsave(root + 'twodskeletonslicesvolume%i' % j + '/' + 'skeletonSlice%i.png' % i, skeleton[i])
        imsave(root + 'twodshortestskeletonslicesvolume%i' % j + '/' + 'shortestPathSlice%i.png' % i, shortest[i])
    j += 1

f.writelines(strLists)
f.close()


# import itertools
# import os
# import time
# import numpy as np
# from scipy.misc import imread
# from scipy import ndimage
# import matplotlib.pyplot as plt
# import matplotlib

# fig = plt.figure()
# ax = fig.add_subplot(111)
# rect1 = matplotlib.patches.Rectangle((1600, 0), 800, 17480, color='white')
# rect2 = matplotlib.patches.Rectangle((4800, 0), 500, 17480, color='white')
# ax.add_patch(rect1)
# ax.add_patch(rect2)
# plt.xlim([0, 8026])
# plt.ylim([17480, 0])
# plt.show()

# verticalBlur = list(range(1600, 2400))
# verticalBlur = verticalBlur + list(range(4800, 5300))

# # valid 3D coordinates inside the brain
# mask = np.load('/media/pranathi/DATA/maskDownsampled.npy')
# aspectRatio = [140, 140]
# listNzi = []; z = []; y = []; x = [];
# for i in range(mask.shape[0]):
#     zoomedOut = ndimage.interpolation.zoom(mask[i], zoom=aspectRatio, order=3, prefilter=False)
#     y_sample, x_sample = np.array(np.where(zoomedOut != 0))
#     y.append(y_sample); x.append(x_sample)
#     # if a index is not in vertical blur i.e 1600 to 2400 and 4800 to 5300 on x axis continue
#     # python - z, y, x z = 799 y = 0 to 17480 x = 0 to 8026
#     valid = [tuple((j, l[0], l[1])) for j in range(0, 20 * (i + 1)) for l in map(list, np.transpose(np.nonzero(zoomedOut))) if l[1] not in verticalBlur]
#     z = z + list(range(0, 20 * (i + 1)))
#     listNzi = listNzi + valid


# def _intersect(a, b):
#     """
#        return the intersection of two lists
#     """
#     if len(set(a) - set(b)) == 0:
#         return 1
#     else:
#         return 0


# # 8 increment corners of a sample
# stepDirect = itertools.product((0, 128), repeat=3)
# listStepDirect = list(stepDirect)

# # if all of the 8 corners of the cube are inside the brain i.e listNzi then it is a valid sample
# # check if i < 8026 / 128 and j < 17480 / 126 and k < 799 / 128


# z_sample, y_sample, x_sample = np.array(np.where(mask != 0))
# z = z_sample * 20; y = y_sample * 140; x = x_sample * 140
# validCoordinates = [tuple((k, j, i)) for i, j, k in zip(x, y, z) if i not in verticalBlur]
# assert len(validCoordinates) < len(z)
# i = 0; j = 0; k = 0; sample = 0;
# while i < 63 and j < 137 and k < 44:
#     if _intersect(listStepDirect, listNzi):
#         sample += 1

# startt = time.time()
# count = 0
# root = input("please enter a root directory where your 2D slices are----")
# formatOfFiles = input("please enter the format of 2D files---")
# # list and sort all the files in the given greyscale
# # input root directory
# # which is to be skeletonized
# listOfJpgs = [os.path.join(root, f) for f in os.listdir(root) if formatOfFiles in f]
# count1 = 0
# inputIm = np.zeros((19, 512, 512), dtype=np.uint8)
# for fileName in subList:
#     inputIm[count1][:][:] = imageExtract
#     count1 += 1
# for i in range(0, 63):
#     r = ndimage.interpolation.zoom(inputIm[:, 0:128, 0: 128 * (i + 1)], [7, 1, 1], order=3, prefilter=False)


# for i in range(0, len(l)):
#     image = imread((os.path.join(root, 'thresh%i.png' %i)))
#     imsave(root + 'rotatedThresh/' + 'binaryIm%i.png' % i, np.rot90(image, 3))

