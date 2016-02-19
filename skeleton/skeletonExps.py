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
    strLists.append("threshold of {} subvolume is {}". format(j, t))
    thresh = grey > t
    maskOut[thresh == 1] = 200
    np.save("/media/pranathi/A336-5F43/thresholded%i.npy" % j, np.uint8(thresh))
    bound = _getBouondariesOfimage(thresh)
    np.save("/media/pranathi/A336-5F43/boundary%i.npy" % j, np.uint8(bound))
    volumeThresh = np.sum(thresh) / thresh.size
    count = ndimage.measurements.label(thresh, structure=np.ones((3, 3, 3), dtype=np.uint8))
    maxip = np.amax(thresh, 0)
    imsave('/media/pranathi/A336-5F43/threshImMaxip%i.png' % j, maxip * 255)
    st = time.time()
    skeleton = getSkeletonize3D(thresh)
    sh = time.time()
    strLists.append("skeletonized %i number of pixels in %0.2f seconds" % (np.sum(thresh), sh - st))
    shortest = getShortestPathskeleton(skeleton)
    strLists.append("shortest path skeleton in %i number of pixels in %0.2f seconds" % (np.sum(skeleton), time.time() - sh))
    np.save("/media/pranathi/A336-5F43/thresholded%i.npy" % j, np.uint8(thresh))
    np.save("/media/pranathi/A336-5F43/skeleton%i.npy" % j, np.uint8(skeleton))
    np.save("/media/pranathi/A336-5F43/shortestPathskeleton%i.npy" % j, np.uint8(shortest))
    strLists.append("disjoint objects of {} thresholded subvolume is {}". format(j, count))
    count = ndimage.measurements.label(skeleton, structure=np.ones((3, 3, 3), dtype=np.uint8))
    strLists.append("threshold of {} skeleton subvolume is {}". format(j, count))
    count = ndimage.measurements.label(shortest, structure=np.ones((3, 3, 3), dtype=np.uint8))
    strLists.append("threshold of {} shortest path skeleton subvolume is {}". format(j, count))
    maxip = np.amax(skeleton, 0)
    imsave('/media/pranathi/A336-5F43/skeletonImMaxip%i.png' % j, maxip * 255)
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
