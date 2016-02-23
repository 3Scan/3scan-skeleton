import os
import time
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage.filters import threshold_otsu
from skeleton.convOptimize import getSkeletonize3D
from skeleton.unitwidthcurveskeleton import getShortestPathskeleton

# os.mkdir("/media/pranathi/User Data/subsubVolumeThinned/")
# os.mkdir("/media/pranathi/User Data/subsubVolumeThresholds/")
# os.mkdir("/media/pranathi/User Data/subsubVolumeSkeletons/")
# valid 3D coordinates inside the brain
mask = np.load('/media/pranathi/A336-5F43/maskDownsampled.npy')
isum = 0; iskpy = iskpx = 560; iskpz = int(0.5 + iskpx * 0.7 / 5.0)
root = '/media/pranathi/A336-5F43/ii-5016-15-ms-brain_1920/filt/'
formatOfFiles = 'png'
f = open("/media/pranathi/A336-5F43/StatsSubVolume.txt", 'w')
strLists = []
# list and sort all the files in the given greyscale
# input root directory
# which is to be skeletonized
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
for i in range(9, 799 - 10, iskpz):
    print(i)
    izd = int(i / 20)
    for j in range(64, 17480 - 64, iskpy):
        iyd = int(j / 140)
        for k in range(64, 8026 - 64, iskpx):
            ixd = int(k / 140)
            if k < 1600 or (k > 2400 and k < 4800) or k > 5300:
                if mask[izd, iyd, ixd]:
                    if j >= 67 and k >= 67:
                        subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
                        subList = listOfJpgs[i - 9: i + 9]
                        count = 0
                        for fileName in subList:
                            subVolume[count][:][:] = imread(fileName)
                            count += 1
                        isum = isum + 1
                        subSubvolume = subVolume[:, j - 67:j + 67, k - 67: k + 67]
                        interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7, 1, 1], order=2, prefilter=False)
                        interpolatedIm = 255 - interpolatedIm
                        t = threshold_otsu(interpolatedIm)
                        thresh = interpolatedIm > t
                        count = ndimage.measurements.label(thresh, structure=np.ones((3, 3, 3), dtype=np.uint8))
                        st = time.time()
                        thinned = getSkeletonize3D(thresh)
                        sh = time.time()
                        strLists.append("skeletonized %i number of pixels in %0.2f seconds" % (thresh.sum(), sh - st))
                        skeleton = getShortestPathskeleton(thinned)
                        strLists.append("shortest path skeleton in %i number of pixels in %0.2f seconds" % (skeleton.sum(), time.time() - sh))
                        np.save("/media/pranathi/User Data/subsubVolumeSkeletons/skeleton_{}_{}_{}.npy".format(i, j, k), skeleton)
                        np.save("/media/pranathi/User Data/subsubVolumeThinned/thinned_{}_{}_{}.npy".format(i, j, k), thinned)
                        np.save("/media/pranathi/User Data/subsubVolumeThresholds/thresh_{}_{}_{}.npy".format(i, j, k), thresh)
                        thinned = getSkeletonize3D(thresh)
                        strLists.append("disjoint objects of {} thresholded subvolume is {}". format(isum, count))
                        count = ndimage.measurements.label(thinned, structure=np.ones((3, 3, 3), dtype=np.uint8))
                        strLists.append("disjoint objects of {} skeletonized subvolume is {}". format(isum, count))
                        count = ndimage.measurements.label(skeleton, structure=np.ones((3, 3, 3), dtype=np.uint8))
                        strLists.append("disjoint objects of {} shortest path skeleton subvolume is {}". format(isum, count))


f.writelines(strLists)
f.close()
print(isum)
