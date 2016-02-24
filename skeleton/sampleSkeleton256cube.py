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
subVolume = np.zeros((36, 17480, 8026), dtype=np.uint8)
for i in range(18, 799 - 19, iskpz):
    print(i)
    subList = listOfJpgs[i - 18: i + 18]
    count = 0
    for fileName in subList:
        subVolume[count][:][:] = imread(fileName)
        count += 1
    assert count == subVolume.shape[0]
    izd = int(i / 20)
    for j in range(128, 17480 - 129, iskpy):
        iyd = int(j / 140)
        for k in range(128, 8026 - 129, iskpx):
            ixd = int(k / 140)
            if k < 1600 or (k > 2350 and k < 4800) or (k > 5520 and k < 8020):
                if mask[izd, iyd, ixd]:
                    isum = isum + 1
                    subSubvolume = subVolume[:, j - 128:j + 129, k - 128: k + 129]
                    interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.703125, 1, 1], order=2, prefilter=False)
                    interpolatedIm = 255 - interpolatedIm
                    t = threshold_otsu(interpolatedIm)
                    thresh = interpolatedIm > t
                    st = time.time()
                    thinned = getSkeletonize3D(thresh)
                    sh = time.time()
                    strLists.append("skeletonized %i number of pixels in %0.2f seconds" % (thresh.sum(), sh - st))
                    skeleton = getShortestPathskeleton(thinned)
                    strLists.append("shortest path skeleton in %i number of pixels in %0.2f seconds" % (thinned.sum(), time.time() - sh))
                    np.save("/media/pranathi/User Data/subsubVolumeSkeletons/skeleton_{}_{}_{}.npy".format(i, j, k), skeleton)
                    np.save("/media/pranathi/User Data/subsubVolumeThresholds/threshold_{}_{}_{}.npy".format(i, j, k), thresh)
                    l, count = ndimage.measurements.label(skeleton, structure=np.ones((3, 3, 3), dtype=np.uint8))
                    strLists.append("disjoint objects of {} shortest path skeleton subvolume is {}". format(isum, count))
f.writelines(strLists)
f.close()
print(isum)
