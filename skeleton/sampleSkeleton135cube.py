import os
import time
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage.filters import threshold_otsu
from skeleton.convOptimize import getSkeletonize3D
from skeleton.unitwidthcurveskeleton import getShortestPathskeleton


mask = np.load('/media/pranathi/DATA/maskDownsampled.npy')
isum = 0; iskpy = iskpx = 560; iskpz = int(0.5 + iskpx * 0.7 / 5.0)
root = '/media/pranathi/DATA/ii-5016-15-ms-brain_1920/filt/'
formatOfFiles = 'png'
f = open("/media/pranathi/DATA/StatsSubVolume.txt", 'w')
strLists = []
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
for i in range(9, 799 - 10, iskpz):
    print(i)
    subList = listOfJpgs[i - 9: i + 10]
    count = 0
    for fileName in subList:
        subVolume[count][:][:] = imread(fileName)
        count += 1
    assert count == subVolume.shape[0]
    for j in range(67, 17480 - 68, iskpy):
        for k in range(67, 8026 - 68, iskpx):
            if k < 1600 or (k > 2350 and k < 4800) or (k > 5520 and k < 8020):
                izd = int(i / 20); ixd = int(k / 140); iyd = int(j / 140)
                if mask[izd, iyd, ixd]:
                    isum = isum + 1
                    subSubvolume = 255 - subVolume[:, j - 67:j + 68, k - 67: k + 68]
                    interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
                    t = threshold_otsu(interpolatedIm)
                    interpolatedIm = interpolatedIm > t
                    st = time.time()
                    thinned = getSkeletonize3D(interpolatedIm)
                    sh = time.time()
                    strLists.append("skeletonized %i number of pixels in %0.2f seconds" % (interpolatedIm.sum(), sh - st))
                    skeleton = getShortestPathskeleton(thinned)
                    strLists.append("shortest path skeleton in %i number of pixels in %0.2f seconds" % (thinned.sum(), time.time() - sh))
                    np.save("/media/pranathi/DATA/subsubVolumeSkeletons/skeleton_{}_{}_{}.npy".format(i, j, k), skeleton)
                    np.save("/media/pranathi/DATA/subsubVolumeThresholds/threshold_{}_{}_{}.npy".format(i, j, k), interpolatedIm)
                    l, count = ndimage.measurements.label(skeleton, structure=np.ones((3, 3, 3), dtype=np.uint8))
                    strLists.append("disjoint objects of {} shortest path skeleton subvolume is {}". format(isum, count))
f.writelines(strLists)
f.close()
print(isum)
