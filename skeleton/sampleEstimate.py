import os
import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage.filters import threshold_otsu
from skeleton.convOptimize import getSkeletonize3D
from skeleton.unitwidthcurveskeleleton import getShortestPathskeleton

# valid 3D coordinates inside the brain
mask = np.load('/media/pranathi/DATA/maskDownsampled.npy')
isum = 0; iskpy = iskpx = 560; iskpz = int(0.5 + iskpx * 0.7 / 5.0)
root = input("please enter a root directory where your 2D slices are----")
formatOfFiles = input("please enter the format of 2D files---")
# list and sort all the files in the given greyscale
# input root directory
# which is to be skeletonized
listOfJpgs = [os.path.join(root, f) for f in os.listdir(root) if formatOfFiles in f]
listOfJpgs.sort()
for i in range(9, 799 - 10, iskpz):
    print(i)
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    subList = listOfJpgs[i - 9: i + 9]
    count = 0
    for fileName in subList:
        subVolume[count][:][:] = imread(fileName)
        count += 1
    izd = int(i / 20)
    for j in range(64, 17480 - 64, iskpy):
        iyd = int(j / 140)
        for k in range(64, 8026 - 64, iskpx):
            ixd = int(k / 140)
            if k < 1600 or (k > 2400 and k < 4800) or k > 5300:
                if mask[izd, iyd, ixd]:
                    isum = isum + 1
                    subSubvolume = subVolume[:, k - 67:k + 67, j - 67: j + 67]
                    subSubvolume = 255 - subSubvolume
                    np.save("/media/pranathi/DATA/subSubvolumeVolumes/subVolume_{}_{}_{}.npy".format(i, j, k), np.uint8(subVolume))
                    interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [7, 1, 1], order=2, prefilter=False)
                    t = threshold_otsu(subSubvolume)
                    thresh = subSubvolume > t
                    thinned = getSkeletonize3D(thresh)
                    skeleton = getShortestPathskeleton(thinned)
                    np.save("/media/pranathi/DATA/subsubVolumeThinned/skeleton_{}_{}_{}.npy".format(i, j, k), np.uint8(skeleton))
                    np.save("/media/pranathi/DATA/subsubVolumeThinned/thinned_{}_{}_{}.npy".format(i, j, k), np.uint8(thinned))
                    np.save("/media/pranathi/DATA/subsubVolumeThresholds/thresh_{}_{}_{}.npy".format(i, j, k), np.uint8(thresh))

print(isum)


