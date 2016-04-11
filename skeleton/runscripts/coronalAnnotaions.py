import os
from scipy.misc import imread, imsave
import numpy as np
from scipy import ndimage

root = '/media/pranathi/DATA/coronalAnnotations/'
formatOfFiles = 'jpg'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()

# sizes = {}
# for imageName in listOfJpgs:
#     print(imageName)
#     image = imread(imageName)
#     sizes[image.shape] = image.size

# maxVal = max(sizes.values())
# dimensions = list(sizes.keys())
# x = []; y = []
# for item1, item2, item3 in dimensions:
#     x.append(item1)
#     y.append(item2)

klist = [k for k in range(2632, 8026 - 68, 135) if (k > 2420 and k < 4000) or (k > 5500 and k < (7267 + 135))]
maxShape = (8528, 11072, 3)
imArr = np.ones(maxShape, dtype=np.uint8) * 255
m, n, channels = imArr.shape
bigArray = np.zeros((132, 8528, 11072, 3), dtype=np.uint8)
count = 0
root = '/media/pranathi/DATA/coronalAnnotations/coronalAnnotationsBigger/'
formatOfFiles = 'jpg'
listOfJpgsWider = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgsWider.sort()
for imageName in listOfJpgsWider:
    image = imread(imageName)
    # p, q, channels = image.shape
    # lowx = ((m - p) / 2); upx = (((m - p) / 2) + p)
    # lowy = ((n - q) / 2); upy = (((n - q) / 2) + q)
    # imArr[lowx: upx, lowy: upy] = image
    # imsave("/media/pranathi/DATA/coronalAnnotations/coronalAnnotationsBigger/" + imageName.split('/')[5], imArr)
    bigArray[count, :] = image
    count += 1

root = '/media/pranathi/DATA/coronalAnnotations/coronalAnnotationsBigger/'
formatOfFiles = 'jpg'
listOfJpgsWider = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgsWider.sort()
klist = [2632]
for k in klist:
    count = 0
    transverseSlice = np.zeros((132, 89, 3), dtype=np.uint8)
    for imageName in listOfJpgsWider:
        image = imread(imageName)
        transverseSlice[count, :] = ndimage.interpolation.zoom(image[k, :, :], [1 / 125, 1], order=0)
        count += 1
    imsave("transverseSlice%i.png" % k, transverseSlice)
