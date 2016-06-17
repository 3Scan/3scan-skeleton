import os
from scipy.misc import imread, imsave
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

root = '/media/pranathi/KINGSTON/coronalAnnotationsBigger/'
formatOfFiles = 'jpg'
listOfJpgsWider = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgsWider.sort()
klist = [k - 420 for k in range(2632, 8026 - 68, 135) if (k > 2420 and k < 4000) or (k > 5500 and k < (6952 + 135))]
transverseSlicearr = np.zeros((len(klist), 132, 89, 3), dtype=np.uint8)
for index, imageName in enumerate(listOfJpgsWider):
    for countSlice, value in enumerate(klist):
        image = np.rot90(imread(imageName), 2)
        transverseSlicearr[countSlice, index, :, :] = ndimage.interpolation.zoom(image[countSlice, :, :], [1 / 125, 1], order=0)
for i in range(0, transverseSlicearr.shape[0]):
    imsave("transverseSlice%i.png" % i, transverseSlicearr[i])

root = '/home/pranathi/transverseSliceAnnotations'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
for imageName in listOfJpgs:
    image = imread(imageName)
    image = np.rot90(image, 3)
    imsave(imageName, image)

root = '/media/pranathi/DATA/PNGS/results_ts_4/'
formatOfFiles = 'png'
listOfJpgsWider = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgsWider.sort()
listOfJpgsWider = listOfJpgsWider[:-3]
root = '/media/pranathi/KINGSTON/brain_marked_th/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
listOfJpgs = listOfJpgs[:-2]
for imageName, imageName2 in zip(listOfJpgsWider, listOfJpgs):
    i = imread(imageName)
    (wsize, baseheight) = i.shape
    img = imread(imageName2)
    # img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
    plt.subplot(2, 1, 1)
    plt.imshow(i)
    plt.subplot(2, 1, 2)
    plt.imshow(img)
for k in klist:
    i[k, :, :] = 0
    plt.imshow(i)
    plt.show()

root = '/media/pranathi/DATA/transverseSlicesAnnotations/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
klist = [k for k in range(2632, 8026 - 68, 135) if (k > 2420 and k < 4000) or (k > 5500 and k < (6952 + 135))]
for index, k in enumerate(klist[:-3]):
    i = imread("/media/pranathi/DATA/results_april19/transverseSlicepercentVasc%i.png" % k)
    plt.subplot(2, 1, 1)
    plt.imshow(i, cmap='gray')
    plt.subplot(2, 1, 2)
    i2 = imread("/media/pranathi/DATA/Pictures_ts_depFriday/annotate/transverseSliceAnnotate%i.png" % klist[index])
    # iarr[index, :] = i2
    plt.imshow(i2)
    plt.tight_layout()
    plt.savefig("mosaic%i.png" % index)

    # ipr = np.rot90(i[:, 0:44, :], 3)
    # im = ndimage.interpolation.zoom(ipr, [79 / 44, 246 / 132, 1], order=0)
    # imsave(imageName, im)


