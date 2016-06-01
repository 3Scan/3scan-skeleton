import numpy as np
from scipy import ndimage
import os
from scipy.misc import imsave, imread
# import cv2

root = '/home/pranathi/mouseBrain-CS/'
formatOfFiles = 'jpg'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
arr = np.zeros((856, 605, 779))
mip = np.ones((6050, 7790)) * 128
for index, fileName in enumerate(listOfJpgs):
    image = imread(fileName)
    inds = image < mip  # find where image intensity < min intensity
    mip[inds] = image[inds]  # update the minimum value at each pixel
    if index % 5 == 0:
        interpolatedIm = ndimage.interpolation.zoom(image, [1 / 7, 1 / 7], order=0)
        arr[int(index / 5)] = interpolatedIm
        imsave("downSampledSlice%i.png" % int(index / 5), interpolatedIm)

np.save("mipCoronal.npy", mip)
np.save("downSampledArrayC.npy", arr)


def imgNorm(img):
    """
    return image with a float64 datatype, scaled between [0, 1] inclusive
    This does NOT scale the max and min values to match 0 and 1, it only fits a 0-255 image into 0-1
    """
    ii = np.iinfo(img.dtype)
    return (1 / (ii.max - ii.min)) * (img - ii.min)


def normalizeStripes(img):
    """
    remove vertical and horizontal stripes from image by normalizing out the row and column medians
    Original method borrowed from TAMU

    dtype: return image array as dtype. default is the same dtype as input array, which will often be a np.uint8
    There are circumstances in which we want to return the np.float64 dtype, such as if we are doing a global
    levels adjustment on the histograms of all images.
    """
    source = imgNorm(img)

    axis_0_median = np.median(source, axis=0)
    axis_1_median = np.median(source, axis=1)

    light = axis_1_median[:, np.newaxis] * axis_0_median[np.newaxis, :]

    # the reason for the 4.0 is still unclear
    # it is likely the square of the max value possible in the median outer product
    clean_img = (source / (light * 4.0)) * 255.
    clean_img = clean_img.clip(0, 255).round().astype(np.uint8)
    # if dtype is None or dtype == np.uint8:  # force default dtype to uint8
    # elif dtype == np.float64:
    #     # we should be normalizing here to return between 0 and 1... but we dont a priori know the max value from the _imgnorm operation
    #     clean_img = clean_img.astype(np.float64)
    # else:
    #     clean_img = clean_img.astype(dtype)

    # if we return floats here, it kills the Spark serializer with files that are bigger than 4GB
    return clean_img

image = imread("/media/pranathi/DATA/mouseBrain-CS/00251.jpg")
result = normalizeStripes(image)
# edges = cv2.Canny(image, 50, 150, apertureSize=3)

# lines = cv2.HoughLines(edges, 1, (30 * np.pi) / 180, 200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('houghlines3.jpg',image)
# transverseSlice = imread("transverseSlice2767.png")
# for coords in validCenters:
#     transverseSlice[coords[1] - 1: coords[1] + 2, int((coords[2] + 6) / 7) - 1: int((coords[2] + 6) / 7) + 2] = 1

root = '/home/pranathi/dsby7-CS/'
formatOfFiles = 'png'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
count = 0
saggitalSlicearr = np.zeros((112, 864, 1725), dtype=np.uint8)
for index in range(0, len(listOfJpgs)):
    for countSlice, value in enumerate(list(range(0, 1113, 10))):
        image = imread(root + 'downSampledSlice%i.png' % index)
        saggitalSlicearr[countSlice, :, index] = image[:, value]
for i in range(0, saggitalSlicearr.shape[0]):
    imsave(root + "saggitalSlices/" + "saggitalSlice%i.jpg" % i, saggitalSlicearr[i])

maskArtVein = np.load('/home/pranathi/maskArtVein.npy')
root = '/home/pranathi/subsubVolumethresholdNew_28/'
formatOfFiles = 'npy'
listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfNpys.sort()
transverseSlice = imread("transverseSlice2767.png")
for npyIndex in listOfNpys:
    npyIndex = npyIndex.replace('.npy', '')
    strs = npyIndex.split('/')[-1].split('_')
    i, j, k = [int(s) for s in strs if s.isdigit()]
    i = i - 160
    if maskArtVein[(int((i + 9) / 10.0), int((j + 6) / 7.0), int((k + 6) / 7.0))] != 255:
        transverseSlice[i - 1: i + 2, int((j + 6) / 7) - 1: int((j + 6) / 7) + 2] = 1
imsave("transverseSliceSamplePoints2767.png", transverseSlice)
