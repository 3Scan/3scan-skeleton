import numpy as np
from scipy import ndimage
import os
from scipy.misc import imsave, imread

root = '/home/pranathi/mouseBrain-CS/'
formatOfFiles = 'jpg'
listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
listOfJpgs.sort()
arr = np.zeros((856, 605, 779))
mip = np.ones((6050, 7790), dtype=np.uint8) * 255
for index, fileName in enumerate(listOfJpgs):
    image = imread(fileName)
    inds = image < mip  # find where image intensity < min intensity
    mip[inds] = image[inds]  # update the minimum value at each pixel
    if index % 10 == 0:
        interpolatedIm = ndimage.interpolation.zoom(image, [1 / 10, 1 / 10], order=0)
        arr[int(index / 10)] = interpolatedIm
        imsave("downSampledSlice%i.png" % int(index / 10), interpolatedIm)

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
