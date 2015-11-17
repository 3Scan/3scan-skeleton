#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from __future__ import division
import numpy as np
from scipy.ndimage.filters import convolve
cimport numpy as np
DTYPE = np.int64


lookUpTablearray = np.load('lookupTablearray.npy')


def _applySubiter(np.ndarray[np.int8_t, ndim=3] image, np.ndarray[np.int64_t, ndim=3] convImage):
    cdef int zPad, mPad, nPad, numpixel_removed
    cdef int zdim, xdim, ydim
    zPad = image.shape[0]
    mPad = image.shape[1]
    nPad = image.shape[2]
    cdef np.ndarray temp_del = np.zeros([zPad, mPad, nPad], dtype='i1')
    for zdim in range(1, zPad - 1):
        for xdim in range(1, mPad - 1):
            for ydim in range(1, nPad - 1):
                temp_del[zdim, ydim, xdim] = lookUpTablearray[convImage[zdim, ydim, xdim]]
    numpixel_removed = np.sum(image*temp_del)
    image[temp_del == 1] = 0
    return numpixel_removed, image


def __convolveImage(np.ndarray[np.int8_t, ndim=3] arr, np.ndarray[np.int64_t, ndim=3] flippedKernel):
    cdef np.ndarray[np.int64_t, ndim=3] result
    result = convolve(arr, flippedKernel, mode='constant', cval=0)
    result[arr == 0] = 0
    return result


def _skeletonPass(np.ndarray[np.int8_t, ndim=3] image):
    cdef np.ndarray[np.int64_t, ndim=3] firstSubiteration, secondSubiteration, thirdSubiteration, fourthSubiteration, fifthSubiteration, sixthSubiteration, seventhSubiteration, eighthSubiteration, ninthSubiteration, tenthSubiteration, eleventhSubiteration, twelvethSubiteration
    firstSubiteration = np.array([[[2 ** 0, 2 ** 1, 2 ** 2],
                                [2 ** 3, 2 ** 4, 2 ** 5],
                                [2 ** 6, 2 ** 7, 2 ** 8]],
                                [[2 ** 9, 2 ** 10, 2 ** 11],
                                [2 ** 12, 0, 2 ** 13],
                                [2 ** 14, 2 ** 15, 2 ** 16]],
                                [[2 ** 17, 2 ** 18, 2 ** 19],
                                [2 ** 20, 2 ** 21, 2 ** 22],
                                [2 ** 23, 2 ** 24, 2 ** 25]]], dtype='i8')

    secondSubiteration = np.array([[[1, 8, 64],
            [2, 16, 128],
            [ 4, 32, 256]],
           [[ 512, 4096, 16384],
            [ 1024, 0, 32768],
            [ 2048, 8192, 65536]],
           [[131072, 1048576, 8388608],
            [262144, 2097152, 16777216],
            [524288, 4194304, 33554432]]], dtype='i8')


    thirdSubiteration = np.array([[[      64,        8,        1],
            [   16384,     4096,      512],
            [ 8388608,  1048576,   131072]],

           [[     128,       16,        2],
            [   32768,        0,     1024],
            [16777216,  2097152,   262144]],

           [[     256,       32,        4],
            [   65536,     8192,     2048],
            [33554432,  4194304,   524288]]], dtype='i8')

    fourthSubiteration = np.array([[[  524288,  4194304, 33554432],
            [  262144,  2097152, 16777216],
            [  131072,  1048576,  8388608]],

           [[    2048,     8192,    65536],
            [    1024,        0,    32768],
            [     512,     4096,    16384]],

           [[       4,       32,      256],
            [       2,       16,      128],
            [       1,        8,       64]]], dtype='i8')


    fifthSubiteration = np.array([[[ 8388608,    16384,       64],
            [ 1048576,     4096,        8],
            [  131072,      512,        1]],

           [[16777216,    32768,      128],
            [ 2097152,        0,       16],
            [  262144,     1024,        2]],

           [[33554432,    65536,      256],
            [ 4194304,     8192,       32],
            [  524288,     2048,        4]]], dtype='i8')


    sixthSubiteration = np.array([[[       1,        2,        4],
            [     512,     1024,     2048],
            [  131072,   262144,   524288]],

           [[       8,       16,       32],
            [    4096,        0,     8192],
            [ 1048576,  2097152,  4194304]],

           [[      64,      128,      256],
            [   16384,    32768,    65536],
            [ 8388608, 16777216, 33554432]]], dtype='i8')


    seventhSubiteration = np.array([[[ 8388608,  1048576,   131072],
            [16777216,  2097152,   262144],
            [33554432,  4194304,   524288]],

           [[   16384,     4096,      512],
            [   32768,        0,     1024],
            [   65536,     8192,     2048]],

           [[      64,        8,        1],
            [     128,       16,        2],
            [     256,       32,        4]]], dtype='i8')

    eighthSubiteration = np.array([[[      64,      128,      256],
            [       8,       16,       32],
            [       1,        2,        4]],

           [[   16384,    32768,    65536],
            [    4096,        0,     8192],
            [     512,     1024,     2048]],

           [[ 8388608, 16777216, 33554432],
            [ 1048576,  2097152,  4194304],
            [  131072,   262144,   524288]]], dtype='i8')


    ninthSubiteration = np.array([[[       4,       32,      256],
            [    2048,     8192,    65536],
            [  524288,  4194304, 33554432]],

           [[       2,       16,      128],
            [    1024,        0,    32768],
            [  262144,  2097152, 16777216]],

           [[       1,        8,       64],
            [     512,     4096,    16384],
            [  131072,  1048576,  8388608]]], dtype='i8')


    tenthSubiteration = np.array([[[     256,       32,        4],
            [     128,       16,        2],
            [      64,        8,        1]],

           [[   65536,     8192,     2048],
            [   32768,        0,     1024],
            [   16384,     4096,      512]],

           [[33554432,  4194304,   524288],
            [16777216,  2097152,   262144],
            [ 8388608,  1048576,   131072]]], dtype='i8')


    eleventhSubiteration = np.array([[[     256,    65536, 33554432],
            [      32,     8192,  4194304],
            [       4,     2048,   524288]],

           [[     128,    32768, 16777216],
            [      16,        0,  2097152],
            [       2,     1024,   262144]],

           [[      64,    16384,  8388608],
            [       8,     4096,  1048576],
            [       1,      512,   131072]]], dtype='i8')

    twelvethSubiteration = np.array([[[  131072,   262144,   524288],
            [ 1048576,  2097152,  4194304],
            [ 8388608, 16777216, 33554432]],

           [[     512,     1024,     2048],
            [    4096,        0,     8192],
            [   16384,    32768,    65536]],

           [[       1,        2,        4],
            [       8,       16,       32],
            [      64,      128,      256]]], dtype='i8')
    cdef numPixelsremovedList = []
    cdef int totalPixels
    # cdef np.ndarray[np.int64_t, ndim=3] convImage
    print(firstSubiteration)
    print(firstSubiteration.dtype)
    convImage = __convolveImage(image, firstSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, secondSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, thirdSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, fourthSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, fifthSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, sixthSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, seventhSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, eighthSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, ninthSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, tenthSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, eleventhSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    convImage = __convolveImage(image, twelvethSubiteration)
    totalPixels, image = _applySubiter(image, convImage)
    numPixelsremovedList.append(totalPixels)
    numPixelsremoved = np.sum(numPixelsremovedList)
    return numPixelsremoved, image

def _getSkeletonize3D(np.ndarray[np.int8_t, ndim=3] image):
    cdef int numpixel_removed, pass_no
    cdef int zOrig, xOrig, yOrig
    zOrig = image.shape[0]
    xOrig = image.shape[1]
    yOrig = image.shape[2]
    cdef nslices = image.shape[0]+2, nrows = image.shape[1]+2, ncols = image.shape[2]+2
    cdef np.ndarray padImage = np.zeros([nslices, nrows, ncols], dtype='i1')
    padImage[1:nslices-1, 1:nrows-1, 1:ncols-1] = image > 0
    pass_no = 0
    numpixel_removed = 0
    while pass_no == 0 or numpixel_removed > 0:
        numpixel_removed, padImage = _skeletonPass(padImage)
        pass_no += 1
    return padImage[1: zOrig + 1, 1: xOrig + 1, 1: yOrig + 1]
