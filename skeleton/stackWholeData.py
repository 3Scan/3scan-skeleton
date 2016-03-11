import time

import os

import numpy as np

from scipy import ndimage

from scipy.misc import imread


def process(listOfJpgs):
    min_ = np.ones((12000, 3000)) * 255
    for fileName in listOfJpgs[0]:
        print(fileName)
        # read the image
        image = imread(fileName)
        # remove the black pixels and extract 12000 x 3000 tissue region from 12000 * 4096 images
        imageExtract = np.zeros((12000, 3000), dtype=int)
        # remove the black pixels on left dure to absence of tissue (tissue already cut)
        imageExtract[:, 0:199] = image[:, 201:400]
        # remove the black pixels on the right beacuse light is focused only in the center and left
        # to pixels on the line scan camera
        imageExtract[:, 200:] = image[:, 400: 3200]
        min_ = np.minimum(min_, imageExtract)
    print(np.sum(min_))
    return min_


def stackVolume():
    startt = time.time()
    root = input("please enter a root directory where your 2D slices are ----")  # enter directory of entire KESM data set is
    beginningFolder = input("please enter the starting folder----")  # enter the starting folder index ex:0000
    forMat = input("enter the format of the files ----")  # entire the format of image data set
    numberOfFolders = input("enter number of folders or cuts----")  # enter the number of cuts per each face of the image
    listOfJpgsCutwise = []  # initialize the list of Jpgs collection
    # for each of the cuts on the face of block of the resin under KESM
    for i in range(0, int(numberOfFolders)):
        directoryCut = root + beginningFolder + str(i) + '/'   # Get the ith cut
        listOffiles = os.listdir(directoryCut)  # list the files in the cut
        count = 0
        listOfJpgs = []  # initialize variable to calculate jpg images in each cut
        # store the file names ending with entered format
        for file in listOffiles:
            if file.endswith(forMat):
                listOfJpgs.append(file)
                count += 1
        listOfJpgsCutwise.append(listOfJpgs)
        # print jpgs per each cut read
        print("files in %i th folder is %i" % (i, count))

    # take in 1 sample image to know x, y dimensions of an image
    directoryCut = root + beginningFolder + str(0) + '/'
    image = imread((os.path.join(directoryCut, listOfJpgsCutwise[0][0])))
    m, n = np.shape(image)  # m = 12000, n = 4096

    # starting indices for cut - obtained by subtracting the maximum and minimum z dimension in a subdirectory/ cut folder
    # and multiply with 1000
    startingIndex = [9230, 9234, 9013, 9018, 0, 5]
    """allocating an array by default all the elements are one, since with no absorption by
    the tissue optics in the KESM would transmit all the light to the line sensor
    3000 is a different number than n- it removes the black patch beneath the knife edge
    where light is not transmitted at all in the image"""
    # 0.005 = 1 / 200 resampling factor = picking 1 for every 200 rows or columns
    resFactor = 0.005
    allocDimz = 9655
    allocDimy = resFactor * (m)
    allocDimx = resFactor * (3000)
    totalArray = np.zeros((allocDimz, allocDimy, int(numberOfFolders) * allocDimx), dtype=int)

    # go through each of the files and look at the z dimension and figure out which slice it belongs
    # for each of the cuts
    for i in range(0, int(numberOfFolders)):
        print("in cut", i)
        zdimlist = []
        inputIm = np.ones((allocDimz, allocDimy, allocDimx), dtype=int)
        # for each of the files in the cut
        for presentIndex, jpgFilename in enumerate(listOfJpgsCutwise[i]):
            # enumerate all the elements in the string and collect the z dimension
            for index, items in enumerate(jpgFilename):
                if items == 'z':
                    # strip of the underscore from the string in z0.0080_
                    # and get a floating point number of where the slice is
                    zdim = float(jpgFilename[index + 1].strip('_'))
                    # store the floating point reprsentation of z dimension from the filename
                    zdimlist.append(zdim)
            directoryCut = root + beginningFolder + str(i) + '/'
            # read the image
            image = imread((os.path.join(directoryCut, jpgFilename)))
            # remove the black pixels and extract 12000 x 3000 tissue region from 12000 * 4096 images
            imageExtract = np.zeros((m, 3000), dtype=int)
            # remove the black pixels on left dure to absence of tissue (tissue already cut)
            imageExtract[:, 0:199] = image[:, 201:400]
            # remove the black pixels on the right beacuse light is focused only in the center and left
            # to pixels on the line scan camera
            imageExtract[:, 200:] = image[:, 400: 3200]
            # skip every 200 rows and 200 columns and downsample the 12000 x 3000 to 60 x 15
            downSampledimage = ndimage.interpolation.zoom(imageExtract, zoom=resFactor, order=0)
            # shape of the downsampled image
            x, y = downSampledimage.shape
            # delete the image and imageExtract to save memory
            del image; del imageExtract;
            # if it is not the first file in the cut displace it by the starting index of the z file for the cut
            # or subdirectory - (startingIndex[i]) and copy the downSampledImage at the displaced index
            if presentIndex != 0:
                inputIm[(startingIndex[i] + (1000 * (zdim - zdimlist[presentIndex - 1]))), :, :] = downSampledimage
            else:
                # if it is the first file then copy the downsampled index at the starting index of the
                # total array for tthe subdirectory
                inputIm[startingIndex[i], :, :] = downSampledimage
            # del the downsampledimage after copying to save memory
            del downSampledimage
        # copy or place all the nth sub-directory/cut images at the nth offset of the total volume/cube of dataset - totalArray
        totalArray[:, :, (i * y): (i + 1) * y] = inputIm
        # delete the nth subdirectory/cut images after copying
        del inputIm
    # Save the whole volume as a numpy array
    np.save("wholeBrainStacked.npy", totalArray)
    # print the time taken to put back the entire volume
    print("time taken to put back the whole volume is", time.time() - startt)


def getProjection():
    root = input("please enter a root directory where your 2D slices are ----")  # enter directory of entire KESM data set is
    beginningFolder = input("please enter the starting folder----")  # enter the starting folder index ex:0000
    startt = time.time()
    forMat = input("enter the format of the files ----")  # entire the format of image data set
    numberOfFolders = input("enter number of folders or cuts----")  # enter the number of cuts per each face of the image
    listOfJpgsCutwise = []  # initialize the list of Jpgs collection
    # for each of the cuts on the face of block of the resin under KESM
    for i in range(0, int(numberOfFolders)):
        directoryCut = root + beginningFolder + str(i) + '/'   # Get the ith cut
        listOffiles = os.listdir(directoryCut)  # list the files in the cut
        count = 0
        listOfJpgs = []  # initialize variable to calculate jpg images in each cut
        # store the file names ending with entered format
        for file in listOffiles:
            if file.endswith(forMat):
                listOfJpgs.append((os.path.join(directoryCut, file)))
                count += 1
        listOfJpgsCutwise.append(listOfJpgs)
        # print jpgs per each cut read
        print("files in %i th folder is %i" % (i, count))
    # take in 1 sample image to know x, y dimensions of an image
    directoryCut = root + beginningFolder + str(0) + '/'
    image = imread((os.path.join(directoryCut, listOfJpgsCutwise[0][0])))
    m, n = np.shape(image)  # m = 12000, n = 4096
    print(listOfJpgsCutwise)
    totalArraymip = np.zeros((m, int(numberOfFolders) * 3000), dtype=int)
    listOfMips = []; p = Pool(8)
    mip = np.ones((m, 3000), dtype=int) * 255
    for cut in range(0, int(numberOfFolders)):
        print("in cut", cut)
        file_chunks = []
        chunksize = 500
        file_chunks = [listOfJpgsCutwise[i:i + chunksize]
        for i in range(0, len(listOfJpgsCutwise[cut]), chunksize):
            ranges = range(0, len(listOfJpgsCutwise[cut]), chunksize)

        for chunk_idx in ranges:
            file_chunks.append(listOfJpgsCutwise[cut][chunk_idx:chunk_idx + chunksize])

        #  find the minimum x,y vals in chunks of 100
        first = True
        print("file chunks are", file_chunks)
        min_arrays = p.map(process, file_chunks)

        #  Find the minimum from the minimums returned from each process
        for array in min_arrays:
            if first:
                mini = array
                first = False
            mip = np.minimum(array, mini)
        listOfMips.append(mip)
        # copy or place all the nth sub-directory/cut images mip at the nth offset of totalArray
        totalArraymip[:, (i * 3000): (i + 1) * 3000] = mip
        # reallocate mip
        mip = np.ones((m, 3000), dtype=int) * 255
    print("time taken to find mip of the whole volume is", time.time() - startt)
    # Save the whole volume as a numpy array
    np.save("wholeBrainMip.npy", totalArraymip)
    return listOfMips, totalArraymip


def upsample():
    pass


if __name__ == '__main__':
    from multiprocessing import Pool
    listOfMips, totalArraymip = getProjection()
    #  p = Pool(8)

    # chunksize = 1500  # 12000/8 = 1500, might have less overhead
    # for cuts in range(0, int(numberOfFolders)):
    #     inputIm = np.zeros((m, 3000), dtype=int)
    #     listOfJpgs = listOfJpgsCutwise(cuts)
    #     chunks = [listOfJpgs[i:i + chunksize]
    #               for i in range(0, len(listOfJpgs), chunksize)]

    #     # this returns an array of (len(files)/chunksize, 4000, 4000)
    #     min_arrays = np.array(p.map(process, chunks))
    #     mini = np.amin(min_arrays, axis=0)
    # stackVolume()

