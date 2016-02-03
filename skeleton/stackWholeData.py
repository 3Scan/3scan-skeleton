import os

import numpy as np

from scipy.misc import imread


root = input("please enter a root directory where your 2D slices are ----")  # enter directory of entire KESM data set is
forMat = input("enter the format of the files ----")  # entire the format of image data set
numberOfFolders = input("enter number of folders or cuts----")  # enter the number of cuts per each face of the image
beginningFolder = input("please enter the starting folder----")  # enter the starting folder index ex:0000
listOfJpgs = []; countList = []  # initialize the list of Jpgs collection
# for each of the cuts on the face of block of the resin under KESM
for i in range(0, int(numberOfFolders)):
    directoryCut = root + beginningFolder + str(i) + '/'   # Get the ith cut
    listOffiles = os.listdir(directoryCut)  # list the files in the cut
    count = 0; listOfJpgsCutwise = [];  # initialize variables to store file names in subdirectory of cuts of a KESM face
    # store the file names ending with entered format
    for file in listOffiles:
        if file.endswith(forMat):
            listOfJpgs.append(file)
            count += 1
    listOfJpgsCutwise.append(listOfJpgs)
    countList.append(count)
    # print jpgs per each cut read
    print("files in %i th folder is %i" % (i, count))

# take in 1 sample image to know x, y dimensions of an image
directoryCut = root + beginningFolder + str(0) + '/'
image = imread((os.path.join(directoryCut, listOfJpgs[0])))
m, n = np.shape(image)  # m = 12000, n = 4096
"""
allocating an array by default all the elements are one, since with no absorption by
the tissue optics in the KESM would transmit all the light to the line sensor
3190 is a different number than n- it removes the black patch in the image
"""
allocDimz = max(countList); allocDimy = allocDimz * (m); allocDimx = allocDimz * (3190);
inputIm = np.ones((allocDimz, allocDimy, allocDimx), dtype=np.uint8)
