import itertools
import numpy as np
import scipy
import time
from scipy import ndimage
import matplotlib.pyplot as plt
import networkx as nx

"""

   reference paper
   http://www.cb.uu.se/~ingela/Manuscripts/PR99_32_7_1225.pdf

"""


def setTemp(inputImb, inputIma, se):

    lb, obefore = scipy.ndimage.measurements.label(inputImb, se)
    lbc, ocbefore = scipy.ndimage.measurements.label(1 - inputImb, se)
    la, oafter = scipy.ndimage.measurements.label(inputIma, se)
    lac, ocafter = scipy.ndimage.measurements.label(1 - inputIma, se)

    if obefore == oafter and ocbefore == ocafter:
        deletableTemp = 1
    else:
        deletableTemp = 0

    return deletableTemp


def getBoundariesOfimage(image):
    sElement = ndimage.generate_binary_structure(3, 1)
    erode_im = scipy.ndimage.morphology.binary_erosion(image, sElement)
    b = image - erode_im
    return b, erode_im


def getPaddedimage(image):
    z, m, n = np.shape(image)
    paddedShape = z + 2, m + 2, n + 2
    padImage = np.zeros((paddedShape), dtype=np.uint8)
    padImage[1:z + 1, 1:m + 1, 1:n + 1] = image
    return padImage


def getConditionA1(image):
    face1 = np.logical_xor(image[0][1][1], image[2][1][1])

    face2 = np.logical_xor(image[1][0][1], image[1][2][1])

    face3 = np.logical_xor(image[1][1][0], image[1][1][2])

    if (face1 and face2 and face3) == 0:
        setA1 = True
    else:
        setA1 = False

    return setA1


def getConditionA2(image):

    edge1 = np.logical_and(image[1][1][1], image[1][0][2]); b1 = np.logical_and(image[1][0][1], image[1][1][2])

    edge2 = np.logical_and(image[1][1][1], image[1][0][0]); b2 = np.logical_and(image[1][0][1], image[1][1][0])

    edge3 = np.logical_and(image[1][1][1], image[1][2][2]); b3 = np.logical_and(image[1][1][2], image[1][2][1])

    edge4 = np.logical_and(image[1][1][1], image[1][2][0]); b4 = np.logical_and(image[1][1][0], image[1][2][1])

    edge5 = np.logical_and(image[1][1][1], image[0][1][0]); b5 = np.logical_and(image[0][1][1], image[1][1][0])

    edge6 = np.logical_and(image[1][1][1], image[2][1][0]); b6 = np.logical_and(image[1][1][0], image[2][1][1])

    edge7 = np.logical_and(image[1][1][1], image[0][0][1]); b7 = np.logical_and(image[0][1][1], image[1][0][1])

    edge8 = np.logical_and(image[1][1][1], image[0][1][2]); b8 = np.logical_and(image[0][1][1], image[1][1][2])

    edge9 = np.logical_and(image[1][1][1], image[2][2][1]); b9 = np.logical_and(image[2][1][1], image[1][2][1])

    edge10 = np.logical_and(image[1][1][1], image[0][2][1]); b10 = np.logical_and(image[0][1][1], image[1][2][1])

    edge11 = np.logical_and(image[1][1][1], image[2][0][0]); b11 = np.logical_and(image[1][0][1], image[2][1][1])

    edge12 = np.logical_and(image[1][1][1], image[2][1][2]); b12 = np.logical_and(image[1][1][2], image[2][1][1])

    edges = [edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge9, edge10, edge11, edge12]

    b = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12]

    decideEdgelist = []
    for i in range(0, 8):
        if edges[i] == 1 and b[i] == 0:
            decideEdgelist.append(1)
        else:
            decideEdgelist.append(1)

    if np.sum(decideEdgelist) != 0:
        setA2 = True
    else:
        setA2 = False

    return setA2


def getConditionA3(image):

    a1 = np.sum([image[1][1][0], image[1][2][0], image[1][2][1], image[2][1][0], image[2][1][1], image[2][2][1]])

    corner1 = np.logical_and(image[1][1][1], image[2][2][0])

    a2 = np.sum([image[1][1][2], image[1][2][1], image[1][2][2], image[2][1][1], image[2][1][2], image[2][2][1]])

    corner2 = np.logical_and(image[1][1][1], image[2][2][2])

    a3 = np.sum([image[1][0][0], image[1][0][1], image[1][1][0], image[2][0][1], image[2][1][0], image[2][1][1]])

    corner3 = np.logical_and(image[1][1][1], image[2][0][0])

    a4 = np.sum([image[1][0][1], image[1][0][2], image[1][1][2], image[2][0][2], image[2][1][1], image[2][1][2]])

    corner4 = np.logical_and(image[1][1][1], image[2][0][2])

    a5 = np.sum([image[0][0][1], image[0][1][0], image[0][1][1], image[1][0][0], image[1][0][1], image[1][1][0]])

    corner5 = np.logical_and(image[1][1][1], image[0][0][0])

    a6 = np.sum([image[0][1][0], image[0][1][1], image[0][2][1], image[1][1][0], image[1][2][0], image[1][2][1]])

    corner6 = np.logical_and(image[1][1][1], image[0][2][0])

    a7 = np.sum([image[0][1][1], image[0][1][2], image[0][2][1], image[1][1][2], image[1][2][1], image[1][2][2]])

    corner7 = np.logical_and(image[1][1][1], image[0][2][2])

    a8 = np.sum([image[0][0][1], image[0][1][1], image[0][1][2], image[1][0][1], image[1][0][2], image[1][1][2]])

    corner8 = np.logical_and(image[1][1][1], image[0][0][2])

    a = [a1, a2, a3, a4, a5, a6, a7, a8]

    corners = [corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8]
    decideCornerlist = []
    for i in range(0, 8):
        if corners[i] == 1 and a[i] == 0:
            decideCornerlist.append(1)
        else:
            decideCornerlist.append(1)

    if np.sum(decideCornerlist) != 0:
        setA3 = True
    else:
        setA3 = False

    return setA3


def getSurfSkeletonpass(image, pass_no):
    z, m, n = np.shape(image)
    # paddedShape = z, m, n
    # temp_del = np.ones((paddedShape), dtype=np.uint8)
    # result = np.ones((paddedShape), dtype=np.uint8)
    b, erodedIm = getBoundariesOfimage(image)
    acopy = np.copy(image)
    setLabelim = np.copy(image)
    # sElement = ndimage.generate_binary_structure(3, 1)
    # sElement2 = ndimage.generate_binary_structure(3, 2)
    # se = np.uint8(sElement2) - np.uint8(sElement)
    # labeledInternal = ndimage.measurements.label(erodedIm, se)
    setLabelim[b == 1] = pass_no + 50
    setLabelimCopy = np.copy(setLabelim)
    # setLabelim[labeledInternal != 1] = pass_no + 51
    # numpixel_removed = 0
    for k in range(1, z - 1):
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if b[k, i, j] != 1:
                    continue
                # setLabelim[k, i, j]
                asub = acopy[k - 1: k + 2, j - 1: j + 2, i - 1: i + 2]
                asub2 = b[k - 1: k + 2, j - 1: j + 2, i - 1: i + 2]
                setA1 = getConditionA1(asub)
                setA2 = getConditionA2(asub2)
                setA3 = getConditionA3(asub2)
                if setA1 or setA2 or setA3:
                    setLabelimCopy[k, i, j] == -1
    # call a function which looks at the 6 connected components in the 18 neighborhood
    # then remove
    setLabelim = setRemainingvoxels(setLabelimCopy)
    return setLabelim
    #             if setLabelimcopy[k, i, j] != -1:
    #                 # count Nf18
    #                 # treat non multiple voxels as already removed
    #                 # nf18 > 1 dont remove it
    #                 pass
    # sumBefore = np.sum(temp_del)
    # temp_del[setLabelim == pass_no + 50] = 0
    # numpixel_removed = sumBefore - np.sum(temp_del)
    # acopy = np.multiply(acopy, temp_del)
    # acopy = np.uint8(acopy)
    # result[:] = acopy[:]
    # return numpixel_removed, result


def setRemainingvoxels(image):
    pass
    # extract image portion
    # find number of 6 connected components with central voxel as face neighbor
    im = image[0: 2][1][1]
    # convert image to graph
    g = binaryImageToGraph(im)
    numConnectedcomponents = nx.number_connected_components(g)
    return numConnectedcomponents


def binaryImageToGraph(npArray):
    """
    convert 2d/3d binary array to skeleton
    and then to a graph
    """
    # Basic binary array sanity checks
    assert npArray.ndim in [2, 3]
    assert npArray.dtype == np.uint8
    assert npArray.min() >= 0
    assert npArray.max() <= 1

    g = nx.Graph()
    # cycles = []
    aShape = npArray.shape
    for coords, value in np.ndenumerate(npArray):
        # The point we start from must be "solid"
        if value != 1:
            continue

        # Check 1 step in each direction
        stepDirect = itertools.product((0, 1), repeat=npArray.ndim)
        listStepDirect = list(stepDirect)
        listStepDirect = listStepDirect[1:]
        for d in listStepDirect:
            # COnstruct the coordinate which is shifted
            # by 1 in the d'th dimension
            nearByCoordinate = np.array(coords)
            nearByCoordinate = nearByCoordinate + np.array(d)
            nearByCoordinate = tuple(nearByCoordinate)
            # if nearByCoordinate not in g.nodes():
            #     # Bounds check on the shifted coordinate
            if nearByCoordinate == aShape:
                continue
                # It it is solid, then assign an edge to that graph
            if npArray[nearByCoordinate] == 1:
                g.add_edge(coords, nearByCoordinate)
    return g


def getMedialSurfaceSkeleton3D(image):
    """function to skeletonize a 3D binary image with object in brighter contrast than background.
    In other words, 1 = object, 0 = background
    """
    assert image.ndim == 3
    assert image.max() == 1
    assert image.min() >= 0
    assert image.dtype == np.uint8
    z, m, n = np.shape(image)
    padImage = getPaddedimage(image)
    start_skeleton = time.time()
    pass_no = 0
    numpixel_removed = 0
    numpixel_removedList = []
    while pass_no == 0 or numpixel_removed > 0:
        numpixel_removed, padImage = getSurfSkeletonpass(padImage, pass_no)
        print("number of pixels removed in pass", pass_no, "is ", numpixel_removed)
        numpixel_removedList.append(numpixel_removed)
        pass_no += 1
    print("done %i number of pixels in %i seconds" % (np.sum(image), time.time() - start_skeleton))

    return padImage[1:z + 1, 1:m + 1, 1:n + 1]

if __name__ == '__main__':
    sampleCube = np.zeros((3, 64, 64), dtype=np.uint8)
    sampleCube[:, 16:49, 16:49] = 1
    resultCube = getMedialSurfaceSkeleton3D(sampleCube)
    plt.subplot(2, 3, 1)
    nr, nc = np.shape(resultCube[0])
    plt.imshow(sampleCube[0], cmap=plt.cm.gray, vmin=0, vmax=1, extent=[0, nr, 0, nc])
    plt.colorbar()
    plt.subplot(2, 3, 2)
    plt.imshow(sampleCube[1], cmap=plt.cm.gray, vmin=0, vmax=1, extent=[0, nr, 0, nc])
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.imshow(sampleCube[2], cmap=plt.cm.gray, vmin=0, vmax=1, extent=[0, nr, 0, nc])
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.imshow(resultCube[0], cmap=plt.cm.gray, vmin=0, vmax=1, extent=[0, nr, 0, nc])
    plt.colorbar()
    plt.subplot(2, 3, 5)
    plt.imshow(resultCube[1], cmap=plt.cm.gray, vmin=0, vmax=1, extent=[0, nr, 0, nc])
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.imshow(resultCube[2], cmap=plt.cm.gray, vmin=0, vmax=1, extent=[0, nr, 0, nc])
    plt.colorbar()
    plt.show()
