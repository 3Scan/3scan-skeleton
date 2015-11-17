import numpy as np

import time

from scipy import ndimage

"""
   radius = distance of node to nearest zero co-ordinate(edge)
   if radius is zero it is a single isolated voxel which may be
   due to noise, itself forms an edge
   these voxels can be removed through some sanity checks with
   their presence in the grey scale iage
"""


def _getBouondariesOfimage(image):
    """
       function to find boundaries/border/edges of the array/image
    """

    sElement = ndimage.generate_binary_structure(3, 1)
    erode_im = ndimage.morphology.binary_erosion(image, sElement)
    boundaryIm = image - erode_im
    assert np.sum(boundaryIm) <= np.sum(image)
    return boundaryIm


def _intersectCrowded(a, b):
    """

       return the intersection of two lists

    """
    if list(set(a) & set(b)) == list(set(a)):
        return 1
    else:
        return 0


def getRadiusByPointsOnCenterline(skeletonIm, boundaryIm):
    from scipy import ndimage
    import time
    startt = time.time()
    nonZeroIndices = list(np.transpose(np.nonzero(skeletonIm)))
    skeletonIm[skeletonIm == 0] = 2
    skeletonIm[boundaryIm == 1] = 0
    distTransformedIm = ndimage.distance_transform_edt(skeletonIm, [0.7, 0.7, 10])
    dictOfNodesAndRadius = {}
    for items in nonZeroIndices:
        dictOfNodesAndRadius[tuple(items)] = distTransformedIm[tuple(items)]
    distTransformedIm[skeletonIm == 0] = 0
    print("time taken to find the nodes and their radius is ", time.time() - startt, "seconds")
    return dictOfNodesAndRadius, distTransformedIm

if __name__ == '__main__':
    startt = time.time()
    # load the skeletonized image
    skeletonIm = np.load('/Users/3scan_editing/records/scratch/shortestPathSkel.npy')
    thresholdIm = np.load('/Users/3scan_editing/records/scratch/threshold3dtestoptimize.npy')
    # finding edges of the microvasculature
    boundaryIm = _getBouondariesOfimage(thresholdIm)
    dictOfNodesAndRadius, distTransformedIm = getRadiusByPointsOnCenterline(skeletonIm, boundaryIm)

    # find the non-zero indices which represent the nodes in the skeletonized image
    # loading the thresholded image to find edges of the microvasuclature# mask out the zeros in the skeletonized image so when distance transform is applied, distance
    # to these zeros are not found

    # mask the co-ordinates obtained in the edge image to be zeros
    # finding EDT- distance from the nonzero points on the centerline to the edges
    # Sanity check for the isolated pixels which are on the edge image themselves
    # take out the distance transform value as the radius of the node
    # /nonzero co-ordinates which are present on the original skeleton
    # # sanity check -  only the nodes/nonzero co-ordinates in the skeletonized image are
    # # looked for radius
    # meanRadius = np.float32(mean(listRadius))
    # print("arithematic mean of all the radius of the vessels is", meanRadius)
    # medianRadius = np.float32(median(listRadius))
    # print("median of all the radius of the vessels is", medianRadius)
    # if lengthOfNodes % 2 == 0:
    #     highmedianRadius = np.float32(median_high(listRadius))
    #     print("high medianRadius is the highest among the two center ones if even number of elements in the list ", highmedianRadius)
    #     lowmedianRadius = np.float32(median_low(listRadius))
    #     print("low medianRadius is lowest among the two center ones if even number of elements in the list ", lowmedianRadius)
    # modeRadius = mode(dictOfNodesAndRadius.values())
    # print("most frequently occuring vessel radius/ mode is", modeRadius)
    # print("voxels with mode radius are %i, out of all the %i nonzero voxels  " % ((listRadius.count(modeRadius)), lengthOfNodes))
    # print("data spread out around the mean is", varianceRadius)
    # stddevRadius = np.float32(sqrt(varianceRadius))
    # print("standard deviation is", stddevRadius)
    # sortedListRadiusDecim = list(np.float16(listRadius))
    # print("unique number of filaments/vessels with different radius is", len(radiusCounts))
    # df = pandas.DataFrame.from_dict(radiusCounts, orient='index')
    # df = df.sort()
    # df2 = pandas.DataFrame.from_dict(radiusCounts2, orient='index')
    # df2 = df2.sort()
    # # df = np.round(df,decimals=2)
    # # df.reindex_axis(sorted(df.columns), axis=0)
    # from pylab import text
    # ax = df.plot(kind='density', rot=75, title='radius of microvasculature plot', fontsize=12, subplots=True)
    # # text(modeRadius, listRadius.count(modeRadius), 'mode and median')
    # ax.set_xlabel('radii of the microvasculature sample in voxels')
    # ax.set_ylabel('frequency of the radii')

    # # ax.annotate('mode, median', xy=(modeRadius, listRadius.count(modeRadius)), xytext=(1.5, 3),
    # #             arrowprops=dict(facecolor='black', shrink=0.1),
    # #             )
    # # ax.annotate('mean', xy=(meanRadius, listRadius.count(meanRadius)))
    # # f = partial(Series.round, decimals=2)
    # # df.apply(f)
    # df.plot(kind='line', rot=75, title='radius of microvasculature plot', fontsize=8, subplots=True)
    # df.plot(kind='bar', rot=75, title='radius of microvasculature plot', fontsize=8, subplots=True)
    # df.plot(kind='density', rot=75, title='radius of microvasculature plot', fontsize=12, subplots=True)
    # df2.plot(kind='pie', rot=75, title='radius of microvasculature plot', fontsize=8, subplots=True)
    # # import matplotlib.pyplot as plt
    # # yaxisVars = list(radiusCounts.values())
    # # xaxisVars = list(radiusCounts.keys())

    # # pos = np.arange(len(xaxisVars))
    # # width = 1  # gives histogram aspect to the bar diagram

    # # ax = plt.axes()
    # # ax.set_xticks(pos + (width / 2))
    # # ax.set_xticklabels(xaxisVars, rotation=75)

    # # plt.bar(pos, yaxisVars, width, color='r')
    # # plt.show()
    # # print("time taken to find the nodes and their radius is ", time.time() - startt, "seconds")
