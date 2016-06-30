import pandas
import json
import time

import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from math import sqrt
from collections import Counter
from statistics import median, pvariance, mean


"""
   The follwing program quantifies a skeletonized structure to meaningful
   statistics like radius, orienation at each node, mean radiuses of the
   nodes in a volume and other statistically significant values
"""


def saveVolumeFeatures(image, nameOfTheImage):
    """
       count number of objects and number of nonzero voxels
       and save them to a json
    """
    date = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
    t = 'countObjects%' + date
    t = t.replace('%', nameOfTheImage)
    label, countObjectsInput = scipy.ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    dictStats = {t: countObjectsInput}
    saveDictAsJson(dictStats)


def saveVolumeMetadata(inputIm):
    """
       save 3D volume meta data like resolution
       and volume of the dataset
    """
    resolution = 0.7 * 0.7 * 10
    volumeOfDataset = inputIm.size * resolution
    dictStats = {'resolution': resolution,
                 'volumeOfDataset': volumeOfDataset}
    saveDictAsJson(dictStats)


def getStatistics(dictF, featureName):
    """
       function to obtain common statistics like mean, median. featureName is the string variable
       that takes in the name of the feature being evaluated
    """
    listF = [float(dictF[k]) for k in dictF]
    meanF = mean(listF)
    varianceF = (pvariance(listF, meanF))
    fCounts = Counter(listF)
    dictStats = {'Zerocount ' + featureName: listF.count(0.0), 'Maximum ' + featureName: max(listF), 'Minimum ' + featureName: min(listF),
                 'Mean ' + featureName: mean(listF),
                 'Median ' + featureName: median(listF),
                 'Variance ' + featureName: varianceF, 'Standard Deviation ' + featureName: sqrt(varianceF),
                 'Unique Counts ' + featureName: len(fCounts)}
    saveDictAsJson(dictStats)
    saveDictAsXlsx(dictStats)


def saveDictAsJson(dictStats, path=None):
    if path is None:
        path = "statistics.json"
    with open(path, "w") as feedsjson:
        feedsjson.write("{}\n".format(json.dumps(dictStats)))
    feedsjson.close()


def saveDictAsXlsx(dictStats, path=None, index=[1]):
    if path is None:
        path = "statistics.xlsx"
    df = pandas.DataFrame(dictStats, index=index)
    writer = pandas.ExcelWriter(path, engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()


def getImportantMetrics(outputDict, binaryVol, skeletonVol, path=None):
    """
    outputList =
    [segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments,
    typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo]
    varList = ['segmentCountdict', 'segmentLengthdict', 'segmentTortuositydict', 'totalSegments', 'typeGraphdict',
    'Average Branching', 'end Points', 'branch Points', 'segmentContractiondict', 'segmentHausdorffDimensiondict', 'cycleInfo']
    """
    dictStats = {'Vascular Volume': np.sum(binaryVol) / binaryVol.size, 'Vectorized Skeleton Volume': np.sum(skeletonVol) / skeletonVol.size,
                 'Percentage skeleton per vascular volume': (np.sum(skeletonVol) / np.sum(binaryVol)) * 100,
                 'Average Branching': outputDict['Average Branching'], 'End Points': outputDict['end Points'],
                 'Branch Points': outputDict['branch Points'], 'Total Number of Segments': sum(list(outputDict['segmentCountdict'].values())) - len(outputDict['cycleInfo']),
                 'Total Number of Cycles': len(outputDict['cycleInfo']),
                 'Total Number of Branch points on cycles': sum([e[0] for e in list(outputDict['cycleInfo'].values())]),
                 'Total Cycle length': sum([e[1] for e in list(outputDict['cycleInfo'].values())]),
                 'Total Length': sum(list(outputDict['segmentLengthdict'].values())),
                 'Mean Length': sum(list(outputDict['segmentLengthdict'].values())) / (sum(list(outputDict['segmentCountdict'].values())) - len(outputDict['cycleInfo'])),
                 'Total Contraction': sum(list(outputDict['segmentContractiondict'].values())),
                 'Mean Contraction': sum(list(outputDict['segmentContractiondict'].values())) / (sum(list(outputDict['segmentCountdict'].values())) - len(outputDict['cycleInfo'])),
                 'Total Hausdorff Dimension': sum(list(outputDict['segmentHausdorffDimensiondict'].values())),
                 'Mean Hausdorff Dimension': sum(list(outputDict['segmentHausdorffDimensiondict'].values())) / (sum(list(outputDict['segmentCountdict'].values())) - len(outputDict['cycleInfo']))}
    saveDictAsJson(dictStats, path)


def plotKDEAndHistogram(ndimarray, path, featureName, chooseBins=False):
    """
       function to obtain a KDE and histogram of the distribution.
       input ndimarray must be array. if it isn't then it is converted
       to one and histogram is plotted
    """
    state = plt.isinteractive()
    # turn interactive mode off to prevent creation of a superfluous figure when viewer is initiated
    plt.ioff()
    if type(ndimarray) == list:
        ndimarray = np.array(ndimarray)
    if chooseBins:
        bins = np.unique(np.round(ndimarray, 0))
        sns.distplot(ndimarray, kde=True, bins=bins)
    else:
        sns.distplot(ndimarray, kde=True)
    plt.xlabel(featureName)
    plt.ylabel("KDE Of " + featureName)
    plt.title("Frequency distribution of " + featureName)
    plt.savefig(path, transparency=True, bbox_inches='tight', pad_inches=1)
    plt.close("all")
    if state:
        plt.ion()


def plotKde(dictionary):
    arr = np.array(list(dictionary.values()))
    X = np.reshape(arr, (arr.size, 1))
    X_plot = np.linspace(X.min(), X.max(), 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
    log_dens = kde.score_samples(X_plot)
    plt.plot(X_plot[:, 0], np.exp(log_dens))
    plt.show()


def saveMultiKde(features, path, featureName, minBin=None, maxBin=None, labels=[], bins=None):
    state = plt.isinteractive()
    plt.ioff()
    for i in range(len(labels)):
        features[i] = np.array(features[i])
        if minBin is not None and maxBin is not None:
            features[i] = features[i][(features[i] >= minBin) & (features[i] < maxBin)]
    if bins is None:
        bins = _getBins(features)
    for i in range(len(labels)):
        clr = sns.color_palette("Set1", n_colors=2, desat=0.5)
        sns.set_palette(clr)
        sns.set_style("whitegrid", {"xtick.color": '0'})
        sns.distplot(features[i], kde=True, label=labels[i], bins=bins)
        plt.xlabel(featureName, fontsize='12')
        plt.ylabel("KDE of " + featureName, fontsize='12')
        plt.title("Frequency distribution of " + featureName, fontsize='15')
    plt.legend()
    plt.savefig(path, transparency=True, bbox_inches='tight', pad_inches=1)
    plt.close("all")
    if state:
        plt.ion()


def _IQR(data):
    """
    return the interquartile range of the data
    """
    return np.subtract(*np.percentile(data, [75, 25]))


def _findBinWidth(data, scale=None):
    """
    given a list of data (1D-array), find the optimal bin width
    based on the Freedman-Diaconis rule
      h = 2 * IQR * n ^(-1/3)
    where IQR is the interquartile range (between 75% and 25%)
    """
    if data.ndim > 1:
        data = data.ravel()
    if not scale:  # we aren't adding a manual scaling value
        scale = 1
    return 2 * _IQR(data) * data.size ** (-1 / 3) / scale


def _getBins(features):

    # typical bin width optimization is the square root of the number of data points
    # bw = np.sqrt(df[metricName].count())
    data = np.empty(())
    for i in range(len(features)):
        data = np.hstack((data, features[i]))
    bw = _findBinWidth(data[1:], scale=2)
    minV = data.min()
    maxV = data.max()
    print(minV, maxV, bw)
    bins = np.arange(minV, maxV, bw)
    return bins


def welchsTtest(a, b):
    """ computes t test on unequal number of sample sizes,
    if sample size is equal returns same answer as a t test
    assumes gaussian distribution of data with unequal variances
    """
    t, p = scipy.stats.ttest_ind(a, b, axis=0, equal_var=False)
    return (t, p, a.size, b.size, pvariance(a.tolist(), mean(a.tolist())), pvariance(b.tolist(), mean(b.tolist())))


def nonParametricRanksums(a, b):
    """
    no parameters of data are assumed
    """
    t, p = scipy.stats.ranksums(a, b)
    return t, p


def plotMultiKde(features, featureName, minBin, maxBin, labels, bins=None):
    state = plt.isinteractive()
    plt.ioff()
    for i in range(len(labels)):
        features[i] = np.array(features[i])
        if minBin is not None and maxBin is not None:
            features[i] = features[i][(features[i] >= minBin) & (features[i] < maxBin)]
    if bins is None:
        bins = _getBins(features)
    for i in range(len(labels)):
        clr = sns.color_palette("Set1", n_colors=2, desat=0.5)
        sns.set_palette(clr)
        sns.set_style("whitegrid", {"xtick.color": '0'})
        sns.distplot(features[i], kde=True, label=labels[i], bins=bins)
        plt.xlabel(featureName, fontsize='12')
        plt.ylabel("KDE of " + featureName, fontsize='12')
        plt.title("Frequency distribution of " + featureName, fontsize='15')
    plt.legend()
    if state:
        plt.ion()
    return plt


def plot3Dfigure(inrerpolatedImage):
    """
       plots a 3D volume
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from skimage import measure
    verts, faces = measure.marching_cubes(inrerpolatedImage, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, inrerpolatedImage.shape[0])
    ax.set_ylim(0, inrerpolatedImage.shape[1])
    ax.set_zlim(0, inrerpolatedImage.shape[2])
    ax.set_aspect('equal')
