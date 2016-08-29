import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from six.moves import cPickle

from KESMAnalysis.imgtools import loadStack, saveStack
from KESMAnalysis.pipeline.pipelineComponents import watershedMarker

from skeleton.thin3DVolume import getThinned3D
from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton
from skeleton.plotStats import getStatistics, plotKDEAndHistogram, saveDictAsJson, nonParametricRanksums
from skeleton.plotStats import getImportantMetrics, plotMultiKde, saveMultiKde, welchsTtest
from runscripts.segmentStatsLRwhitecutoffs import getSegmentStats
from skeleton.pruning import getPrunedSkeleton

# load 2D facelets median filtered to be vectorized
filePath = input("please enter a path to directory with 2D grey scale slices to be skeletonized")
stack = loadStack(filePath)

# load aspect ratio to make the 3D volume isotropic using quadratic interpolation
# aspectRatio = input("please enter resolution of a voxel in 3D with resolution in x followed by y and z (spaces in between)")
# aspectRatio = [float(item) for item in aspectRatio.split(' ')]
aspectRatio = [0.7, 0.7, 5]
stack = ndimage.interpolation.zoom(stack, zoom=aspectRatio, order=2, prefilter=False)

# binarize using 3D watershed transform
binaryVol = watershedMarker(stack)

# save binary volume
np.save(filePath + "/" + "binary.npy", binaryVol)

# convert to boolean becasue getThinned expects a boolean input
binaryVol = binaryVol.astype(bool)

# thin binarized volume of vessels
thinnedVol = getShortestPathSkeleton(getThinned3D(np.swapaxes(binaryVol, 0, 2)))
thinnedVol = np.swapaxes(thinnedVol, 0, 2)
# decluster thinned volume
skeletonVol = getPrunedSkeleton(thinnedVol, cutoff=5)

# save the skeleton volume as pngs
saveStack(skeletonVol, filePath + "/skeleton")

# save the skeleton volume as npy
np.save(filePath + "/skeleton/" + "skeleton.npy", skeletonVol)

# vectorize and find metrics
(segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments,
 typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo, isolatedEdgeInfo) = getSegmentStats(skeletonVol, False)

# save the metrics dumping using cPickle as a list of elements as obtained from getSegmentStats
# as segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondic


outputList = [segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching,
              endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo, isolatedEdgeInfo]
varList = ['segmentCountdict', 'segmentLengthdict', 'segmentTortuositydict', 'totalSegments', 'typeGraphdict', 'Average Branching',
           'end Points', 'branch Points', 'segmentContractiondict', 'segmentHausdorffDimensiondict', 'cycleInfo', 'isolatedEdgeInfo']
outputDict = {}
for var, op in zip(varList, outputList):
    outputDict[var] = op
cPickle.dump(outputDict, open("/home/pranathi/MTR/metrics_cerebellum.p", "wb"))

# to load the statistics
# outputDict = cPickle.load(open("/home/3scan-data/exports/78c507c6e37294470/block-00000000/region-00023120-00023632-00023124-00023636-00000282-00000354/median/skeleton/skeletonregion-00023120-00023632-00023124-00023636-00000282-00000354.npymetrics.p", "rb"))

# save important statistics in a json file
getImportantMetrics(outputDict, binaryVol, skeletonVol)

graphs = ['segmentLengthdict', 'segmentHausdorffDimensiondict', 'segmentContractiondict']
FeatureName = ['Segment Length(um)', 'Segment Hausdorff Dimension', 'Segment Contraction']
dictforebrain = cPickle.load(open("/home/pranathi/metrics_Forebrain.p", "rb"))
dictcerebellum = cPickle.load(open("/home/pranathi/metrics_Cerebellum.p", "rb"))
binsmin = [0, 0.7, 0.2]
binsmax = [200, 1.25, 1]
for i, graph in enumerate(graphs):
    saveMultiKde([list(dictcerebellum[graph].values()), list(dictforebrain[graph].values())], "/home/pranathi/MTR/" + "Segment Length" +
                 "Histogram.svg", "Segment Length(um)", minBin=0, maxBin=200, labels=["cerebellum", "forebrain"])
    plotKDEAndHistogram(list(dictforebrain[graph].values()), "/home/pranathi/MTR/" + FeatureName[i] + "Histogram_Forebrain.png", FeatureName[i])
    plotKDEAndHistogram(list(dictcerebellum[graph].values()), "/home/pranathi/MTR/" + FeatureName[i] + "Histogram_Cerebellum.png", FeatureName[i])
    getStatistics(dictforebrain[graph], FeatureName[i] + "Histogram_Forebrain")
    getStatistics(dictcerebellum[graph], FeatureName[i] + "Histogram_Cerebellum")


otherStats = ['total Segments', 'total Cycles', 'end Points', 'branch Points']
stats = [dictcerebellum['totalSegments'], len(dictcerebellum['cycleInfo']), dictcerebellum['end Points'], dictcerebellum['branch Points']]
otherStatsDict = {}
for i, stat in enumerate(otherStats):
    otherStatsDict[stat] = stats[i]
saveDictAsJson(otherStatsDict, "/home/pranathi/MTR/otherStats_cerebellum.json")


fig_object = plt.figure()
binsmin = [0, 0, 0.2, 0.7, 0]
binsmax = [4, 200, 1, 1.25, 6]
plt.suptitle("Vectorization Metrics", fontsize='20')
plotName = ['Branching Index', 'Segment Length(um)', 'Segment Contraction', 'Segment Hausdorff Dimension']
plotGraphs = ['segmentCountdict', 'segmentLengthdict', 'segmentContractiondict', 'segmentHausdorffDimensiondict']
for i in range(4):
    plt.subplot(2, 2, i + 1)
    graph = plotGraphs[i]
    plotMultiKde([list(dictforebrain[graph].values()), list(dictcerebellum[graph].values())], plotName[i], minBin=binsmin[i], maxBin=binsmax[i],
                 labels=["Forebrain", "Cerebellum"])

cPickle.dump(fig_object, open("fig_picke", "wb"))
plotName = ['Branching Index', 'Segment Length(um)', 'Segment Contraction', 'Segment Hausdorff Dimension']
plotGraphs = ['segmentCountdict', 'segmentLengthdict', 'segmentContractiondict', 'segmentHausdorffDimensiondict']
tstats = {}
rankStats = {}
for i in range(4):
    graph = plotGraphs[i]
    a = np.array(list(dictforebrain[graph].values()))
    b = np.array(list(dictcerebellum[graph].values()))
    t, p, n1, n2, v1, v2 = welchsTtest(a, b)
    tstats[plotName[i]] = (t, p, n1, n2, v1, v2)
    rankStats[plotName[i]] = nonParametricRanksums(a, b)

saveDictAsJson(tstats, "/home/pranathi/MTR/tstats.json")
saveDictAsJson(rankStats, "/home/pranathi/MTR/wilcoxon.json")
