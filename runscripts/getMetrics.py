import numpy as np

from scipy import ndimage
from six.moves import cPickle

from KESMAnalysis.imgtools import loadStack, saveStack
from KESMAnalysis.pipeline.pipelineComponents import watershedMarker

from skeleton.thin3DVolume import getThinned3D
from skeleton.orientationStatisticsSpline import getStatistics, plotKDEAndHistogram, getImportantMetrics, plotMultiKde
from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton
from skeleton.segmentStats import getSegmentStats
from skeleton.pruning import getPrunedSkeleton

# load 2D facelets median filtered to be vectorized
filePath = "/home/3scan-data/exports/78c507c6e37294470/block-00000000/region-00013048-00013560-00023340-00023852-00000117-00000189/median/"
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
thinnedVol = getThinned3D(np.swapaxes(binaryVol, 0, 2))
thinnedVol = np.swapaxes(thinnedVol, 0, 2)
# decluster thinned volume
skeletonVol = getPrunedSkeleton(getShortestPathSkeleton(thinnedVol), cutoff=5)

# save the skeleton volume as pngs
saveStack(skeletonVol, filePath + "/skeleton")

# save the skeleton volume as npy
np.save(filePath + "/skeleton/" + "skeleton.npy", skeletonVol)

# vectorize and find metrics
(segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments,
 typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo, isolatedEdgeInfo) = getSegmentStats(skeletonVol, False)

# save the metrics dumping using cPickle as a list of elements as obtained from getSegmentStats
# as segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict

outputList = [segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo, isolatedEdgeInfo]
varList = ['segmentCountdict', 'segmentLengthdict', 'segmentTortuositydict', 'totalSegments', 'typeGraphdict', 'Average Branching', 'end Points', 'branch Points', 'segmentContractiondict', 'segmentHausdorffDimensiondict', 'cycleInfo', 'isolatedEdgeInfo']
outputDict = {}
for var, op in zip(varList, outputList):
    outputDict[var] = op
cPickle.dump(outputDict, open("/home/pranathi/MTR/metrics_cerebellum.p", "wb"))

# to load the statistics
# outputDict = cPickle.load(open("/home/3scan-data/exports/78c507c6e37294470/block-00000000/region-00023120-00023632-00023124-00023636-00000282-00000354/median/skeleton/skeletonregion-00023120-00023632-00023124-00023636-00000282-00000354.npymetrics.p", "rb"))

# save important statistics in a json file
getImportantMetrics(outputDict, binaryVol, skeletonVol)

graphs = ['segmentCountdict', 'segmentLengthdict', 'segmentHausdorffDimensiondict', 'segmentContractiondict', 'typeGraphdict']
FeatureName = ['Branching Index', 'Segment Length(um)', 'Segment Hausdorff Dimension', 'Segment Contraction', 'Type of Subgraphs']
dictforebrain = cPickle.load(open("/home/pranathi/MTR/metrics_forebrain.p", "rb"))
dictcerebellum = cPickle.load(open("/home/pranathi/MTR/metrics_cerebellum.p", "rb"))
for i, graph in enumerate(graphs):
    plotMultiKde(list(dictforebrain[graph].values()), list(dictcerebellum[graph].values()), "/home/pranathi/MTR/" + FeatureName[i] + "Histogram.png", FeatureName[i])
    plotKDEAndHistogram(list(dictforebrain[graph].values()), "/home/pranathi/MTR/" + FeatureName[i] + "Histogram_Forebrain.png", FeatureName[i])
    plotKDEAndHistogram(list(dictcerebellum[graph].values()), "/home/pranathi/MTR/" + FeatureName[i] + "Histogram_Cerebellum.png", FeatureName[i])
    getStatistics(dictforebrain[graph], FeatureName[i] + "Histogram_Forebrain")
    getStatistics(dictcerebellum[graph], FeatureName[i] + "Histogram_Cerebellum")