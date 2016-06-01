import numpy as np

from scipy import ndimage
from six.moves import cPickle

from KESMAnalysis.imgtools import loadStack, saveStack
from KESMAnalysis.pipeline.pipelineComponents import watershedMarker

from skeleton.thin3DVolume import getThinned3D
from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton
from skeleton.segmentStats import getSegmentStats

# load 2D facelets to be processed
filePath = input("please enter a root directory where your median filtered 2D slices are----")
stack = loadStack(filePath)

# load aspect ratio to make the 3D volume isotropic using quadratic interpolation
aspectRatio = input("please enter resolution of a voxel in 3D with resolution in x followed by y and z (spaces in between)")
aspectRatio = [float(item) for item in aspectRatio.split(' ')]
stack = ndimage.interpolation.zoom(stack, zoom=aspectRatio, order=2, prefilter=False)

# binarize using 3D watershed transform
binaryVol = watershedMarker(stack)
binaryVol[binaryVol == 255] = 1
binaryVol = binaryVol.astype(bool)
# save binary volume
np.save(filePath + "binary.npy", binaryVol)
# thin binarized volume of vessels
thinnedVol = getThinned3D(binaryVol)

# decluster thinned volume
skeletonVol = getShortestPathSkeleton(thinnedVol)

# save the skeleton volume as pngs
saveStack(skeletonVol, filePath + "/skeleton")

# save the skeleton volume as npy
np.save(filePath + "/skeleton/" + "skeleton.npy", skeletonVol)

# vectorize and find metrics
segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo = getSegmentStats(skeletonVol)

# save the metrics dumping using cPickle as a list of elements as obtained from getSegmentStats
# as segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict

outputList = [segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict, cycleInfo]
varList = ['segmentCountdict', 'segmentLengthdict', 'segmentTortuositydict', 'totalSegments', 'typeGraphdict', 'Average Branching', 'end Points', 'branch Points', 'segmentContractiondict', 'segmentHausdorffDimensiondict', 'cycleInfo']
outputDict = {}
for var, op in zip(varList, outputList):
    outputDict[var] = op
cPickle.dump(outputDict, open(filePath + "metrics.p", "wb"))

# to load the statistics
# outputDict = cPickle.load(open("metrics.p", "rb"))
