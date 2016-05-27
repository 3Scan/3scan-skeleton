from scipy import ndimage

from KESMAnalysis.imgtools import loadStack
from skeleton.thin3DVolume import getThinned3D
from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton
from skeleton.segmentStats import getSegmentStats

# load 2D facelets to be processed
filePath = input("please enter a root directory where your median filtered 2D slices are----")
stack = loadStack(filePath)

# load aspect ratio to make the 3D volume isotropic
aspectRatio = input("please enter resolution of a voxel in 3D with resolution in z followed by y and x")
aspectRatio = [float(item) for item in aspectRatio.split(' ')]
stack = ndimage.interpolation.zoom(stack, zoom=aspectRatio, order=2, prefilter=False)

# binarize using 3D watershed transform
binaryVol = binarize(stack)

# thin binarized volume of vessels
thinnedVol = getThinned3D(binaryVol)

# decluster thinned volume
skeletonVol = getShortestPathSkeleton(thinnedVol)

# vectorize and find metrics
segmentCountdict, segmentLengthdict, segmentTortuositydict, totalSegments, typeGraphdict, avgBranching, endP, branchP, segmentContractiondict, segmentHausdorffDimensiondict = getSegmentStats(skeletonVol)

