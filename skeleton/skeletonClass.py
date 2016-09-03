from kesm.projects.KESMAnalysis.imgtools import loadStack, saveStack

from metrics.segmentStats import SegmentStats

from scipy import ndimage

from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.networkxGraphFromArray import getNetworkxGraphFromArray
from skeleton.thinVolume import getThinned
from skeleton.pruning import getPrunedSkeleton
from skeleton.unitWidthCurveSkeleton import getShortestPathSkeleton

"""
abstract class that encompasses all stages of skeletonization leading to quantification
    1) thinning
    2) unit width curve skeleton to remove crowded regions(
       regions with more than 4 ones in a 2nd ordered neighborhood of a voxel)
    3) pruning
    4) graph conversion
"""


class Skeleton:
    def __init__(self, path, **kwargs):
        # initialize input array
        if type(path) is str:
            self.path = path
            self.inputStack = loadStack(self.path).astype(bool)
        else:
            self.inputStack = path
        if kwargs != {}:
            aspectRatio = kwargs["aspectRatio"]
            self.inputStack = ndimage.interpolation.zoom(self.inputStack, zoom=aspectRatio, order=2, prefilter=False)

    def setThinningOutput(self):
        # Thinning output
        self.thinnedStack = getThinned(self.inputStack)

    def setUnitWidthSkeletonOutput(self):
        # Crowded regions removed from thinning output
        self.setThinningOutput()
        self.skeletonStack = getShortestPathSkeleton(self.thinnedStack)

    def setNetworkGraph(self, findSkeleton=True):
        # Network graph of the crowded region removed output
        if findSkeleton:
            self.setUnitWidthSkeletonOutput()
        else:
            self.skeletonStack = self.inputStack
        self.graph = removeCliqueEdges(getNetworkxGraphFromArray(self.skeletonStack))

    def setPrunedSkeletonOutput(self):
        # Prune unnecessary segments in crowded regions removed skeleton
        self.setNetworkGraph()
        self.outputStack = getPrunedSkeleton(self.skeletonStack, self.graph)

    def getNetworkGraph(self):
        # Network graph of the final output skeleton stack
        self.setPrunedSkeletonOutput()
        self.outputGraph = removeCliqueEdges(getNetworkxGraphFromArray(self.outputStack))

    def saveStack(self):
        # Save output skeletonized stack as series of pngs in the path under a subdirectory skeleton
        self.setPrunedSkeletonOutput()
        saveStack(self.outputStack, self.path + "skeleton/")

    def segmentStatsBeforePruning(self):
        self.setNetworkGraph()
        self.statsBefore = SegmentStats(self.graph)

    def segmentStatsAfterPruning(self):
        self.getNetworkGraph()
        self.statsAfter = SegmentStats(self.outputGraph)
