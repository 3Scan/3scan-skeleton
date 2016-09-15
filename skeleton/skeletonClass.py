import os

import numpy as np
from scipy import ndimage

from kesm.projects.KESMAnalysis.imgtools import loadStack, saveStack
from metrics.segmentStats import SegmentStats
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
        # path : can be an 3D binary array or series of png images
        # or a numpy(.npy) array
        # if path is a 3D volume saveSkeletonStack, saves series of
        # skeleton pngs in present directory
        if type(path) is str:
            if path.endswith("npy"):
                # extract rootDir of path
                self.path = os.path.split(path)[0] + os.sep
                self.inputStack = np.load(path)
            else:
                self.path = path
                self.inputStack = loadStack(self.path).astype(bool)
        else:
            self.path = os.getcwd()
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

    def setNetworkGraph(self, findSkeleton=False):
        # Network graph of the crowded region removed output
        # Generally the function expects a skeleton
        # and findSkeleton is False by default
        if findSkeleton is True:
            self.setUnitWidthSkeletonOutput()
        else:
            self.skeletonStack = self.inputStack
        self.graph = removeCliqueEdges(getNetworkxGraphFromArray(self.skeletonStack))

    def setPrunedSkeletonOutput(self):
        # Prune unnecessary segments in crowded regions removed skeleton
        self.setNetworkGraph(findSkeleton=True)
        self.outputStack = getPrunedSkeleton(self.skeletonStack, self.graph)

    def getNetworkGraph(self):
        # Network graph of the final output skeleton stack
        self.setPrunedSkeletonOutput()
        self.outputGraph = removeCliqueEdges(getNetworkxGraphFromArray(self.outputStack))

    def saveSkeletonStack(self):
        # Save output skeletonized stack as series of pngs in the path under a subdirectory skeleton
        # in the input "path"
        self.setPrunedSkeletonOutput()
        saveStack(self.outputStack, self.path + "skeleton/")

    def getSegmentStatsBeforePruning(self):
        # stats before pruning the braches
        self.setNetworkGraph()
        self.statsBefore = SegmentStats(self.graph)
        self.statsBefore.setStats()

    def setSegmentStatsAfterPruning(self):
        # stats after pruning the braches
        self.getNetworkGraph()
        self.statsAfter = SegmentStats(self.outputGraph)
        self.statsAfter.setStats()
