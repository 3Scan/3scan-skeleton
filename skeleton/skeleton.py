from kesm.projects.KESMAnalysis.imgtools import loadStack

from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.thinVolume import getThinned
from skeleton.pruning import getPrunedSkeleton
from skeleton.unitwidthcurveskeleton import getShortestPathSkeleton


class Skeleton:
    def __init__(self, path, **kwargs):
        if type(path) is str:
            self.inputStack = loadStack(path).astype(bool)
        else:
            self.inputStack = path

    def setThinningOutput(self):
        self.thinnedStack = getThinned(self.inputStack)

    def setUnitWidthSkeletonOutput(self):
        self.setThinningOutput()
        self.skeletonStack = getShortestPathSkeleton(self.thinnedStack)

    def setNetworkGraph(self, findSkeleton=True):
        if findSkeleton:
            self.setUnitWidthSkeletonOutput()
        else:
            self.skeletonStack = self.inputStack
        self.graph = removeCliqueEdges(getNetworkxGraphFromarray(self.skeletonStack))

    def setPrunedSkeletonOutput(self):
        self.setNetworkGraph()
        self.outputStack = getPrunedSkeleton(self.skeletonStack, self.graph)

    def getNetworkGraph(self):
        self.setPrunedSkeletonOutput()
        self.outputGraph = removeCliqueEdges(getNetworkxGraphFromarray(self.outputStack))


