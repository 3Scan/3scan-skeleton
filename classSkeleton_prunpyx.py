# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:43:58 2019

@author: skyscan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:15:33 2019

@author: skyscan
"""
import os

import numpy as np
from scipy import ndimage
from skeleton.io_tools import loadStack, saveStack
from metrics.segmentStats import SegmentStats
from skeleton.networkx_graph_from_array import get_networkx_graph_from_array
# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA
from skeleton.thinVolume import get_thinned
import skeleton.pruning as pr

class Skeleton:
    def __init__(self, path, **kwargs):
        # initialize input array
        # path : can be an 3D binary array or a numpy(.npy) array
        # if path is a 3D volume saveSkeletonStack, saves series of
        # skeleton pngs in present directory
        if type(path) is str:
            print("path is a string")
            if path.endswith("npy"):
                # extract rootDir of path
                self.path = os.path.split(path)[0] + os.sep
                self.inputStack = np.load(path)
            else:
                self.path = path
                self.inputStack = loadStack(self.path).astype(bool)
                print(self.inputStack)
        else:
            self.path = os.getcwd()
            self.inputStack = path
        if kwargs != {}:
            aspectRatio = kwargs["aspectRatio"]
            self.inputStack = ndimage.interpolation.zoom(self.inputStack, zoom=aspectRatio, order=2, prefilter=False)

    def setThinningOutput(self, mode="reflect"):
        # Thinning output
        self.skeletonStack = get_thinned(self.inputStack, mode)
        saveStack(self.skeletonStack, self.path + "skeleton/")
    def setNetworkGraph(self, findSkeleton=False):
        # Network graph of the crowded region removed output
        # Generally the function expects a skeleton
        # and findSkeleton is False by default
        if findSkeleton is True:
            self.setThinningOutput()
        else:
            self.skeletonStack = self.inputStack
        self.graph = get_networkx_graph_from_array(self.skeletonStack)
       
    def setPrunedSkeletonOutput(self):
        # Prune unnecessary segments in crowded regions removed skeleton
        self.setNetworkGraph(findSkeleton=True)
        self.outputStack = pr.getPrunedSkeleton(self.skeletonStack, self.graph)
        saveStack(self.outputStack, self.path + "pruned/")
    def getNetworkGraph(self):
        # Network graph of the final output skeleton stack
        self.setPrunedSkeletonOutput()
        self.outputGraph = get_networkx_graph_from_array(self.outputStack)

    def saveSkeletonStack(self):
        # Save output skeletonized stack as series of pngs in the path under a subdirectory skeleton
        # in the input "path"
        self.setPrunedSkeletonOutput()
#        self.outputStack = self.outputStack.astype(np.uint8)
        saveStack(self.outputStack, self.path + "skeleton/")

    def getSegmentStatsBeforePruning(self):
        # stats before pruning the branches
        self.setNetworkGraph()
        self.statsBefore = SegmentStats(self.graph)
        self.statsBefore.setStats()

    def setSegmentStatsAfterPruning(self):
        # stats after pruning the branches
        self.getNetworkGraph()
        self.statsAfter = SegmentStats(self.outputGraph)
        self.statsAfter.setStats()
