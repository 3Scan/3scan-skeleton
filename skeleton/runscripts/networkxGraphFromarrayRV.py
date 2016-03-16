import numpy as np
import networkx as nx

import itertools

from scipy.ndimage import convolve


"""
   program to look up adjacent elements and calculate degree
   this dictionary can be used for graph creation
   since networkx graph based on looking up the array and the
   adjacent coordinates takes long time. create a dict
   using dictOfIndicesAndAdjacentcoordinates. Refer the following link
   https://networkx.github.io/documentation/development/reference/generated/networkx.convert.from_dict_of_lists.html
"""

# permutations of (-1, 0, 1) in three/two dimensional tuple format
# representing 8 and 26 increments around a pixel at origin (0, 0, 0)
stepDirect = itertools.product((-1, 0, 1), repeat=3)
listStepDirect = list(stepDirect)
listStepDirect.remove((0, 0, 0))


def _getIncrements(configNumber):
    """
       takes in a configNumber and converts into
       binary sequence of 1s and 0s, returns the tuple
       increments corresponding to them
    """
    configNumber = np.int64(configNumber)
    neighborValues = [(configNumber >> digit) & 0x01 for digit in range(26)]
    return [neighborValue * increment for neighborValue, increment in zip(neighborValues, listStepDirect)]


def _setAdjacencylistarray(arr):
    """
        takes in an array and returns a dictionary with nonzero voxels/ pixels
        and their adjcent nonzero coordinates
    """
    template = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                        [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                        [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)
    arr = np.ascontiguousarray(arr, dtype=np.uint64)
    result = convolve(arr, template, mode='constant', cval=0)
    result[arr == 0] = 0
    dictOfIndicesAndAdjacentcoordinates = {}
    # list of nonzero tuples
    nonZeros = list(set(map(tuple, np.transpose(np.nonzero(arr)))))
    if np.sum(arr) == 1:
        # if there is just one nonzero elemenet there are no adjacent coordinates
        dictOfIndicesAndAdjacentcoordinates[nonZeros[0]] = []
        return dictOfIndicesAndAdjacentcoordinates
    else:
        for item in nonZeros:
            adjacentCoordinatelist = []
            for increments in _getIncrements(result[item]):
                if increments == (()):
                    continue
                adjCoord = np.array(item) + np.array(increments)
                adjacentCoordinatelist.append(tuple(adjCoord))
            dictOfIndicesAndAdjacentcoordinates[item] = adjacentCoordinatelist
    return dictOfIndicesAndAdjacentcoordinates


def getNetworkxGraphFromarray(skeleton):
    """

        if skeletonIm = True input is already skeletonized
        takes in a binary array of skeleton converts it to adictionary of lists
        of existing adjacent coordinates and forms a
        networkx graph from the dictionary

    """
    dictOfIndicesAndAdjacentcoordinates = _setAdjacencylistarray(skeleton)
    G = nx.from_dict_of_lists(dictOfIndicesAndAdjacentcoordinates)
    return G


def getGraphProperties(G, nameOfTheGraph):
    """
       takes in a networkx graph, it's name
       & prints out number of cycles, disjoint graphs in it
    """
    printString = '%'
    printString = printString.replace('%', nameOfTheGraph)
    numberOfCycles = len(nx.cycle_basis(G))
    disjointGraphs = list(nx.connected_component_subgraphs(G))
    numberOfDisjointGraphs = len(disjointGraphs)
    return numberOfCycles, numberOfDisjointGraphs, disjointGraphs

