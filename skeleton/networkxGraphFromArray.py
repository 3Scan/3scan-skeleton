import itertools
import numpy as np
import networkx as nx
import time

from scipy.ndimage import convolve


"""
program to look up adjacent elements and calculate degree
this dictionary can be used for graph creation
since networkx graph based on looking up the array and the
adjacent coordinates takes long time. create a dict
using dictOfIndicesAndAdjacentcoordinates.
(-1 -1 -1) (-1 0 -1) (-1 1 -1)
(-1 -1 0)  (-1 0 0)  (-1 1 0)
(-1 -1 1)  (-1 0 1)  (-1 1 1)

(0 -1 -1) (0 0 -1) (0 1 -1)
(0 -1 0)  (0 0 0)  (0 1 0)
(0 -1 1)  (0 0 1)  (0 1 1)

(1 -1 -1) (1 0 -1) (1 1 -1)
(1 -1 0)  (1 0 0)  (1 1 0)
(1 -1 1)  (1 0 1)  (1 1 1)
"""

# permutations of (-1, 0, 1) in three/two dimensional tuple format
# representing 8 and 26 increments around a pixel at origin (0, 0, 0)
# 2nd ordered neighborhood around a voxel/pixel
LISTSTEPDIRECTIONS3D = list(itertools.product((-1, 0, 1), repeat=3))
LISTSTEPDIRECTIONS3D.remove((0, 0, 0))

LISTSTEPDIRECTIONS2D = list(itertools.product((-1, 0, 1), repeat=2))
LISTSTEPDIRECTIONS2D.remove((0, 0))


def _getIncrements(configNumber, dimensions):
    """
    Return position of non zero voxels/pixels in the
    binary string of config number
    Parameters
    ----------
    configNumber : int64
        integer less than 2 ** 26

    dimensions: int
        number of dimensions, can only be 2 or 3

    Returns
    -------
    list
        a list of incremental direction of a non zero voxel/pixel

    Notes
    ------
    As in the beginning of the program, there are incremental directions
    around a voxel at origin (0, 0, 0) which are returned by this function.
    configNumber is a decimal number representation of 26 binary numbers
    around a voxel at the origin in a second ordered neighborhood
    """
    configNumber = np.int64(configNumber)
    if dimensions == 3:
        # convert decimal number to a binary string
        neighborValues = [(configNumber >> digit) & 0x01 for digit in range(26)]
        return [neighborValue * increment for neighborValue, increment in zip(neighborValues, LISTSTEPDIRECTIONS3D)]
    else:
        neighborValues = [(configNumber >> digit) & 0x01 for digit in range(8)]
        return [neighborValue * increment for neighborValue, increment in zip(neighborValues, LISTSTEPDIRECTIONS2D)]


def _setAdjacencyList(arr):
    """
    Return position of non zero voxels/pixels in the
    binary string of config number
    Parameters
    ----------
    arr : numpy array
        binary numpy array can only be 2D Or 3D

    Returns
    -------
    dictOfIndicesAndAdjacentcoordinates: Dictionary
        key is the nonzero coordinate in input "arr" and value
        is all the position of nonzero coordinates around it
        in it's second order neighborhood

    """
    dimensions = arr.ndim
    if dimensions == 3:
        # flipped 3D template in advance
        template = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                            [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                            [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)
    elif dimensions == 2:
        # 2 dimensions
        template = np.array([[2 ** 0, 2 ** 1, 2 ** 2], [2 ** 3, 0, 2 ** 4], [2 ** 5, 2 ** 6, 2 ** 7]])
        template = np.fliplr(np.flipud(template))
    else:
        assert 1 != 2, "array dimensions are not 2 or 3"
    # convert the binary array to a configuration number array of same size
    # by convolving with template
    arr = np.ascontiguousarray(arr, dtype=np.uint64)
    result = convolve(arr, template, mode='constant', cval=0)
    result[arr == 0] = 0
    dictOfIndicesAndAdjacentcoordinates = {}
    # list of nonzero tuples
    nonZeros = list(set(map(tuple, np.transpose(np.nonzero(arr)))))
    if np.sum(arr) == 1:
        # if there is just one nonzero element there are no adjacent coordinates
        dictOfIndicesAndAdjacentcoordinates[nonZeros[0]] = []
        return dictOfIndicesAndAdjacentcoordinates
    else:
        for item in nonZeros:
            adjacentCoordinatelist = [tuple(np.array(item) + np.array(increments))
                                      for increments in _getIncrements(result[item], dimensions) if increments != ()]
            dictOfIndicesAndAdjacentcoordinates[item] = adjacentCoordinatelist
    assert set(dictOfIndicesAndAdjacentcoordinates.keys()) == set(nonZeros)
    return dictOfIndicesAndAdjacentcoordinates


def getNetworkxGraphFromArray(arr):
    """
    Return a networkx graph from an array
    Parameters
    ----------
    arr : numpy array
        binary numpy array can only be 2D Or 3D

    Returns
    -------
    networkxGraph : Networkx graph
        graphical representation of the input array
    """
    assert arr.max() == 1
    assert arr.min() in [0, 1]
    start = time.time()
    dictOfIndicesAndAdjacentcoordinates = _setAdjacencyList(arr)
    G = nx.from_dict_of_lists(dictOfIndicesAndAdjacentcoordinates)

    # asserting no extra nodes other than nonzero coordinates on skeleton
    # are added in the graph
    assert set(dictOfIndicesAndAdjacentcoordinates.keys()) == set(G.nodes())
    print("time taken to obtain networkxGraph is %0.3f seconds" % (time.time() - start))
    return G


if __name__ == '__main__':
    # read points into array from the path and convert to a networkx graph
    shskel = np.load(input("please enter a path to your unit width voxelised skeleton"))
    networkxGraph = getNetworkxGraphFromArray(shskel)