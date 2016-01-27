import networkx as nx
import numpy as np
from skimage import morphology
from math import pow, sqrt


def getDim(maxIndex, dims=3):
    for i in range(0, 1000):
        if maxIndex < pow(i, dims):
            return i
            break
        else:
            continue


def getTreeArray(branching=3, height=1, dims=3):
    if dims == 3:
        selem = morphology.ball(1)
    else:
        selem = morphology.disk(1)
    treeGraph = nx.balanced_tree(branching, height)
    maxIndex = max(treeGraph.nodes())
    dim = getDim(maxIndex, dims)
    dims = tuple([dim] * dims)
    treeArray = np.zeros(dims, dtype=np.uint8)
    for edge in treeGraph.edges():
        nzi1 = np.unravel_index(edge[0], dims=dims)
        nzi2 = np.unravel_index(edge[1], dims=dims)
        treeArray[tuple(nzi1)] = 1; treeArray[tuple(nzi2)] = 1;
        direction = (np.array(nzi1) - np.array(nzi2))
        dist = sqrt(np.sum((np.array(nzi1) - np.array(nzi2)) ** 2))
        normalized = direction / dist
        if dist < 1:
            for i in range(0, dist):
                treeArray[tuple((normalized * i) + nzi1)] = 1
    synVessels = morphology.binary_dilation(treeArray, selem)
    return synVessels, treeArray


if __name__ == '__main__':
    twoDVesselarray, twoDTreearray = getTreeArray(5, 2, 2)
