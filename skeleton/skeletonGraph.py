import itertools
import numpy as np
import networkx as nx
import time

from skimage.morphology import skeletonize as getskeletonize2d
from KESMAnalysis.skeleton.convOptimize import getSkeletonize3D
from KESMAnalysis.skeleton.unitwidthcurveskeleton_Edited1 import getShortestPathskeleton


# def outOfPixBOunds(neighborCoordinate, aShape):
#     zMax, yMax, xMax = aShape
#     isAtZBoundary = neighborCoordinate[0] == 0 or neighborCoordinate[0] == zMax
#     isAtYBoundary = neighborCoordinate[1] == 0 or neighborCoordinate[1] == yMax
#     isAtXBoundary = neighborCoordinate[2] == 0 or neighborCoordinate[2] == xMax
#     if isAtXBoundary or isAtYBoundary or isAtZBoundary:
#         return 1
#     else:
#         return 0


def binaryImageToGraph(skeletonIm):
    """
    convert 2d/3d binary array to skeleton
    and then to a graph
    """
    skeletonIm = np.uint8(skeletonIm)
    # Basic binary array sanity checks
    assert skeletonIm.ndim in [2, 3]
    assert skeletonIm.dtype == np.uint8
    assert skeletonIm.min() >= 0
    assert skeletonIm.max() <= 1

    if skeletonIm.ndim == 2:
        npResult = getskeletonize2d(skeletonIm)
        npResult = np.lib.pad(npResult, 1, 'constant', constant_values=0)
        stepDirect = itertools.product((-1, 0, 1), repeat=2)
        listStepDirect = list(stepDirect)
        listStepDirect.remove((0, 0))
        startt = time.time()
        g = nx.Graph()
        aShape = npResult.shape
        nZi = list(set(map(tuple, list(np.transpose(np.nonzero(npResult))))))
        for indices in nZi:
            for d in listStepDirect:
                nearByCoordinate = tuple(np.array(indices) + np.array(d))
                if nearByCoordinate[0] == (aShape[0] or 0) or nearByCoordinate[1] == (aShape[1] or 0):
                    continue
                if npResult[nearByCoordinate] == 1:
                    g.add_edge(indices, nearByCoordinate)
    else:
        npResult = getSkeletonize3D(skeletonIm)
        npResult = getShortestPathskeleton(npResult)
        # print("Sum is ", np.sum(npResult))
        npResult = np.lib.pad(npResult, 1, 'constant', constant_values=0)
        stepDirect = itertools.product((-1, 0, 1), repeat=3)
        listStepDirect = list(stepDirect)
        listStepDirect.remove((0, 0, 0))
        startt = time.time()
        g = nx.Graph()
        aShape = npResult.shape
        nZi = list(set(map(tuple, list(np.transpose(np.nonzero(npResult))))))
        if np.sum(npResult) != 1:
            for indices in nZi:
                for d in listStepDirect:
                    nearByCoordinate = tuple(np.array(indices) + np.array(d))
                    if nearByCoordinate[0] == aShape[0] or nearByCoordinate[1] == (aShape[1] or 0) or nearByCoordinate[2] == (aShape[2] or 0):
                        continue
                    if npResult[nearByCoordinate] == 1:
                        g.add_edge(indices, nearByCoordinate)
        else:
            g.add_edge(nZi[0], (0, 0, 1))
            print(g.number_of_edges(), g.number_of_nodes())
    h = g.to_undirected()
    print("time taken to obtain the graph is", time.time() - startt, "seconds")
    return h


if __name__ == "__main__":
    n = np.array([[0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [1, 1, 0, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)
    n3d = np.zeros((5, 5, 5), dtype=np.uint8)
    n3d[0] = n; n3d[1] = n; n3d[2] = n
    g, c = binaryImageToGraph(n3d)
    print("graph 1 drawn")

    randomTest = np.random.randint(2, size=(25, 25, 25))
    r, c2 = binaryImageToGraph(np.uint8(randomTest))
    print("graph 2 drawn")
    threeTestHard = np.array([[[1, 1, 1],
                               [0, 0, 1],
                               [1, 1, 1]],
                              [[1, 0, 0],
                               [0, 0, 0],
                               [1, 0, 0]],
                              [[1, 1, 1],
                               [0, 0, 1],
                               [1, 1, 1]]
                              ], dtype=np.uint8)
    snake, c3 = binaryImageToGraph(threeTestHard)
    print("graph 3 drawn")
    graph3d = np.zeros_like(n3d)
    pdict = {(1, 1, 2): 0, (1, 2, 1): 1, (1, 2, 3): 1, (1, 3, 2): 2, (1, 3, 3): 3}
    # vals = list(pdict.values())
    # keyss = list(pdict.keys())
    # for
    # graph3d[item1] = item2

    #     # nonZero = np.transpose(np.nonzero(skeletonIm))
    # # src = tuple(nonZero[0])
    # # dest = tuple(nonZero[-1])
    # # paths = [p for p in nx.all_shortest_paths(g, source=src, target=dest)]
    # paths = nx.all_pairs_shortest_path(g)
    # # G = nx.from_dict_of_dicts(paths)
