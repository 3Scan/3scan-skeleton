import numpy as np

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from metrics.branchAngle import getBranchAngles
from skeleton.cliqueRemoving import removeCliqueEdges


def getDisjointTreesNoCycle3d(size=(10, 10, 10)):
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    crosPairgraph = getNetworkxGraphFromarray(crosPair)
    crosPairgraph = removeCliqueEdges(crosPairgraph)
    return crosPairgraph


def getTreeNoCycle2d(size=(5, 5)):
    cros = np.zeros(size, dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosGraph = getNetworkxGraphFromarray(cros)
    crosGraph = removeCliqueEdges(crosGraph)
    return crosGraph


def test_treeNocycle2d():
    crosGraph = getTreeNoCycle2d()
    ba = getBranchAngles(crosGraph, True, False)
    print(ba)
    angles = []
    for value in ba.values():
        angles.append(set(value))
    assert angles == [{90.0}]


def test_treeNocycle3d():
    crosPairgraph = getDisjointTreesNoCycle3d()
    ba = getBranchAngles(crosPairgraph, True, False)
    print(ba)
    angles = []
    for value in ba.values():
        angles.append(set(value))
    assert angles[0] == {90.0}
