import numpy as np
# import skimage
from random import randint, seed

from skeleton.networkxGraphFromarray import listStepDirect, listStepDirect2d


"program generates treee like vessel structures"


def createBranching(syntheticVessels, origin3dVertex, increments, branching):
    # print(origin3dVertex)
    aShape = syntheticVessels.shape
    randomBranchdirections = [randint(1, len(increments) - 1) for p in range(1, branching + 1)]
    for index, direction in enumerate(randomBranchdirections):
        nextOrigin = tuple(np.array(origin3dVertex) + np.array(increments[direction]))
        # + np.array([5] * syntheticVessels.ndim)
        # print(nextOrigin)
        if outOfPixBOunds(nextOrigin, aShape) == 0:
            syntheticVessels[nextOrigin] = 1
        else:
            continue
    origin3dVertex = nextOrigin


def outOfPixBOunds(nearByCoordinate, aShape):
    onbound = 0
    for index, maxVal in enumerate(aShape):
        isAtBoundary = nearByCoordinate[index] in [0, maxVal] or nearByCoordinate[index] > maxVal
        if isAtBoundary:
            onbound = 1
            break
        else:
            continue
    return onbound


def generateSyntheticvessels(shape=(64, 64, 64), branching=2):
    syntheticVessels = np.zeros(shape, dtype=np.uint8)
    limit = np.sum([val - 1 for index, val in enumerate(shape)])
    originVessels = randint(1, limit)
    if len(shape) == 3:
        increments = listStepDirect
    else:
        increments = listStepDirect2d
    origin3dVertex = np.unravel_index(originVessels, dims=shape, order='C')
    for i in range(0, 1):
        createBranching(syntheticVessels, origin3dVertex, increments, branching)
    # if syntheticVessels.sum() == 0:
    #     generateSyntheticvessels(shape=shape, branching=branching)
    # selem = skimage.morphology.ball(radius=5)
    # syntheticVessels = skimage.morphology.binary_dilation(syntheticVessels, selem)
    return syntheticVessels


def generate_random_tree(node_list, idx=0, parent=None, depth=0, max_children=2, max_depth=2):
    """
    Build a list of nodes in a random tree up to a maximum depth.
        :param:    node_list    list of nodes in the tree; each node is a list with elements [idx, parent, depth]
        :param:    idx          int, the index of a node
        :param:    parent       int, the index of the node's parent
        :param:    depth        int, the distance of a node from the root
        :param:    max_children int, the maximum number of children a node can have
        :param:    max_depth    int, the maximum distance from the tree to the root
    """
    def add_children(node_list, idx, parent, depth, max_children):
        """Helper function for generate_random_tree() that adds n random child nodes to node_list."""
        n = randint(0, max_children)
        node_list.extend([[idx + i, parent, depth] for i in range(0, n)])
        return n

    if 0 <= depth < max_depth:
        # add a random number n of children
        n = add_children(node_list, idx, parent, depth, max_children)
        # for each new child, add new children
        [generate_random_tree(node_list, len(node_list), idx + i, depth + 1, max_children, max_depth) for i in range(0, n)]

    elif depth == max_depth:
        # add a random number of leaves
        add_children(node_list, idx, parent, depth, max_children)
        return


if __name__ == '__main__':
    syntheticVessels3D = generateSyntheticvessels()
    tree = [[0, None, 0]]  # the algorithm starts with a root node which has no parents and depth 0
    seed(0)
    generate_random_tree(node_list=tree, idx=len(tree), parent=0, depth=1, max_children=3, max_depth=5)
    tree.remove(tree[0])
    treeStructure = np.zeros(tuple(tree[-1]), dtype=np.uint8)
    aShape = treeStructure.shape
    for vertex in tree:
        vertex = tuple(vertex)
        if outOfPixBOunds(vertex, aShape) == 0:
            treeStructure[vertex] = 1
        else:
            continue
