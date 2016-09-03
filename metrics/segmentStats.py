import itertools
import time
import numpy as np
import networkx as nx


"""
Find the segments, lengths and tortuosity of a networkx graph by
        1) go through each of the disjoint graphs
        2) decide if it is one of the following a) line
        b) cycle c) acyclic tree like structure d) cyclic tree like structure
        e) single node
        3) Find all the paths in a given disjoint graph from a point whose degree is greater
        than 2 (branch point) to a point whose degree equal to one (end point)
        4) calculate distance between edges in each path and displacement to find curve length and
        curve displacement to find tortuosity, hausdorff dimension and contraction
        5) Remove all the edges in this path once they are traced
"""


class SegmentStats:
    """
    Find statistics on a networkx graph of a skeleton
    Parameters
    ----------
    Graph : networkx graph
       networkx graph of a skeleton

    Examples
    ------
           SegmentStats.countDict - A dictionary with key as the node(branch or end point)
                                          and number of branches connected to it

           SegmentStats.lengthDict - A dictionary with key as the (branching index (nth branch from the start node),
                                           start node, end node of the branch) value = length of the branch

           SegmentStats.tortuosityDict - A dictionary with key as the (branching index (nth branch from the start node),
                                             start node, end node of the branch) value = tortuosity of the branch

           SegmentStats.totalSegments - total number of branches (segments between branch and branch points, branch and end points)

           SegmentStats.typeGraphdict - A dictionary with the nth disjoint graph as the key and the type of graph as the value

           SegmentStats.avgBranching - Average branching index (number of branches at a branch point) of the network

           SegmentStats.endP - nodes with only one other node connected to them

           SegmentStats.branchP - nodes with more than 2 nodes connected to them

           SegmentStats.contractionDict - A dictionary with key as the (branching index (nth branch from the start node), start node,
                                                end node of the branch) value = contraction of the branch

           SegmentStats.hausdorffDimensionDict - A dictionary with key as the (branching index (nth branch from the start node), start node,
                                                       end node of the branch) value = hausdorff dimension of the branch


    Notes
    --------

    tortuosity = curveLength / curveDisplacement
    contraction = curveDisplacement / curveLength (better becuase there is no change of instability (undefined) in case of cycles)
    Hausdorff Dimension = np.log(curveLength) / np.log(curveDisplacement) https://en.wikipedia.org/wiki/Hausdorff_dimension
    Type of subgraphs:
    0 = if graph a single node
    1 = if graph is a single cycle
    2 = line (highest degree in the subgraph is 2)
    3 = undirected acyclic graph
    4 = undirected cyclic graph

    """
    def __init__(self, networkxGraph):
        self.networkxGraph = networkxGraph
        # intitialize all the instance variables of SegmentStats class
        self.contractionDict = {}
        self.countDict = {}
        self.cycleInfo = {}
        self.hausdorffDimensionDict = {}
        self.isolatedEdgeInfo = {}
        self.lengthDict = {}
        self.nodeDegreedict = {}
        self.tortuosityDict = {}
        self.typeGraphdict = {}

        self.cycles = 0
        self.edgesUntraced = 0
        self.totalSegments = 0

        self.branchpoints = []
        self.endpoints = []
        self.sortedSegments = []
        self.visitedSources = []
        self.visitedPaths = []
        # list of disjointgraphs
        self.disjointGraphs = list(nx.connected_component_subgraphs(self.networkxGraph))

    def _getLengthAndRemoveTracedPath(self, subGraphskeleton, path, isCycle=0, remove=1):
        """
        Find length of a path as distance between nodes in it
        and Remove edges belonging to "path" from "subGraphskeleton"
        Parameters
        ----------
        subGraphskeleton : Networkx graph

        path : list
           list of nodes in the path

        isCycle : boolean
           Specify if path is a cycle or not

        Returns
        -------
        subGraphskeleton : Networkx graph
            graph changed inplace with visited nodes removed

        length : float
            Length of path

        Notes
        ------
        given a visited path in variable path, the edges in the
        path are removed in the graph
        if cycle = 1 , the given path belongs to a cycle,
        so an additional edge is formed between the last and the
        first node to form a closed cycle/ path and is removed
        and remove the edges

        Examples
        --------
        >>> path = [(1, 2, 3), (1, 3, 3), (1, 4, 5)]
        >>> length = _getLengthAndRemoveTracedPath(subGraphskeleton, path)
        >>> length
        3.2360679774997898
        >>> lengthCycle = _getLengthAndRemoveTracedPath(subGraphskeleton, path, 1)
        6.0644951022459797
        """
        length = 0
        shortestPathedges = []
        if isCycle:
            for index, item in enumerate(path):
                if index + 1 < len(path):
                    item2 = path[index + 1]
                else:
                    item2 = path[0]
                if remove:
                    shortestPathedges.append(tuple((item, item2)))
                length += np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2))
        else:
            for index, item in enumerate(path[:-1]):
                item2 = path[index + 1]
                length += np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2))
                if remove:
                    shortestPathedges.append(tuple((item, item2)))
        if remove:
            subGraphskeleton.remove_edges_from(shortestPathedges)
        return length

    def _checkSegmentNotTraced(self, simplePath, sortedSegments):
        """
        Find if simplePath is not already been visited for sorted segments of a cycle

        Parameters
        ----------
        simplePath: list
           list of nodes in the path

        sortedSegments: list of lists
           list of paths already traced

        Returns
        -------
        check: Boolean
            if simplePath is in sortedSegments check = 0 else check = 1

        """
        check = 1
        for path in sortedSegments:
            if len(set(path) & set(simplePath)) > 2:
                check = 0
                break
        return check

    def _singleCycle(self, subGraphskeleton, cycle):
        """
        Find statistics of a single cycle of a disjoint graph

        Parameters
        ----------
        subGraphskeleton: Networkx Graph
            disjoint subgraph of the skeleton's networkx graph

        cycle: list
            list of nodes in the cycle

        Returns
        -------
        subGraphskeleton : Networkx graph
            graph changed inplace with visited nodes removed

        cycleInfo : Dictionary
            class instance variable changed in place with key as nth cycle, value as list with
            [number of branch points on cycle, length of the cycle]

        countDict: Dictionary
            class instance variable changed in place with key as source node,
            value is the number of segments attached to the source node

        lengthDict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the length of the segment

        tortuosityDict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the tortuosity of the segment (Nan but saved as zero)

        contractionDictdict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the contraction of the segment (0) since it is a cycle
        """
        sourceOnCycle = cycle[0]
        if sourceOnCycle not in self.visitedSources:
            self.countDict[sourceOnCycle] = 1
            self.visitedSources.append(sourceOnCycle)
        else:
            self.countDict[sourceOnCycle] += 1
        curveLength = self._getLengthAndRemoveTracedPath(subGraphskeleton, cycle, 1)
        self.cycleInfo[self.cycles] = [0, curveLength]
        self.lengthDict[self.countDict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = curveLength
        self.tortuosityDict[self.countDict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0
        self.contractionDict[self.countDict[sourceOnCycle], sourceOnCycle, cycle[len(cycle) - 1]] = 0

    def _cyclicTree(self, subGraphskeleton, cycleList):
        """
        Find statistics of a cyclic tree of a disjoint graph

        Parameters
        ----------
        subGraphskeleton: Networkx Graph
            disjoint subgraph of the skeleton's networkx graph

        cyclesList: list of lists
            list of list of nodes in the cycles

        cycles: int
            number of cycles in the graph so far
        Returns
        -------
        subGraphskeleton : Networkx graph
            graph changed inplace with visited nodes removed

        cycleInfo : Dictionary
            class instance variable changed in place with key as nth cycle, value as list with
            [number of branch points on cycle, length of the cycle]

        countDict : Dictionary
            class instance variable changed in place with key as source node,
            value is the number of segments attached to the source node

        lengthDict : Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the length of the segment

        tortuosityDict : Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the tortuosity of the segment

        contractionDictdict : Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the contraction of the segment

        hausdorffDimensionDict : Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the hausdorff dimension of the segment
        """
        for nthcycle, cycle in enumerate(cycleList):
            nodeDegreedictFilt = {key: value for key, value in self.nodeDegreedict.items() if key in cycle}
            branchpoints = [k for (k, v) in nodeDegreedictFilt.items() if v != 2 and v != 1]
            sourceOnCycle = branchpoints[0]
            if len(branchpoints) == 1:
                self._singleCycle(subGraphskeleton, cycle)
            else:
                for point in cycle:
                    if point in branchpoints:
                        if nx.has_path(subGraphskeleton, source=sourceOnCycle, target=point) and sourceOnCycle != point:
                            simplePath = nx.shortest_path(subGraphskeleton, source=sourceOnCycle, target=point)
                            sortedSegment = sorted(simplePath)
                            if (sortedSegment not in self.sortedSegments and
                               sum([1 for pathpoint in simplePath if pathpoint in branchpoints]) == 2 and
                               self._checkSegmentNotTraced(simplePath, self.sortedSegments)):
                                if sourceOnCycle not in self.visitedSources:
                                    self.countDict[sourceOnCycle] = 1
                                    self.visitedSources.append(sourceOnCycle)
                                else:
                                    self.countDict[sourceOnCycle] += 1
                                curveLength = self._getLengthAndRemoveTracedPath(subGraphskeleton, simplePath, isCycle=0, remove=0)
                                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnCycle) - np.array(point)) ** 2))
                                self.lengthDict[self.countDict[sourceOnCycle], sourceOnCycle, point] = curveLength
                                self.tortuosityDict[self.countDict[sourceOnCycle], sourceOnCycle, point] = curveLength / curveDisplacement
                                self.contractionDict[self.countDict[sourceOnCycle], sourceOnCycle, point] = curveDisplacement / curveLength
                                if curveDisplacement != 0.0:
                                    if np.log(curveDisplacement) != 0.0:
                                        self.hausdorffDimensionDict[self.countDict[sourceOnCycle], sourceOnCycle, point] = np.log(curveLength) / np.log(curveDisplacement)
                                self.visitedPaths.append(simplePath)
                                self.sortedSegments.append(sortedSegment)
                        sourceOnCycle = point
            self.cycleInfo[self.cycles + nthcycle] = [len(branchpoints), self._getLengthAndRemoveTracedPath(subGraphskeleton, cycle, 1, remove=0)]
        for path in self.visitedPaths:
            self._getLengthAndRemoveTracedPath(subGraphskeleton, simplePath, isCycle=0, remove=1)
        self.cycles += len(cycleList)

    def _tree(self, subGraphskeleton):
        """
        Find statistics of a tree like structure of disjoint graph

        Parameters
        ----------
        subGraphskeleton: Networkx Graph
            disjoint subgraph of the skeleton's networkx graph

        cycle: list
            list of nodes in the cycle

        Returns
        -------
        subGraphskeleton : Networkx graph
            graph changed inplace with visited nodes removed

        cycleInfo : Dictionary
            class instance variable changed in place with key as nth cycle, value as list with
            [number of branch points on cycle, length of the cycle]

        countDict: Dictionary
            class instance variable changed in place with key as source node,
            value is the number of segments attached to the source node

        lengthDict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the length of the segment

        tortuosityDict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the tortuosity of the segment (Nan but saved as zero)

        contractionDictdict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the contraction of the segment (0) since it is a cycle
        """
        listOfPerms = list(itertools.product(self.branchpoints, self.endpoints))
        for sourceOnTree, item in listOfPerms:
            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                for simplePath in simplePaths:
                    if sum([1 for point in simplePath if point in self.branchpoints]) == 1:
                        if sourceOnTree not in self.visitedSources:
                            self.countDict[sourceOnTree] = 1
                            self.visitedSources.append(sourceOnTree)
                        else:
                            self.countDict[sourceOnTree] = self.countDict[sourceOnTree] + 1
                        curveLength = self._getLengthAndRemoveTracedPath(subGraphskeleton, simplePath)
                        curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                        self.lengthDict[self.countDict[sourceOnTree], sourceOnTree, item] = curveLength
                        self.tortuosityDict[self.countDict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                        self.contractionDict[self.countDict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                        if curveDisplacement != 0.0:
                            if np.log(curveDisplacement) != 0.0:
                                self.hausdorffDimensionDict[self.countDict[sourceOnTree], sourceOnTree, item] = np.log(curveLength) / np.log(curveDisplacement)

    def _branchToBranch(self, subGraphskeleton):
        """
        Find statistics of a single cycle of a disjoint graph

        Parameters
        ----------
        subGraphskeleton: Networkx Graph
            disjoint subgraph of the skeleton's networkx graph

        cycle: list
            list of nodes in the cycle

        Returns
        -------
        subGraphskeleton : Networkx graph
            graph changed inplace with visited nodes removed

        countDict: Dictionary
            class instance variable changed in place with key as source node,
            value is the number of segments attached to the source node

        lengthDict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the length of the segment

        tortuosityDict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the tortuosity of the segment (Nan but saved as zero)

        contractionDictdict: Dictionary
            class instance variable changed in place with key as index of the segment from the source node,
            source node, target node, value is the contraction of the segment (0) since it is a cycle

        """
        listOfPerms = list(itertools.permutations(self.branchpoints, 2))
        for sourceOnTree, item in listOfPerms:
            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                for simplePath in simplePaths:
                    simplePath
                    if sum([1 for point in simplePath if point in self.branchpoints]) == 2:
                        if sourceOnTree not in self.visitedSources:
                            self.countDict[sourceOnTree] = 1
                            self.visitedSources.append(sourceOnTree)
                        else:
                            self.countDict[sourceOnTree] = self.countDict[sourceOnTree] + 1
                        curveLength = self._getLengthAndRemoveTracedPath(subGraphskeleton, simplePath)
                        curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                        self.lengthDict[self.countDict[sourceOnTree], sourceOnTree, item] = curveLength
                        self.tortuosityDict[self.countDict[sourceOnTree], sourceOnTree, item] = curveLength / curveDisplacement
                        self.contractionDict[self.countDict[sourceOnTree], sourceOnTree, item] = curveDisplacement / curveLength
                        if curveDisplacement != 0.0:
                            if np.log(curveDisplacement) != 0.0:
                                self.hausdorffDimensionDict[self.countDict[sourceOnTree], sourceOnTree, item] = np.log(curveLength) / np.log(curveDisplacement)

    def setStats(self):
        start = time.time()
        self.totalSegmentLength = sum([np.sqrt(np.sum((np.array(item) - np.array(item2)) ** 2)) for item, item2 in self.networkxGraph.edges()])
        for ithDisjointGraph, subGraphskeleton in enumerate(self.disjointGraphs):
            startDisjoint = time.time()
            numNodes = subGraphskeleton.number_of_nodes()
            print("     processing {}th disjoint graph with {} nodes".format(ithDisjointGraph, numNodes))
            nodes = subGraphskeleton.nodes()
            if len(nodes) == 1:
                # if it is a single node
                self.typeGraphdict[ithDisjointGraph] = 0
            else:
                # if there are more than one nodes decide what kind of subgraph it is
                # if it has cycles alone, or a straight line or a undirected cyclic/acyclic graph
                cycleList = nx.cycle_basis(subGraphskeleton)
                cycleCount = len(cycleList)
                if cycleCount != 0:
                    self.typeGraphdict[ithDisjointGraph] = 3
                else:
                    self.typeGraphdict[ithDisjointGraph] = 4
                self.nodeDegreedict = nx.degree(subGraphskeleton)
                degreeList = list(self.nodeDegreedict.values())
                endPointdegree = min(degreeList)
                branchPointdegree = max(degreeList)
                if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount == 1:
                    # if the maximum degree is equal to minimum degree it is a circle, set
                    # tortuosity to infinity (NaN) set to zero here
                    self.typeGraphdict[ithDisjointGraph] = 1
                    cycle = cycleList[0]
                    self._singleCycle(subGraphskeleton, cycle)
                    self.cycles = self.cycles + 1
                elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
                    # disjoint line or a bent line at 45 degrees appearing as dichtonomous tree but an error due to
                    # improper binarization, so remove them and do not account for statistics
                    edges = subGraphskeleton.edges()
                    listOfPerms = list(itertools.combinations(nodes, 2))
                    if type(nodes[0]) == int:
                        modulus = [[start - end] for start, end in listOfPerms]
                        dists = [abs(i[0]) for i in modulus]
                    else:
                        dims = len(nodes[0])
                        modulus = [[start[dim] - end[dim] for dim in range(0, dims)] for start, end in listOfPerms]
                        dists = [sum(modulus[i][dim] * modulus[i][dim] for dim in range(0, dims)) for i in range(0, len(modulus))]
                    if len(list(nx.articulation_points(subGraphskeleton))) == 1 and set(dists) != 1:
                        # each node is connected to one or two other nodes which are not a distance of 1 implies there is a
                        # one branch point with two end points in a single dichotomous tree"""
                        for sourceOnTree, item in listOfPerms:
                            if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                                simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                                simplePath = simplePaths[0]
                                if sum([1 for point in simplePath if point in nodes]) == 2:
                                    curveLength = self._getLengthAndRemoveTracedPath(subGraphskeleton, simplePath, 0)
                                    self.isolatedEdgeInfo[sourceOnTree, item] = curveLength
                    else:
                        # each node is connected to one or two other nodes implies it is a line,
                        endpoints = [k for (k, v) in self.nodeDegreedict.items() if v == 1]
                        sourceOnLine = endpoints[0]
                        targetOnLine = endpoints[1]
                        simplePath = nx.shortest_path(subGraphskeleton, source=sourceOnLine, target=targetOnLine)
                        curveLength = self._getLengthAndRemoveTracedPath(subGraphskeleton, simplePath, 0, remove=0)
                        self.isolatedEdgeInfo[sourceOnLine, targetOnLine] = curveLength
                        subGraphskeleton.remove_edges_from(edges)
                        self.typeGraphdict[ithDisjointGraph] = 2
                else:
                    # cyclic or acyclic tree
                    if cycleCount != 0:
                        self._cyclicTree(subGraphskeleton, cycleList)
                    # sorted branch and end points in trees
                    self.branchpoints = [k for (k, v) in self.nodeDegreedict.items() if v != 2 and v != 1]
                    self.endpoints = [k for (k, v) in self.nodeDegreedict.items() if v == 1]
                    self.branchpoints.sort()
                    self.endpoints.sort()
                    self._tree(subGraphskeleton)
                    if subGraphskeleton.number_of_edges() != 0:
                        self._branchToBranch(subGraphskeleton)
                self.edgesUntraced += subGraphskeleton.number_of_edges()
                assert subGraphskeleton.number_of_edges() == 0, "edges not removed are %i" % self.edgesUntraced
                print("     %ith disjoint graph took %0.2f seconds" % (ithDisjointGraph, time.time() - startDisjoint))
        self.totalSegments = len(self.lengthDict)
        listCounts = list(self.countDict.values())
        self.avgBranching = 0
        if len(self.countDict) != 0:
            self.avgBranching = sum(listCounts) / len(self.countDict)
        ndd = nx.degree(self.networkxGraph)
        self.countEndPoints = sum([1 for key, value in ndd.items() if value == 1])
        self.countBranchPoints = sum([1 for key, value in ndd.items() if value > 2])
        print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - start))

