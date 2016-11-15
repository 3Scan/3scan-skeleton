import networkx as nx
import numpy as np
from skeleton.cliqueRemoving import removeCliqueEdges
from skeleton.networkxGraphFromArray import getNetworkxGraphFromArray
from skeleton.networkxGraphFromArray_tests import getGraphsWithCliques


def test_cliqueRemoval():
    # test edges after removing cliques are less and disjoint graphs remain same
    for networkxGraph, nonZeros in getGraphsWithCliques():
        disjointGraphsBefore = len(list(nx.connected_component_subgraphs(networkxGraph)))
        edgesBefore = networkxGraph.number_of_edges()
        networkxGraphAfter = removeCliqueEdges(networkxGraph)
        disjointGraphsAfter = len(list(nx.connected_component_subgraphs(networkxGraphAfter)))
        edgesAfter = networkxGraphAfter.number_of_edges()
        assert edgesBefore >= edgesAfter, (
            "clique removed, original graph have {}, {} edges respectively"
            .format(edgesAfter, edgesBefore))
        assert disjointGraphsAfter == disjointGraphsBefore, (
            "clique removed, original graph have {}, {} disjoint graphs respectively"
            .format(disjointGraphsAfter, disjointGraphsBefore))


def _getSpecialCaseGraph():
    SPECIAL_CASE_ARRAY = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                  [[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=bool)
    return getNetworkxGraphFromArray(SPECIAL_CASE_ARRAY)


def test_specialCaseCliqueRemoval():
    graph = _getSpecialCaseGraph()
    cliqueRemovedGraph = removeCliqueEdges(graph)
    assert graph.number_of_edges() - cliqueRemovedGraph.number_of_edges() == 1
