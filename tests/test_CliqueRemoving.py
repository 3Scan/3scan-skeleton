import networkx as nx
from skeleton.cliqueRemoving import removeCliqueEdges
from tests.test_NetworkxGraphFromArray import getWithCliquesGraphs


def test_cliqueRemoval():
    # test edges after removing cliques are less and disjoint graphs remain same
    for networkxGraph, nonZeros in getWithCliquesGraphs():
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

