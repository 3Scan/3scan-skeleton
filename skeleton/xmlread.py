import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
from xml.dom import minidom
from skeleton.MouseBrainIndianinkTamu import skeletonizeAndSave
from skeleton.objWrite import getObjWriteWithradius
from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline


def xmlToNetworkx():
    pathToXml = input("enter the path to xml file you want to convert to networkx graph------")
    doc = minidom.parse(pathToXml)
    positions = doc.getElementsByTagName("tup")
    positionids = doc.getElementsByTagName("node")
    edgeIds = doc.getElementsByTagName("edge")
    mapping = {}
    dictOfNodesAndRadiusgt = {}
    ids = {}; idsVertices = {}
    for i, position in enumerate(positions):
        floats = position.getElementsByTagName("float")
        vertex = tuple((float(floats[0].firstChild.nodeValue), float(floats[1].firstChild.nodeValue), float(floats[2].firstChild.nodeValue)))
        mapping[i] = vertex
        ids[(positionids[i].getAttribute("id"))] = i
        idsVertices[(positionids[i].getAttribute("id"))] = vertex

    ebunch = []
    childOfEdgenodes = edgeIds[0].childNodes
    l2 = childOfEdgenodes[3].childNodes
    dictOfNodesAndRadiusgt[mapping[0]] = float(l2[1].firstChild.nodeValue)
    for index, edge in enumerate(edgeIds):
        fromVert = edgeIds[index].getAttribute("from")
        toVert = edgeIds[index].getAttribute("to")
        ebunch.append(tuple((ids[fromVert], ids[toVert])))
        childOfEdgenodes = edgeIds[index].childNodes
        l2 = childOfEdgenodes[3].childNodes
        dictOfNodesAndRadiusgt[idsVertices[toVert]] = float(l2[1].firstChild.nodeValue)

    G = nx.Graph()
    G.add_edges_from(ebunch)

    networkGraphFloatNodes = nx.relabel_nodes(G, mapping, False)
    return networkGraphFloatNodes, dictOfNodesAndRadiusgt


def dict_to_image(dictAdjacency, shape=(101, 101, 101)):
    im = np.zeros((101, 101, 101), dtype=bool)
    for index in dictGraph.keys():
        im[index] = True
    return im


if __name__ == '__main__':
    networkGraphFloatNodes, dictOfNodesAndRadiusgt = xmlToNetworkx()
    nx.draw(networkGraphFloatNodes, node_size=[0.5] * networkGraphFloatNodes.number_of_nodes())
    disjointGraphs = len(list(nx.connected_component_subgraphs(networkGraphFloatNodes)))
    print("disjointGraphs in the original centerline is", disjointGraphs)
    # plt.show()
    dictGraph = nx.to_dict_of_lists(networkGraphFloatNodes)
    im = dict_to_image(dictGraph)
    getObjWriteWithradius(networkGraphFloatNodes, "/media/pranathi/A336-5F43/image1/VascuSynth-GT.obj", dictOfNodesAndRadiusgt)
    shortestPathSkel, boundaryIm = skeletonizeAndSave(contrast=True, zoom=False)
    dictOfNodesAndRadiust, distIm = getRadiusByPointsOnCenterline(shortestPathSkel, boundaryIm)
    getObjWriteWithradius(shortestPathSkel, "/media/pranathi/A336-5F43/image1/VascuSynth-T2.obj", dictOfNodesAndRadiust)
