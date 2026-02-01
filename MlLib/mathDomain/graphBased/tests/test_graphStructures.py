import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from MlLib.mathDomain.graphBased.graphStructures import Graph, GraphNode


def test_graph_node_defaults():
    node = GraphNode()

    assert node.nodeId == 0
    assert node.data is None
    assert node.outboundEdges == []
    assert node.edgeCount == 0


def test_graph_node_with_edges_sets_count():
    node = GraphNode(nodeIdentifier=3, edges=[1, 2, 4])

    assert node.nodeId == 3
    assert node.outboundEdges == [1, 2, 4]
    assert node.edgeCount == 3


def test_graph_node_add_and_remove_edges():
    node = GraphNode(nodeIdentifier=1)
    target = GraphNode(nodeIdentifier=5)

    node.addToEdge(target)
    assert node.outboundEdges == [5]
    assert node.edgeCount == 1

    node.removeEdge(target)
    assert node.outboundEdges == []
    assert node.edgeCount == 0


def test_graph_add_node_auto_identifier():
    graph = Graph()

    graph.addNode(data="a")
    graph.addNode(data="b")

    assert graph.nodeCount == 2
    assert [node.nodeId for node in graph.nodes] == [1, 2]
    assert [node.data for node in graph.nodes] == ["a", "b"]


def test_graph_add_node_custom_identifier():
    graph = Graph(useCustomIdentifier=True)

    graph.addNode(data="x", customIdentifier=10)
    graph.addNode(data="y", customIdentifier=42)

    assert graph.nodeCount == 2
    assert [node.nodeId for node in graph.nodes] == [10, 42]
    assert graph.identifierIncrementor == 0
