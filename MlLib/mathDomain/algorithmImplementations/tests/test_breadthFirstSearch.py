import pytest

from botMaker.MlLib.mathDomain.algorithmImplementations.abstractGraphAlgorithm import SearchContext
from botMaker.MlLib.mathDomain.algorithmImplementations.breadthFirstSearch import BreadthFirstSearch
from botMaker.MlLib.mathDomain.graphBased.graphStructures import Graph


def build_graph():
    graph = Graph()
    node_a = graph.addNode(data="A")
    node_b = graph.addNode(data="B")
    node_c = graph.addNode(data="C")
    node_d = graph.addNode(data="D")
    node_e = graph.addNode(data="E")

    graph.addEdge(node_a, node_b)
    graph.addEdge(node_a, node_c)
    graph.addEdge(node_b, node_d)
    graph.addEdge(node_c, node_e)

    return graph, node_a, node_b, node_c, node_d, node_e


def test_bfs_traversal_order_without_target():
    graph, node_a, node_b, node_c, node_d, node_e = build_graph()

    context = SearchContext(start_node_id=node_a)
    with pytest.raises(KeyError):
        BreadthFirstSearch(graph).run(context)


def test_bfs_stops_on_target_criteria_match():
    graph, node_a, node_b, node_c, node_d, node_e = build_graph()

    context = SearchContext(start_node_id=node_a, target_node_criteria= lambda node: node.data == "C")
    traversal = BreadthFirstSearch(graph).run(context)

    assert traversal == [node_a, node_b, node_c]

