import pytest

from mllib.mathDomain.algorithmImplementations.abstractGraphAlgorithm import SearchContext
from mllib.mathDomain.algorithmImplementations.breadthFirstSearch import BreadthFirstSearch
from mllib.mathDomain.graphBased.graphStructures import Graph


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

    context = SearchContext(
        start_node_id=node_a, target_node_criteria=lambda node: node.data == "C"
    )
    traversal = BreadthFirstSearch(graph).run(context)

    assert traversal == [node_a, node_b, node_c]


def test_bfs_terminates_on_cycle_without_revisiting():
    graph = Graph()
    a = graph.addNode(data="A")
    b = graph.addNode(data="B")
    graph.addEdge(a, b)
    graph.addEdge(b, a)

    context = SearchContext(start_node_id=a, target_node_criteria=lambda node: False)
    traversal = BreadthFirstSearch(graph).run(context)

    assert traversal == [a, b]


def test_bfs_max_depth_limits_expansion():
    graph, node_a, node_b, node_c, node_d, node_e = build_graph()

    context = SearchContext(
        start_node_id=node_a, target_node_criteria=lambda node: False, max_depth=1
    )
    traversal = BreadthFirstSearch(graph).run(context)

    assert traversal == [node_a, node_b, node_c]
