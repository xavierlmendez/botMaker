import pytest

from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.algorithms.breadth_first_search import BreadthFirstSearch
from mllib.math.graph.graph_structures import Graph


def build_graph():
    graph = Graph()
    node_a = graph.add_node(data="A")
    node_b = graph.add_node(data="B")
    node_c = graph.add_node(data="C")
    node_d = graph.add_node(data="D")
    node_e = graph.add_node(data="E")

    graph.add_edge(node_a, node_b)
    graph.add_edge(node_a, node_c)
    graph.add_edge(node_b, node_d)
    graph.add_edge(node_c, node_e)

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
    a = graph.add_node(data="A")
    b = graph.add_node(data="B")
    graph.add_edge(a, b)
    graph.add_edge(b, a)

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
