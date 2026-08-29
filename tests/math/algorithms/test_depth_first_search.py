import pytest

from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.algorithms.depth_first_search import DepthFirstSearch
from mllib.math.graph.graph_structures import Graph


def build_graph():
    graph = Graph()
    a = graph.add_node(data="A")
    b = graph.add_node(data="B")
    c = graph.add_node(data="C")
    d = graph.add_node(data="D")
    e = graph.add_node(data="E")
    graph.add_edge(a, b)
    graph.add_edge(a, c)
    graph.add_edge(b, d)
    graph.add_edge(c, e)
    return graph, a, b, c, d, e


def test_dfs_requires_target_criteria():
    graph, a, *_ = build_graph()
    with pytest.raises(KeyError):
        DepthFirstSearch(graph).run(SearchContext(start_node_id=a))


def test_dfs_goes_deep_before_wide():
    graph, a, b, c, d, e = build_graph()
    context = SearchContext(start_node_id=a, target_node_criteria=lambda node: node.data == "E")
    assert DepthFirstSearch(graph).run(context) == [a, b, d, c, e]


def test_dfs_stops_on_target():
    graph, a, b, c, d, e = build_graph()
    context = SearchContext(start_node_id=a, target_node_criteria=lambda node: node.data == "D")
    assert DepthFirstSearch(graph).run(context) == [a, b, d]


def test_dfs_terminates_on_cycle_without_revisiting():
    graph = Graph()
    a = graph.add_node(data="A")
    b = graph.add_node(data="B")
    graph.add_edge(a, b)
    graph.add_edge(b, a)
    context = SearchContext(start_node_id=a, target_node_criteria=lambda node: False)
    assert DepthFirstSearch(graph).run(context) == [a, b]


def test_dfs_max_depth_limits_expansion():
    graph, a, b, c, d, e = build_graph()
    context = SearchContext(start_node_id=a, target_node_criteria=lambda node: False, max_depth=1)
    assert DepthFirstSearch(graph).run(context) == [a, b, c]
