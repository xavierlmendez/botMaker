import pytest

from MlLib.mathDomain.algorithmImplementations.abstractGraphAlgorithm import SearchContext
from MlLib.mathDomain.algorithmImplementations.depthFirstSearch import DepthFirstSearch
from MlLib.mathDomain.graphBased.graphStructures import Graph


def build_graph():
    graph = Graph()
    a = graph.addNode(data="A")
    b = graph.addNode(data="B")
    c = graph.addNode(data="C")
    d = graph.addNode(data="D")
    e = graph.addNode(data="E")
    graph.addEdge(a, b)
    graph.addEdge(a, c)
    graph.addEdge(b, d)
    graph.addEdge(c, e)
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
    a = graph.addNode(data="A")
    b = graph.addNode(data="B")
    graph.addEdge(a, b)
    graph.addEdge(b, a)
    context = SearchContext(start_node_id=a, target_node_criteria=lambda node: False)
    assert DepthFirstSearch(graph).run(context) == [a, b]


def test_dfs_max_depth_limits_expansion():
    graph, a, b, c, d, e = build_graph()
    context = SearchContext(start_node_id=a, target_node_criteria=lambda node: False, max_depth=1)
    assert DepthFirstSearch(graph).run(context) == [a, b, c]
