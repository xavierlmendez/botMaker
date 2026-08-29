import networkx as nx
import pytest

from mllib.math.graph.graph_structures import Edge, Graph, GraphNode


def test_add_node_assigns_incrementing_ids_and_data():
    graph = Graph()

    first_id = graph.add_node(data="first")
    second_id = graph.add_node(data=None)

    assert first_id == 1
    assert second_id == 2
    assert isinstance(graph.nodes[first_id], GraphNode)
    assert graph.nodes[first_id].data == "first"
    assert graph.nodes[first_id].neighbors == set()
    assert graph.nodes[second_id].data is None


def test_add_edge_connects_neighbors_and_tracks_edge():
    graph = Graph()
    node_a = graph.add_node()
    node_b = graph.add_node()

    graph.add_edge(node_a, node_b)

    assert Edge(node_a, node_b) in graph.edges
    assert node_b in graph.nodes[node_a].neighbors
    assert node_a in graph.nodes[node_b].neighbors


def test_add_edge_rejects_unknown_or_duplicate_nodes():
    graph = Graph()
    node_a = graph.add_node()

    with pytest.raises(KeyError):
        graph.add_edge(node_a, 999)

    node_b = graph.add_node()
    graph.add_edge(node_a, node_b)

    with pytest.raises(KeyError):
        graph.add_edge(node_a, node_b)


def test_remove_node_disconnects_neighbors():
    graph = Graph()
    node_a = graph.add_node()
    node_b = graph.add_node()
    node_c = graph.add_node()

    graph.add_edge(node_a, node_b)
    graph.add_edge(node_a, node_c)

    graph.remove_node(node_a)

    assert node_a not in graph.nodes
    assert node_a not in graph.nodes[node_b].neighbors
    assert node_a not in graph.nodes[node_c].neighbors


def test_get_nx_graph_contains_nodes_and_edges():
    graph = Graph()
    node_a = graph.add_node()
    node_b = graph.add_node()
    node_c = graph.add_node()

    graph.add_edge(node_a, node_b)
    graph.add_edge(node_b, node_c)

    nx_graph = graph.get_nx_graph()

    assert isinstance(nx_graph, nx.Graph)
    assert nx_graph.number_of_nodes() == 3

    edge_sets = {frozenset(edge) for edge in nx_graph.edges()}
    assert edge_sets == {frozenset((node_a, node_b)), frozenset((node_b, node_c))}
