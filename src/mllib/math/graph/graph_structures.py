from dataclasses import dataclass, field
from typing import Any

import networkx as nx


# https://dzone.com/articles/understanding-pythons-dataclass-decorator
@dataclass(frozen=True, slots=True)
class Edge:
    metadata = {
        "name": "Graph Edge",
        "description": "Simple edge representation connecting two node IDs with optional data.",
    }
    # TODO(BL-16): derive metadata by introspection
    u: int
    v: int
    data: Any = None


@dataclass(slots=True)
class GraphNode:
    metadata = {
        "name": "Graph Node",
        "description": "Graph node storing an ID, payload, and neighbor set.",
    }
    # TODO(BL-16): derive metadata by introspection
    node_id: int
    data: Any
    neighbors: set[int] = field(default_factory=set)


class Graph:
    def __init__(self, init_nodes=None):
        self.metadata = {
            "name": "Graph Structure",
            "description": "Undirected graph structure with nodes, edges, and adjacency helpers.",
        }
        # TODO(BL-16): derive metadata by introspection
        self.nodes: dict[int, GraphNode] = {}
        self.edges = []  # (Bi)directional edges will have to be on a directional graph implementation
        self.id_incrementor = 0  # Prefer this as searching burned identifiers will add to run time

        if init_nodes is not None:
            for node_id in init_nodes:
                self.nodes[node_id] = init_nodes[node_id]

    def add_node(self, data=None):
        self.id_incrementor += 1
        node_id = self.id_incrementor

        new_node = GraphNode(node_id, data)
        self.nodes[node_id] = new_node
        return node_id

    def add_edge(self, node_id_one, node_id_two):
        if node_id_one not in self.nodes:
            raise KeyError(f"Unknown node: {node_id_one}")
        if node_id_two not in self.nodes:
            raise KeyError(f"Unknown node: {node_id_two}")

        new_edge = Edge(node_id_one, node_id_two)
        if new_edge in self.edges:
            raise KeyError(f"Duplicate edge: {new_edge}")

        self.edges.append(new_edge)
        self.nodes[node_id_one].neighbors.add(node_id_two)
        self.nodes[node_id_two].neighbors.add(node_id_one)

    def remove_node(self, node_id):
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node: {node_id}")

        for connected_node_id in self.nodes[node_id].neighbors:
            self.nodes[connected_node_id].neighbors.remove(node_id)

        del self.nodes[node_id]

    def get_nx_graph(self):
        graph = nx.Graph()
        for node_id in self.nodes:
            graph.add_node(node_id)

        for node_id in self.nodes:
            for neighbor_id in self.nodes[node_id].neighbors:
                graph.add_edge(node_id, neighbor_id)

        return graph
