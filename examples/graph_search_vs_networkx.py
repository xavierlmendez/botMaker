import matplotlib.pyplot as plt
import networkx as nx

from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.algorithms.breadth_first_search import BreadthFirstSearch
from mllib.math.graph.graph_structures import Graph


def main():
    # Create a graph
    graph = Graph()
    node_ids = [graph.add_node(data=f"Node {idx}") for idx in range(1, 16)]

    # Backbone chain
    for i in range(len(node_ids) - 1):
        graph.add_edge(node_ids[i], node_ids[i + 1])

    # Cross-link to add structure
    graph.add_edge(node_ids[0], node_ids[4])
    graph.add_edge(node_ids[2], node_ids[6])
    graph.add_edge(node_ids[3], node_ids[7])
    graph.add_edge(node_ids[5], node_ids[10])
    graph.add_edge(node_ids[7], node_ids[12])
    graph.add_edge(node_ids[8], node_ids[14])

    # Small clusters
    graph.add_edge(node_ids[1], node_ids[3])
    graph.add_edge(node_ids[9], node_ids[11])
    graph.add_edge(node_ids[11], node_ids[13])

    nx_graph = graph.get_nx_graph()

    # Compute layout
    pos = nx.spring_layout(nx_graph, seed=42)

    # Step-through traversal (BFS) with highlighting
    traversal = list(nx.bfs_tree(nx_graph, source=node_ids[0]).nodes())
    search_context = SearchContext(
        start_node_id=node_ids[0],
        target_node_criteria=lambda node: node.node_id == 7,
    )

    traversal_two = BreadthFirstSearch(graph).run(search_context)

    plt.figure()
    for step_index, current in enumerate(traversal_two, start=1):
        plt.clf()
        plt.title(f"Traversal Step {step_index}/{len(traversal_two)} - Current: {current}")
        node_colors = ["#E94F37" if n == current else "#6CB4EE" for n in nx_graph.nodes()]
        nx.draw(
            nx_graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=800,
            edge_color="#444",
            width=2,
        )
        plt.pause(0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
