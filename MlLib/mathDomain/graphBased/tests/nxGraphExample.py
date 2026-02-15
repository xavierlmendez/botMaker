import networkx as nx
import matplotlib.pyplot as plt

from botMaker.MlLib.mathDomain.algorithmImplementations.abstractGraphAlgorithm import SearchContext
from botMaker.MlLib.mathDomain.algorithmImplementations.breadthFirstSearch import BreadthFirstSearch
from botMaker.MlLib.mathDomain.graphBased.graphStructures import Graph


def main():
    # Create a graph
    graph = Graph()
    node_ids = [graph.addNode(data=f"Node {idx}") for idx in range(1, 16)]

    # Backbone chain
    for i in range(len(node_ids) - 1):
        graph.addEdge(node_ids[i], node_ids[i + 1])

    # Cross-link to add structure
    graph.addEdge(node_ids[0], node_ids[4])
    graph.addEdge(node_ids[2], node_ids[6])
    graph.addEdge(node_ids[3], node_ids[7])
    graph.addEdge(node_ids[5], node_ids[10])
    graph.addEdge(node_ids[7], node_ids[12])
    graph.addEdge(node_ids[8], node_ids[14])

    # Small clusters
    graph.addEdge(node_ids[1], node_ids[3])
    graph.addEdge(node_ids[9], node_ids[11])
    graph.addEdge(node_ids[11], node_ids[13])

    nx_graph = graph.getNxGraph()


    # Compute layout
    pos = nx.spring_layout(nx_graph, seed=42)

    # Step-through traversal (BFS) with highlighting
    traversal = list(nx.bfs_tree(nx_graph, source=node_ids[0]).nodes())
    searchContext = SearchContext(
        start_node_id=node_ids[0],
        target_node_criteria = lambda node: node.node_id == 7,
    )

    traversal_two = BreadthFirstSearch(graph).run(searchContext)

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
