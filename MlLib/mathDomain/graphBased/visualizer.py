import time

import matplotlib.pyplot as plt
import networkx as nx

from MlLib.mathDomain.graphBased.graphStructures import Graph

class Visualizer:
    def __init__(self):
        self.metadata = {
            "name": "Graph Visualizer",
            "description": "Utility for visualizing graph traversal with matplotlib and networkx."
        }
        # TODO: review metadata (auto-generated)

# tutorial used https://www.youtube.com/watch?v=7XVTnCrWDPY
    def show(self, graph:Graph, traversalOrder=None, position = None):
        plt.figure()
        plt.title(graph.name)
        if traversalOrder is not None:
            for i, node in enumerate(traversalOrder, start=1):
                plt.clf()
                plt.title(graph.name)
                nx.draw(
                    graph.getNxGraph(),
                    position,
                    node_size=500,
                    with_labels=True,
                    node_color=['r' if n == node else 'b' for n in graph.nodes]
                )
                plt.draw()
                plt.pause(0.5)
            plt.show()
            time.sleep(0.5)
