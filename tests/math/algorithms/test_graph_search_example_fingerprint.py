"""The fingerprint of `examples/graph_search_vs_networkx.py`, captured before the example changed.

CONTRIBUTING § "Behavioural baseline before a refactor": when the code a snapshot was taken from is
deleted, its fingerprints are captured first and tested against — the guarantee outlives the oracle.
The oracle here was the example as it stood at `1cc6fb1`: a matplotlib animation that stepped a BFS
over a fifteen-node graph and drew a frame per step, printing nothing a test could read. So the
fingerprint was taken by replicating that file's graph builder exactly and running the same search
directly, on 2026-09-10, before `math/graph/visualizer.py` was deleted and the example rewritten
(BL-43 slice 4). The two lines are not the same kind of claim:

    BFS from node 1, target `node.node_id == 7`  ->  [1, 2, 5, 3, 4, 6, 7]
        what the deleted animation walked, node for node: the example's own behaviour, and the
        thing the rewrite is obliged to reproduce.

    DFS from node 1, target `node.node_id == 7`  ->  [1, 2, 3, 4, 5, 6, 7]
        `main`'s depth-first search over the same graph, captured at the same moment. The old
        example never ran DFS — the rewritten one does — so this is not a fingerprint of anything
        that was deleted; it is the behaviour the new example inherits, pinned here beside the
        BFS order so that both halves of the example have a snapshot of the same vintage.

The graph is replicated here rather than imported so that this test keeps holding the *old*
example's graph. If a future edit to the example changes the graph, that is a change to what the
example demonstrates and this test is what says so out loud; it does not silently follow along.
"""

from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.algorithms.breadth_first_search import BreadthFirstSearch
from mllib.math.algorithms.depth_first_search import DepthFirstSearch
from mllib.math.graph.graph_structures import Graph

# Both captured at 1cc6fb1, before the example was switched to a recording; see the module
# docstring for which of the two the deleted animation actually produced.
BFS_FINGERPRINT = [1, 2, 5, 3, 4, 6, 7]
DFS_FINGERPRINT = [1, 2, 3, 4, 5, 6, 7]

TARGET_NODE_ID = 7


def build_example_graph() -> tuple[Graph, list[int]]:
    """The fifteen-node graph the example builds.

    A fourteen-edge backbone chain, six cross-links, and the three edges the old file grouped under
    the comment "Small clusters" — three edges, not three clusters: they thicken the chain around
    nodes 2 to 4 and 10 to 14 rather than adding components, and the graph stays connected
    throughout.
    """
    graph = Graph()
    node_ids = [graph.add_node(data=f"Node {idx}") for idx in range(1, 16)]

    for i in range(len(node_ids) - 1):
        graph.add_edge(node_ids[i], node_ids[i + 1])

    graph.add_edge(node_ids[0], node_ids[4])
    graph.add_edge(node_ids[2], node_ids[6])
    graph.add_edge(node_ids[3], node_ids[7])
    graph.add_edge(node_ids[5], node_ids[10])
    graph.add_edge(node_ids[7], node_ids[12])
    graph.add_edge(node_ids[8], node_ids[14])

    graph.add_edge(node_ids[1], node_ids[3])
    graph.add_edge(node_ids[9], node_ids[11])
    graph.add_edge(node_ids[11], node_ids[13])

    return graph, node_ids


def _context(start_node_id: int) -> SearchContext:
    return SearchContext(
        start_node_id=start_node_id,
        target_node_criteria=lambda node: node.node_id == TARGET_NODE_ID,
    )


def test_the_example_graph_is_the_one_the_fingerprint_was_taken_from():
    # The fingerprints below say nothing unless the graph is the same graph, so its shape is pinned
    # first: fifteen nodes, fourteen backbone edges, six cross-links and three cluster edges.
    graph, node_ids = build_example_graph()

    assert node_ids == list(range(1, 16))
    assert len(graph.edges) == 23
    assert sorted(graph.nodes[1].neighbors) == [2, 5]


def test_breadth_first_search_reproduces_the_example_traversal():
    graph, node_ids = build_example_graph()

    assert BreadthFirstSearch(graph).run(_context(node_ids[0])) == BFS_FINGERPRINT


def test_depth_first_search_reproduces_the_example_traversal():
    graph, node_ids = build_example_graph()

    assert DepthFirstSearch(graph).run(_context(node_ids[0])) == DFS_FINGERPRINT
