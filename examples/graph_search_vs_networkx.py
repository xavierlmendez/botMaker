"""Breadth-first and depth-first search over one small graph, recorded and compared to networkx.

This example used to be an animation: fifteen nodes, a matplotlib figure, and a redraw every 0.6
seconds while a BFS walked them. Watching it told you the order — once, at the speed the loop chose,
with no way back to the step you missed and nothing left over when the window closed. BL-43 slice 4
replaced it with the two artefacts a reader can actually use: a **recording**, the run's frames and
result as a versioned JSON document, and a **walkthrough**, that recording rendered as one offline
HTML page with a stepper. The animation's own oracle survives as
`tests/math/algorithms/test_graph_search_example_fingerprint.py`, which pins the traversals this
graph produced before the switch (CONTRIBUTING § "Behavioural baseline before a refactor").

The graph is unchanged and so is the comparison with networkx, which is the other half of what this
file is for: the library's BFS and `networkx.bfs_tree` walk the same graph from the same start, and
the run says whether they agree and where they part company.

This is a composition root (`docs/ARCHITECTURE.md` §1): it wires an algorithm from `math` to a
recorder from `visualization`, which is the only place those two are allowed to meet.

    uv run python examples/graph_search_vs_networkx.py --output-dir /tmp/graph-walkthrough
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx

from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.algorithms.breadth_first_search import BreadthFirstSearch
from mllib.math.algorithms.depth_first_search import DepthFirstSearch
from mllib.math.graph.graph_structures import Graph
from mllib.visualization.html_renderer import write_walkthrough
from mllib.visualization.recorders.graph_traversal import TraversalRecorder
from mllib.visualization.recording import Recording
from mllib.visualization.views import view_for

# The string that ties a recording to the picture drawn of it: `view_for` dispatches on it, and
# `views/graph_traversal.py` registers itself under the same name.
PROBLEM_KIND = "graph_traversal"

# The node the searches are looking for, and the seed the drawing positions come from. The seed is
# fixed so that two runs of this example produce the same document, byte for byte: a recording with
# a different layout every time would be undiffable and useless as a fixture.
TARGET_NODE_ID = 7
LAYOUT_SEED = 0

ALGORITHMS = {"bfs": BreadthFirstSearch, "dfs": DepthFirstSearch}


def build_graph() -> tuple[Graph, list[int]]:
    """The fifteen-node graph: a fourteen-edge backbone chain, six cross-links, three cluster edges.

    The last three are what the old file's "Small clusters" comment named: three edges thickening
    the chain around nodes 2 to 4 and 10 to 14, not three components. The graph stays connected
    throughout, and it is unchanged from the animation this example replaced — the fingerprint test
    depends on that.
    """
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

    return graph, node_ids


def build_layout(graph: Graph, node_ids: list[int]) -> dict[str, Any]:
    """Where the nodes sit and which pairs are joined: everything the picture needs, computed once.

    The positions come from a seeded spring layout, computed here rather than in the browser for the
    same reason every other view's geometry is: numpy is here, the recording has to be reproducible,
    and a layout that a page recomputed would move every time somebody opened it. ``node_ids`` is
    carried beside ``positions`` because the two lists are parallel — a node's position is the entry
    at its index, not at its id, and the ids happen to start at one.
    """
    nx_graph = graph.get_nx_graph()
    positions = nx.spring_layout(nx_graph, seed=LAYOUT_SEED)
    return {
        "node_ids": list(node_ids),
        "positions": [
            [float(positions[node_id][0]), float(positions[node_id][1])] for node_id in node_ids
        ],
        "edges": [[int(edge.u), int(edge.v)] for edge in graph.edges],
    }


def record_traversal(
    algorithm: str,
    graph: Graph,
    context: SearchContext,
    layout: dict[str, Any],
) -> tuple[Recording, list[int]]:
    """Run one traversal with a recorder attached and assemble its recording.

    The recorder is constructed here, handed to the engine, and read back here: the engine's return
    value is the traversal order and nothing else, exactly as it was before it could be watched
    (D-28, D-32). ``Recording.from_recorder`` does the assembling, so the frames and the result are
    converted by the recorder that watched them and this file never learns what a frame holds.
    """
    recorder = TraversalRecorder()
    traversal_order = ALGORITHMS[algorithm](graph, recorder=recorder).run(context)

    recording = Recording.from_recorder(
        recorder,
        problem={
            "kind": PROBLEM_KIND,
            "algorithm": algorithm,
            "node_count": len(graph.nodes),
            "target_node_id": TARGET_NODE_ID,
            "layout": layout,
        },
        configuration={
            "start": context.start_node_id,
            "max_depth": context.max_depth,
            "allow_revisiting": context.allow_revisiting,
        },
        result=traversal_order,
    )
    return recording, traversal_order


def write_algorithm(
    algorithm: str,
    graph: Graph,
    context: SearchContext,
    layout: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path, list[int]]:
    """One traversal, written as both artefacts: `<algorithm>.json` and `<algorithm>.html`.

    The page is the recording plus a picture of it — `view_for` picks the picture off the
    recording's problem kind, which is why the kind is the only thing tying the two halves
    together — so the pair is written from one run and can never disagree about what it walked.
    """
    recording, traversal_order = record_traversal(algorithm, graph, context, layout)
    recording_path = recording.save(output_dir / f"{algorithm}.json")
    page_path = write_walkthrough(recording, view_for(recording), output_dir / f"{algorithm}.html")
    return recording_path, page_path, traversal_order


def compare_with_networkx(graph: Graph, start_node_id: int, traversal_order: list[int]) -> None:
    """Put the library's exhaustive BFS beside `networkx.bfs_tree` over the same graph.

    The two need not agree node for node — a BFS is only obliged to visit level by level, and which
    node of a level comes first is a choice about neighbour order, not about correctness — so the
    run prints both orders and the first place they part company rather than asserting they match.
    """
    networkx_order = list(nx.bfs_tree(graph.get_nx_graph(), source=start_node_id).nodes())
    print(f"  mllib BFS     : {traversal_order}")
    print(f"  networkx BFS  : {networkx_order}")
    if traversal_order == networkx_order:
        print("  the two orders agree node for node.")
        return

    divergence = next(
        (
            position
            # strict=False on purpose: two traversals of different lengths are exactly the case
            # this branch reports, and the fallback below names the shorter one's end.
            for position, (mine, theirs) in enumerate(
                zip(traversal_order, networkx_order, strict=False)
            )
            if mine != theirs
        ),
        min(len(traversal_order), len(networkx_order)),
    )
    print(
        f"  same levels, different order within them: they first differ at position {divergence}."
    )


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("walkthroughs"),
        help="Where the recordings and the walkthrough pages are written.",
    )
    arguments = parser.parse_args(argv)

    graph, node_ids = build_graph()
    layout = build_layout(graph, node_ids)
    start_node_id = node_ids[0]

    searching = SearchContext(
        start_node_id=start_node_id,
        target_node_criteria=lambda node: node.node_id == TARGET_NODE_ID,
    )
    # The same walk with a criterion nothing matches: the whole graph, which is what a comparison
    # with networkx's traversal needs and what makes the "ended, target not found" frame reachable.
    exhaustive = SearchContext(
        start_node_id=start_node_id,
        target_node_criteria=lambda node: False,
    )

    written: list[Path] = []
    for algorithm in ALGORITHMS:
        recording_path, page_path, traversal_order = write_algorithm(
            algorithm, graph, searching, layout, arguments.output_dir
        )
        written.extend((recording_path, page_path))
        print(
            f"{algorithm.upper()} from node {start_node_id} for node {TARGET_NODE_ID}: "
            f"{traversal_order} ({len(traversal_order) + 1} frames)"
        )

    print(f"\nBFS over the whole graph, against networkx (start {start_node_id}):")
    _, full_order = record_traversal("bfs", graph, exhaustive, layout)
    compare_with_networkx(graph, start_node_id, full_order)

    print("\nWritten:")
    for path in written:
        print(f"  {path}")
    return written


if __name__ == "__main__":
    main()
