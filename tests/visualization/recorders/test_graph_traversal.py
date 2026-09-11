"""The traversal recorder against the real searches, on a graph small enough to check by eye.

Every claim here is about what the frames say and about the searches being unchanged by being
watched. The graph is four nodes and a tail:

        2 — 5
       /
    1 — 3
       \\
        4

which is the smallest shape that tells the two algorithms apart. Breadth-first search takes 2, then
3, then 4 before it ever reaches 5; depth-first search takes 2, dives to 5, and only then comes back
for 3 and 4. The pending container is where that difference lives, and it is the thing these tests
watch most closely.
"""

import json

import pytest

from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.algorithms.breadth_first_search import BreadthFirstSearch
from mllib.math.algorithms.depth_first_search import DepthFirstSearch
from mllib.math.graph.graph_structures import Graph
from mllib.math.recorder import AbstractTraversalRecorder, NullRecorder
from mllib.visualization.recorders.graph_traversal import TraversalFrame, TraversalRecorder

ALGORITHMS = (BreadthFirstSearch, DepthFirstSearch)


def build_graph() -> Graph:
    """Node 1 joined to 2, 3 and 4; node 2 joined to 5."""
    graph = Graph()
    for name in "ABCDE":
        graph.add_node(data=name)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)
    return graph


def _run(engine, *, target=None, watched=True):
    """One traversal over the graph above, with or without a recorder; returns both.

    ``target`` of ``None`` matches no node — a node id is never ``None`` — so the walk exhausts the
    graph, which is the run whose end frame reports "not found".
    """
    recorder = TraversalRecorder() if watched else None
    context = SearchContext(
        start_node_id=1,
        target_node_criteria=lambda node: node.node_id == target,
    )
    return engine(build_graph(), recorder=recorder).run(context), recorder


@pytest.mark.parametrize("engine", ALGORITHMS)
def test_a_run_leaves_one_frame_per_visit_and_one_for_the_end(engine):
    order, recorder = _run(engine)

    assert len(recorder.frames) == len(order) + 1
    assert [frame.index for frame in recorder.frames] == list(range(len(order) + 1))
    assert [frame.end for frame in recorder.frames] == [False] * len(order) + [True]


@pytest.mark.parametrize("engine", ALGORITHMS)
def test_the_recorded_visits_are_the_traversal_the_search_returned(engine):
    order, recorder = _run(engine)

    visits = [frame.node_id for frame in recorder.frames if not frame.end]
    assert visits == order
    assert recorder.frames[-1].traversal_order == order
    assert recorder.describe_result(order) == {"traversal_order": order, "visit_count": len(order)}


def test_breadth_first_pending_is_a_queue_snapshot():
    # Taken after the node comes off the queue and before its neighbours go on, so the pending list
    # is what was already waiting. A queue hands out its head next, which is why 3 stands in front
    # of 4 at the visit of node 2 and is the next of the two to be visited.
    order, recorder = _run(BreadthFirstSearch)

    assert order == [1, 2, 3, 4, 5]
    assert [frame.pending for frame in recorder.frames if not frame.end] == [
        [],
        [(3, 1), (4, 1)],
        [(4, 1), (5, 2)],
        [(5, 2)],
        [],
    ]
    for frame in recorder.frames:
        if frame.pending:
            head = frame.pending[0][0]
            assert all(order.index(head) < order.index(other) for other, _ in frame.pending[1:])


def test_depth_first_pending_is_a_stack_snapshot():
    # The same snapshot of the same moment, on a container that hands out its *last* entry next: at
    # the visit of node 2 the stack reads 4 then 3 — the reverse of the queue's order — and 3 is
    # the next of the two to be visited.
    order, recorder = _run(DepthFirstSearch)

    assert order == [1, 2, 5, 3, 4]
    assert [frame.pending for frame in recorder.frames if not frame.end] == [
        [],
        [(4, 1), (3, 1)],
        [(4, 1), (3, 1)],
        [(4, 1)],
        [],
    ]
    for frame in recorder.frames:
        if frame.pending:
            top = frame.pending[-1][0]
            assert all(order.index(top) < order.index(other) for other, _ in frame.pending[:-1])


def test_breadth_first_visited_holds_discovered_nodes_where_depth_first_holds_walked_ones():
    # The same field, two meanings, because the algorithms put a node in the set at different
    # moments: BFS marks a node when it *queues* it, so by its second visit all three of node 1's
    # neighbours are already in; DFS marks a node when it *pops* it, so its second visit knows only
    # the two nodes actually walked. Recording the set as each algorithm keeps it is what makes the
    # frames honest — normalising the two to one meaning would misreport both.
    _, breadth = _run(BreadthFirstSearch)
    _, depth = _run(DepthFirstSearch)

    assert breadth.frames[1].node_id == depth.frames[1].node_id == 2
    assert breadth.frames[1].visited == [1, 2, 3, 4]
    assert depth.frames[1].visited == [1, 2]
    # The first visit agrees: one node discovered, one node walked, the start in both.
    assert breadth.frames[0].visited == depth.frames[0].visited == [1]
    # And by the last visit both have seen the whole graph.
    assert breadth.frames[-2].visited == depth.frames[-2].visited == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("engine", ALGORITHMS)
def test_the_end_frame_says_the_target_was_found(engine):
    order, recorder = _run(engine, target=5)

    assert order[-1] == 5
    assert recorder.frames[-1].end is True
    assert recorder.frames[-1].found is True
    assert recorder.frames[-1].caption.endswith("target found.")


@pytest.mark.parametrize("engine", ALGORITHMS)
def test_the_end_frame_says_the_traversal_exhausted_the_graph(engine):
    order, recorder = _run(engine)

    assert sorted(order) == [1, 2, 3, 4, 5]
    assert recorder.frames[-1].found is False
    assert recorder.frames[-1].caption.endswith("target not found.")


@pytest.mark.parametrize("engine", ALGORITHMS)
def test_watching_a_traversal_does_not_change_it(engine):
    # The null path and the watched path must produce the same walk, and the explicit NullRecorder
    # must be the same as no recorder at all: that is the whole guarantee the `enabled` guard buys.
    unwatched, _ = _run(engine, watched=False)
    with_null = engine(build_graph(), recorder=NullRecorder()).run(
        SearchContext(start_node_id=1, target_node_criteria=lambda node: False)
    )
    watched, _ = _run(engine)

    assert unwatched == with_null == watched


@pytest.mark.parametrize("engine", ALGORITHMS)
def test_every_frame_is_json_without_a_default_encoder(engine):
    # A frame that needed `json.dumps(..., default=...)` would be a frame holding something that is
    # not plain Python, which is exactly what the recorder exists to prevent.
    _, recorder = _run(engine, target=5)

    for payload in recorder.frame_dicts():
        assert json.loads(json.dumps(payload)) == payload


def test_a_visit_frame_captions_the_node_its_depth_and_what_is_waiting():
    _, recorder = _run(BreadthFirstSearch)

    assert (
        recorder.frames[1].caption
        == "Visited node 2 at depth 1; 2 nodes pending; 2 visited so far."
    )
    assert recorder.frames[-1].caption == "Traversal ended after 5 visits; target not found."


def test_the_null_recorder_refuses_a_traversal_frame():
    # It is the traversal recorder's shape, and every recording method on it raises: a call that
    # gets through means an engine's `if self.recorder.enabled` guard is missing.
    null = NullRecorder()

    assert isinstance(null, AbstractTraversalRecorder)
    assert null.enabled is False
    with pytest.raises(RuntimeError, match="guard is missing"):
        null.record_visit(1, 0, [], [], [1])
    with pytest.raises(RuntimeError, match="guard is missing"):
        null.record_traversal_end([1], False)


def test_a_traversal_recorder_is_on_by_default():
    recorder = TraversalRecorder()

    assert recorder.enabled is True
    assert recorder.frames == []


def test_an_end_frame_observes_no_node():
    # It is about the run, not about a node: nothing was visited at that moment, and a frame that
    # repeated the last node would invite a reader to count one visit twice.
    _, recorder = _run(BreadthFirstSearch, target=5)
    end = recorder.frames[-1]

    assert isinstance(end, TraversalFrame)
    assert (end.node_id, end.depth) == (None, None)
    assert end.to_dict()["node_id"] is None
