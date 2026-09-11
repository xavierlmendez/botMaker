"""The graph-traversal view: what it is chosen by, and what it hands the page.

Nothing here looks at pixels (P-5 of the plan). A view promises the renderer three things — that it
is chosen by the recording's problem kind, that its JavaScript registers itself under that kind, and
that its layout is data a JSON document can hold — plus, for this problem, that the graph it places
is the graph the recording carries: every node placed once, every edge kept.

The recording is loaded from the committed fixture rather than re-run. A view is a function of the
document, and the document is already on disk — the same reason the two-hot view's tests load
theirs.
"""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from mllib.visualization.explain import pattern_for
from mllib.visualization.glossary import GLOSSARY
from mllib.visualization.recording import Recording
from mllib.visualization.views import View, view_for
from mllib.visualization.views.graph_traversal import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    _view_javascript,
    explain,
    graph_traversal_view,
)

LEGEND_TAG = re.compile(r'data-legend="([a-z_]+)"')

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "graph_traversal_bfs.json"


@pytest.fixture
def bfs_recording() -> Recording:
    return Recording.load(FIXTURE)


@pytest.fixture
def dfs_recording(bfs_recording: Recording) -> Recording:
    """The same document relabelled: enough to check what the view does with the algorithm name."""
    return replace(bfs_recording, problem={**bfs_recording.problem, "algorithm": "dfs"})


@pytest.fixture
def not_found_recording(bfs_recording: Recording) -> Recording:
    """The same walk, looking for a node it never reaches: the ending the fixtures never take.

    Node 15 is in the graph and not in this run's order, so the document stays coherent — a walk
    that stopped without its target — while every frame says the search failed.
    """
    return replace(
        bfs_recording,
        problem={**bfs_recording.problem, "target_node_id": 15},
        frames=[{**frame, "found": False} for frame in bfs_recording.frames],
    )


def test_a_recording_is_dispatched_to_the_view_its_problem_kind_names(bfs_recording):
    view = view_for(bfs_recording)

    assert isinstance(view, View)
    assert view.kind == bfs_recording.problem["kind"] == "graph_traversal"


def test_the_view_javascript_registers_itself_under_its_own_kind(bfs_recording):
    view = graph_traversal_view(bfs_recording)

    assert "window.walkthroughViews[KIND] = function" in view.javascript
    assert f'var KIND = "{view.kind}";' in view.javascript


def test_the_layout_is_json_serialisable_so_it_can_ride_in_the_page(bfs_recording):
    layout = graph_traversal_view(bfs_recording).layout

    assert json.loads(json.dumps(layout)) == layout


def test_the_layout_places_every_node_the_recording_carries_exactly_once(bfs_recording):
    embedded = bfs_recording.problem["layout"]
    placed = graph_traversal_view(bfs_recording).layout["nodes"]

    assert len(placed) == len(embedded["node_ids"]) == bfs_recording.problem["node_count"] == 15
    assert [node["id"] for node in placed] == embedded["node_ids"]


def test_every_placed_node_sits_inside_the_canvas(bfs_recording):
    # The page scales the canvas to its width; a node outside it is a node the reader never sees.
    layout = graph_traversal_view(bfs_recording).layout

    for node in layout["nodes"]:
        assert 0 <= node["x"] <= CANVAS_WIDTH
        assert 0 <= node["y"] <= CANVAS_HEIGHT
    assert (layout["width"], layout["height"]) == (CANVAS_WIDTH, CANVAS_HEIGHT)


def test_the_layout_keeps_the_recordings_edge_list(bfs_recording):
    # The drawing joins node ids, so the edges must arrive as the recording states them: a view
    # that re-derived them from the frames would draw only the edges the walk happened to cross.
    embedded = bfs_recording.problem["layout"]["edges"]
    layout = graph_traversal_view(bfs_recording).layout

    assert layout["edges"] == [[int(u), int(v)] for u, v in embedded]
    assert len(layout["edges"]) == 23


def test_the_title_and_subtitle_name_the_algorithm_that_was_watched(bfs_recording, dfs_recording):
    breadth = graph_traversal_view(bfs_recording)
    depth = graph_traversal_view(dfs_recording)

    assert breadth.title == "Breadth-first search walkthrough"
    assert depth.title == "Depth-first search walkthrough"
    assert "15 nodes" in breadth.layout["subtitle"]
    assert "looking for node 7" in breadth.layout["subtitle"]
    assert breadth.layout["algorithm"] == "bfs"
    assert depth.layout["algorithm"] == "dfs"


# --- The explanation: the words the page says about the run ------------------------------------
#
# Every sentence below is written out rather than recomputed from the frames. Recomputing it would
# only assert that the code agrees with itself; written out, a narration that drifts from what the
# traversal did shows up as a diff a reader can judge (plan § 4, "narration drifts from the
# recording"). What *is* recomputed from the fixture is where the moments fall, because that is a
# claim about the frames and not about the prose.


def test_the_opening_names_the_graph_the_start_and_the_target(bfs_recording):
    explanation = explain(bfs_recording)

    assert len(explanation.opening) == 4
    assert explanation.opening[0] == (
        "Instance: a graph of 15 nodes; the traversal starts at node 1 and looks for node 7."
    )


def test_the_opening_names_the_container_the_algorithm_takes_from(bfs_recording, dfs_recording):
    assert explain(bfs_recording).opening[1] == (
        "Algorithm: breadth-first search — the next node visited is the one that has waited "
        "longest in the queue, so nodes are visited in order of depth."
    )
    assert explain(dfs_recording).opening[1] == (
        "Algorithm: depth-first search — the next node visited is the one most recently pushed "
        "on the stack, so the traversal runs deep before it runs wide."
    )


def test_there_is_exactly_one_sentence_per_frame(bfs_recording):
    explanation = explain(bfs_recording)

    assert len(explanation.narration) == len(bfs_recording.frames) == 8
    assert all(line.endswith(".") for line in explanation.narration)


def test_the_first_visit_is_narrated_with_nothing_yet_waiting(bfs_recording):
    # The recorder writes the frame at the pop, before the visited node's neighbours are pushed,
    # so the first frame's pending list is empty — and the sentence says so rather than inventing
    # the neighbours the next frame will reveal.
    assert explain(bfs_recording).narration[0] == (
        "Visit 1: node 1 at depth 0 left the queue, which is empty until its neighbours are pushed."
    )


def test_a_mid_run_visit_credits_the_arrivals_to_the_visit_before_it(bfs_recording):
    # Node 7 was pushed while node 3 was being visited; this frame, node 4's, is only the first
    # one that can see it. The sentence says "since the previous visit" for exactly that reason.
    assert explain(bfs_recording).narration[4] == (
        "Visit 5: node 4 at depth 2 left the queue; since the previous visit node 7 joined it, "
        "and 2 nodes are pending (6, 7); node 7 is among them."
    )


def test_the_end_frame_is_narrated_as_the_walk_it_finished(bfs_recording):
    assert explain(bfs_recording).narration[-1] == (
        "Traversal ended after 7 visits: target found at visit 7; order 1, 2, 5, 3, 4, 6, 7."
    )


def test_a_depth_first_run_says_stack_wherever_a_breadth_first_run_says_queue(dfs_recording):
    # Only the wording is checked here: the derived document's frames are the breadth-first ones,
    # relabelled, so what it can prove is that the algorithm decides the container's name.
    explanation = explain(dfs_recording)

    assert explanation.narration[2] == (
        "Visit 3: node 5 at depth 1 left the stack; since the previous visit nodes 3, 4 joined "
        "it, and 2 nodes are pending (3, 4)."
    )
    assert "the stack with the next node first." in explanation.opening[3]
    assert all("queue" not in line for line in explanation.narration + explanation.opening)


def test_the_ending_says_why_the_run_stopped_and_where_it_went(bfs_recording):
    ending = explain(bfs_recording).ending

    assert ending == (
        "Stopped because node 7 was visited, after 7 visits.",
        "Visit order: 1, 2, 5, 3, 4, 6, 7.",
    )


def test_a_run_that_never_reached_its_target_says_so_and_marks_nothing(not_found_recording):
    # Both committed runs stop on their target, so the other half of every target-shaped sentence
    # would otherwise ship unread: the ending's first line, the end frame's outcome, and the two
    # moments that only exist when there is a target to point at.
    explanation = explain(not_found_recording)

    assert explanation.ending[0] == (
        "Stopped because the queue emptied without visiting node 15, after 7 visits."
    )
    assert explanation.narration[-1] == (
        "Traversal ended after 7 visits: target not found; order 1, 2, 5, 3, 4, 6, 7."
    )
    assert [moment.label for moment in explanation.moments] == ["peak"]


def test_every_moment_sits_on_the_frame_the_frames_themselves_name(bfs_recording):
    frames = bfs_recording.frames
    target = bfs_recording.problem["target_node_id"]
    moments = {moment.label: moment.frame_index for moment in explain(bfs_recording).moments}

    pending_sizes = [len(frame["pending"]) for frame in frames]
    target_pending = next(
        index
        for index, frame in enumerate(frames)
        if target in [entry[0] for entry in frame["pending"]]
    )
    target_visited = next(index for index, frame in enumerate(frames) if frame["node_id"] == target)

    assert moments["peak"] == pending_sizes.index(max(pending_sizes))
    assert moments["target pending"] == target_pending
    assert moments["target visited"] == target_visited


def test_moments_are_sorted_and_never_two_to_a_frame(bfs_recording):
    moments = explain(bfs_recording).moments
    indices = [moment.frame_index for moment in moments]

    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert all(len(moment.label.split()) <= 3 for moment in moments)


def test_a_traversal_shows_no_quantity_strip(bfs_recording):
    # The depth and the pending count are in every sentence already; a strip repeating them would
    # be the narration with the words taken out, and the page hides it when this is empty.
    assert explain(bfs_recording).quantities == ()


def test_every_legend_entry_is_a_thing_the_drawing_actually_tags(bfs_recording):
    # The legend is the page's key to the picture; an entry nobody draws is a lie and a drawn
    # element nobody explains is the thing the legend exists to prevent.
    drawn = set(LEGEND_TAG.findall(_view_javascript()))
    explained = {entry.key for entry in explain(bfs_recording).legend}

    assert (
        drawn
        == explained
        == {
            "edge",
            "crossed_edge",
            "node_visited",
            "node_current",
            "node_pending",
            "node_untouched",
            "pending_strip",
            "banner",
        }
    )


def test_each_visit_names_exactly_the_nodes_that_appeared_since_the_frame_before(bfs_recording):
    # The one arithmetic claim the sentence makes: the ids it says joined are the frame's pending
    # set minus the previous frame's, and nothing else. Recomputed here because it is a claim
    # about the frames rather than about the prose.
    frames = bfs_recording.frames
    narration = explain(bfs_recording).narration

    for index, frame in enumerate(frames):
        if frame["end"]:
            continue
        pending = {entry[0] for entry in frame["pending"]}
        before = {entry[0] for entry in frames[index - 1]["pending"]} if index else set()
        arrived = sorted(pending - before)
        named = re.search(r"since the previous visit nodes? ([\d, ]+) joined it", narration[index])

        if not pending:
            assert named is None, index
            continue
        if not arrived:
            assert "nothing joined it since the previous visit" in narration[index], index
            continue
        assert named, index
        assert [int(node_id) for node_id in named.group(1).split(", ")] == arrived, index


def test_the_narration_never_says_which_node_discovered_which(bfs_recording, dfs_recording):
    # The frames carry no parent pointer (plan § 4, "a narration needs a value the frame lacks"),
    # so a sentence naming the node that found another would be a guess. The narration counts what
    # joined the container and stops there.
    for recording in (bfs_recording, dfs_recording):
        text = " ".join(explain(recording).narration).lower()
        for phrase in ("from node", "discovered", "parent", "via"):
            assert phrase not in text, phrase


def test_the_page_carries_a_definition_for_every_term_it_uses_and_no_others(bfs_recording):
    explanation = explain(bfs_recording)
    text = "\n".join(
        list(explanation.opening)
        + list(explanation.narration)
        + list(explanation.ending)
        + [entry.name for entry in explanation.legend]
        + [entry.meaning for entry in explanation.legend]
    )

    assert explanation.glossary
    assert set(explanation.glossary) <= set(GLOSSARY)
    for term, definition in explanation.glossary.items():
        assert definition == GLOSSARY[term]
        assert pattern_for(term).search(text), term


def test_the_explanation_rides_in_the_layout_the_page_embeds(bfs_recording):
    layout = graph_traversal_view(bfs_recording).layout

    assert layout["explain"] == explain(bfs_recording).to_dict()
    assert json.loads(json.dumps(layout)) == layout
