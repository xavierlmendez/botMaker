"""The A* landmark view: what it is chosen by, and what it hands the page.

Nothing here looks at pixels (P-5 of the plan). What a view promises the renderer is exactly three
things — that it is chosen by the recording's problem kind, that its JavaScript registers itself
under that kind, and that its layout is data a JSON document can hold — and those are testable
without a browser.
"""

import json
import re
from dataclasses import replace

import pytest

from mllib.visualization.explain import pattern_for
from mllib.visualization.glossary import GLOSSARY
from mllib.visualization.views import View, view_for
from mllib.visualization.views.astar_landmark import (
    _pushed_bound,
    _view_javascript,
    astar_landmark_view,
    explain,
)

LEGEND_TAG = re.compile(r'data-legend="([a-z_]+)"')


def test_a_recording_is_dispatched_to_the_view_its_problem_kind_names(small_recording):
    view = view_for(small_recording)

    assert isinstance(view, View)
    assert view.kind == small_recording.problem["kind"] == "astar_landmark"


def test_a_problem_kind_with_no_view_is_refused_by_name(small_recording):
    # A kind no slice registers: the registry has to refuse it by name rather than draw a search
    # with somebody else's picture. (It read "two_hot_span" until slice 3 registered that one.)
    unknown = replace(small_recording, problem={"kind": "no_such_problem"})

    with pytest.raises(ValueError, match="no_such_problem"):
        view_for(unknown)


def test_the_view_javascript_registers_itself_under_its_own_kind(small_recording):
    view = astar_landmark_view(small_recording)

    assert "window.walkthroughViews[KIND] = function" in view.javascript
    assert f'var KIND = "{view.kind}";' in view.javascript


def test_the_layout_is_json_serialisable_so_it_can_ride_in_the_page(small_recording):
    layout = astar_landmark_view(small_recording).layout

    assert json.loads(json.dumps(layout)) == layout


def _drawn_values(recording) -> list[float]:
    """Every value the drawing puts on the bound scale: bars, the curve, the incumbent line."""
    values: list[float] = []
    for frame in recording.frames:
        values.append(frame["bound"])
        values.extend(child[1] for child in frame["children"])
        values.extend(entry[0] for entry in frame["frontier"])
        if frame["extras"].get("incumbent") is not None:
            values.append(frame["extras"]["incumbent"])
    return values


def test_the_layout_scales_bars_over_every_bound_the_run_produced(small_recording):
    # A per-frame scale would make every frontier look alike; the run's scale is what lets a reader
    # see the bounds climb between frames.
    layout = astar_landmark_view(small_recording).layout
    every_bound = _drawn_values(small_recording)

    assert layout["bound_min"] == min(every_bound)
    assert layout["bound_max"] == max(every_bound)


def test_an_incumbent_below_every_bound_still_falls_inside_the_scale(capped_recording):
    # The incumbent is drawn as a line across the curve, on the same scale as the bars. On a
    # capped run it is the cost of a goal already priced, which can sit below every bound left in
    # play — and a scale that ignored it would put that line off the bottom of the panel.
    layout = astar_landmark_view(capped_recording).layout
    incumbents = [
        frame["extras"]["incumbent"]
        for frame in capped_recording.frames
        if frame["extras"].get("incumbent") is not None
    ]

    assert incumbents, "the capped run held no incumbent"
    assert layout["bound_min"] == min(_drawn_values(capped_recording))
    assert layout["bound_max"] == max(_drawn_values(capped_recording))
    for incumbent in incumbents:
        assert layout["bound_min"] <= incumbent <= layout["bound_max"]


def test_the_view_titles_itself_with_the_cell_it_drew(small_recording):
    view = astar_landmark_view(small_recording)

    assert "rbf_chain_4x4_k2" in view.title
    assert "2 landmarks from 4 columns" in view.title


# --- The explanation: the words the page says about the run ------------------------------------
#
# Every string below is written out rather than recomputed from the frames. Recomputing it would
# only assert that the code agrees with itself; written out, a narration that drifts from what the
# search did shows up as a diff a reader can judge (plan § 4, "narration drifts from the
# recording"). What *is* recomputed from the fixture is where the moments fall, because that is a
# claim about the frames and not about the prose.


def test_the_opening_names_the_instance_the_recording_holds(fixture_recording):
    explanation = explain(fixture_recording)

    assert len(explanation.opening) == 4
    assert explanation.opening[0] == (
        "Instance: cell rbf_chain_8x8_k3, a kernel on n = 8 points; "
        "the search chooses k = 3 landmarks."
    )


def test_there_is_exactly_one_sentence_per_frame(fixture_recording):
    explanation = explain(fixture_recording)

    assert len(explanation.narration) == len(fixture_recording.frames)
    assert all(line.endswith(".") for line in explanation.narration)


def test_the_first_frame_is_narrated_as_the_frontier_opening(fixture_recording):
    assert explain(fixture_recording).narration[0] == (
        "Expanded the empty set at bound 0.5345; priced 6 children and pushed 6, so the frontier "
        "opens with 6 states and a minimum bound of 0.5847."
    )


def test_a_mid_run_frame_is_narrated_from_itself_and_the_frame_before_it(fixture_recording):
    assert explain(fixture_recording).narration[4] == (
        "Expanded {1, 6} (depth 2) at bound 0.6345, the smallest on the frontier; priced 1 "
        "children, pushed 1. The frontier stayed at 13 states and its minimum bound rose from "
        "0.6345 to 0.6629. A goal was priced for the first time: {1, 6, 7} at 1.7232."
    )


def test_the_goal_frame_is_narrated_as_the_proof_it_is(fixture_recording):
    assert explain(fixture_recording).narration[-1] == (
        "Goal {1, 3, 6} expanded at residual trace 0.7231 after 11 expansions: 22 states remain "
        "on the frontier, none with a bound below 0.7231, so {1, 3, 6} is proved optimal; 1 of "
        "them ties with it at 0.7231 to four decimals; the expanded one had the smaller bound in "
        "the digits not shown, so the tie-break was never consulted."
    )


def test_a_tie_at_display_precision_is_not_a_tie_the_tie_break_decided(fixture_recording):
    # A tie is the difference between "this is the answer" and "this is an answer" — but which
    # mechanism chose is a question about the raw bounds. Here the heap keys differ in the
    # fifteenth decimal, so the bound decided and `tie_break` was never reached; saying otherwise
    # would name a mechanism the frames show was not used.
    goal_frame = fixture_recording.frames[-1]
    cost = f"{goal_frame['bound']:.4f}"
    tied = [entry for entry in goal_frame["frontier"] if f"{entry[0]:.4f}" == cost]
    pushed_at = _pushed_bound(fixture_recording, goal_frame)

    assert [entry[1] for entry in tied] == [[1, 4, 6]]
    assert fixture_recording.configuration["tie_tolerance"] == 0.0
    assert pushed_at == 0.7231008921940996
    assert tied[0][0] == 0.7231008921941 != pushed_at
    sentence = explain(fixture_recording).narration[-1]
    assert "to four decimals; the expanded one had the smaller bound" in sentence


def test_equal_heap_keys_are_narrated_as_the_tie_break_deciding(fixture_recording):
    # The same run with the tied state's key moved onto the goal's exactly: now nothing but the
    # tie-break separates them, and the sentence has to say so.
    frames = [dict(frame) for frame in fixture_recording.frames]
    pushed_at = _pushed_bound(fixture_recording, frames[-1])
    frames[-1]["frontier"] = [
        [pushed_at, entry[1]] if f"{entry[0]:.4f}" == "0.7231" else entry
        for entry in frames[-1]["frontier"]
    ]

    sentence = explain(replace(fixture_recording, frames=frames)).narration[-1]
    assert sentence.endswith(
        "so {1, 3, 6} is proved optimal; 1 of them ties with it at 0.7231, and with equal "
        "bounds the tie-break (fifo) decided which was expanded first."
    )


def test_a_goal_nothing_ties_with_is_narrated_without_a_tie_break(fixture_recording):
    # The same run with the one tied state lifted off its last frontier: the certificate then
    # names a single optimum and the sentence has no tie-break clause to add.
    frames = [dict(frame) for frame in fixture_recording.frames]
    frames[-1]["frontier"] = [
        entry for entry in frames[-1]["frontier"] if f"{entry[0]:.4f}" != "0.7231"
    ]
    frames[-1]["frontier_size"] = len(frames[-1]["frontier"])

    assert explain(replace(fixture_recording, frames=frames)).narration[-1] == (
        "Goal {1, 3, 6} expanded at residual trace 0.7231 after 11 expansions: 21 states remain "
        "on the frontier, none with a bound below 0.7231, so {1, 3, 6} is proved optimal."
    )


def test_the_ending_says_why_the_run_stopped(fixture_recording):
    ending = explain(fixture_recording).ending

    assert 2 <= len(ending) <= 4
    assert ending[0] == (
        "Stopped because a goal state was expanded: {1, 3, 6} at residual trace 0.7231 is proved "
        "optimal after 11 expansions."
    )
    assert ending[-1] == "Every number above is read from the recording's frames and result."


def test_every_moment_sits_on_the_frame_the_frames_themselves_name(fixture_recording):
    frames = fixture_recording.frames
    moments = {moment.label: moment.frame_index for moment in explain(fixture_recording).moments}

    sizes = [frame["frontier_size"] for frame in frames]
    first_goal_priced = next(
        index
        for index, frame in enumerate(frames)
        if any(child[2] and len(child[0]) == 3 for child in frame["children"])
    )

    assert moments["peak"] == sizes.index(max(sizes))
    assert moments["goal priced"] == first_goal_priced
    assert moments["proved"] == len(frames) - 1


def test_moments_are_sorted_and_never_two_to_a_frame(fixture_recording):
    moments = explain(fixture_recording).moments
    indices = [moment.frame_index for moment in moments]

    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert all(len(moment.label.split()) <= 3 for moment in moments)


def test_every_legend_entry_is_a_thing_the_drawing_actually_tags(fixture_recording):
    # The legend is the page's key to the picture; an entry nobody draws is a lie and a drawn
    # element nobody explains is the thing the legend exists to prevent.
    drawn = set(LEGEND_TAG.findall(_view_javascript()))
    explained = {entry.key for entry in explain(fixture_recording).legend}

    assert drawn
    assert drawn == explained


def test_every_quantity_carries_one_value_per_frame(fixture_recording):
    explanation = explain(fixture_recording)

    assert {quantity.key for quantity in explanation.quantities} == {
        "bound",
        "frontier_min",
        "frontier_size",
        "incumbent",
        "gap",
        "expansions",
    }
    for quantity in explanation.quantities:
        assert len(quantity.values) == len(fixture_recording.frames), quantity.key


def test_an_exact_run_shows_no_incumbent_and_so_no_gap(fixture_recording):
    quantities = {q.key: q.values for q in explain(fixture_recording).quantities}

    assert set(quantities["incumbent"]) == {None}
    assert set(quantities["gap"]) == {None}


def test_the_page_carries_a_definition_for_every_term_it_uses_and_no_others(fixture_recording):
    explanation = explain(fixture_recording)
    text = "\n".join(
        list(explanation.opening)
        + list(explanation.narration)
        + list(explanation.ending)
        + [entry.meaning for entry in explanation.legend]
        + [quantity.definition for quantity in explanation.quantities]
        + [quantity.term for quantity in explanation.quantities]
    )

    assert explanation.glossary
    assert set(explanation.glossary) <= set(GLOSSARY)
    for term, definition in explanation.glossary.items():
        assert definition == GLOSSARY[term]
        assert pattern_for(term).search(text), term


def test_the_explanation_rides_in_the_layout_the_page_embeds(fixture_recording):
    layout = astar_landmark_view(fixture_recording).layout

    assert layout["explain"] == explain(fixture_recording).to_dict()
    assert json.loads(json.dumps(layout)) == layout


def test_a_capped_run_is_marked_where_it_stopped_and_ends_without_a_certificate(capped_recording):
    explanation = explain(capped_recording)
    labels = [moment.label for moment in explanation.moments]

    assert labels[-1] == "capped"
    assert explanation.moments[-1].frame_index == len(capped_recording.frames) - 1
    assert explanation.ending[0] == (
        "Stopped because the cap of 5 expansions was reached before a goal was expanded."
    )
    assert explanation.ending[1].startswith("The incumbent {0, 1} at residual trace 6.0000 is ")
    assert "(certified gap)" in explanation.ending[1]


def test_a_capped_runs_gap_is_the_incumbent_over_the_frontiers_minimum(capped_recording):
    quantities = {q.key: q.values for q in explain(capped_recording).quantities}

    assert all(value is not None for value in quantities["frontier_min"])
    for incumbent, gap in zip(quantities["incumbent"], quantities["gap"], strict=True):
        assert (gap is None) == (incumbent is None)


def test_a_run_that_set_a_goal_aside_says_the_incumbent_was_returned(superseded_recording):
    explanation = explain(superseded_recording)

    assert explanation.narration[-1] == (
        "Goal {0, 1, 2, 3, 6} at residual trace 0.2925 was within the tie tolerance of the held "
        "incumbent {0, 2, 4, 5, 7} at 0.0408, so the incumbent is returned after 71 expansions."
    )
    assert explanation.moments[-1].label == "returned"
    assert explanation.moments[-1].frame_index == len(superseded_recording.frames) - 1
    assert explanation.ending[0].startswith("Stopped because a goal state was expanded: ")
    assert "was returned after 71 expansions." in explanation.ending[0]


def test_the_goal_that_was_set_aside_is_not_narrated_as_proved(superseded_recording):
    # Its own frame's frontier holds states cheaper than it: the optimality sentence would be
    # visibly false on the page it was printed on.
    assert explain(superseded_recording).narration[-2] == (
        "Goal {0, 1, 2, 3, 6} expanded at residual trace 0.2925 after 71 expansions, but it is "
        "within the tie tolerance of the held incumbent {0, 2, 4, 5, 7} at 0.0408, so it is set "
        "aside."
    )


def test_a_pruned_run_counts_its_prunings_by_the_mechanism_that_declined_them(capped_recording):
    narration = explain(capped_recording).narration

    assert ", pruned 6 (6 goal siblings)." in narration[1]


def test_every_incumbent_moment_is_a_frame_where_the_incumbent_actually_moved(capped_recording):
    # Recomputed from the frames rather than read back from the explanation: the claim is that a
    # marker sits where the incumbent first appears or falls, and nowhere else. The terminal
    # moment outranks an incumbent on the same frame, so a frame it claimed is excluded.
    frames = capped_recording.frames
    moved = set()
    seen = None
    for index, frame in enumerate(frames):
        incumbent = frame["extras"].get("incumbent")
        if incumbent is not None and (seen is None or float(incumbent) < seen):
            moved.add(index)
        if incumbent is not None:
            seen = float(incumbent)

    moments = explain(capped_recording).moments
    marked = {moment.frame_index for moment in moments if moment.label == "incumbent"}
    outranked = {moment.frame_index for moment in moments if moment.label != "incumbent"}

    assert moved, "the capped run never held an incumbent"
    assert marked == moved - outranked
    assert marked.isdisjoint(set(range(len(frames))) - moved)
