"""The two-hot span view: what it is chosen by, and what it hands the page.

Nothing here looks at pixels (P-5 of the plan). A view promises the renderer four things — that it
is chosen by the recording's problem kind, that its JavaScript registers itself under that kind,
that its layout is data a JSON document can hold, and that every scale in that layout covers the
whole run rather than one frame — and all four are testable without a browser.

The recording is the committed fixture, loaded from disk. That is deliberate: it means these tests
run without the optional torch group, because a view is a function of the document and the document
is already written.
"""

import json
import re
from dataclasses import replace
from itertools import pairwise

import numpy as np

from mllib.visualization.explain import pattern_for
from mllib.visualization.glossary import GLOSSARY
from mllib.visualization.views import View, view_for
from mllib.visualization.views.two_hot_span import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    TWO_HOT_SHARE,
    _two_hot_columns,
    _view_javascript,
    explain,
    two_hot_span_view,
)

LEGEND_TAG = re.compile(r'data-legend="([a-z_]+)"')


def _full_frames(recording):
    return [frame for frame in recording.frames if frame["full"]]


def _stepping_frames(recording):
    """The full frames that took a step: the end frame restates the last one and takes none."""
    return [
        (index, frame)
        for index, frame in enumerate(recording.frames)
        if frame["full"] and not frame["end"]
    ]


def _two_hot_by_hand(spanning_set):
    """The 0.95 rule written out again here, from the frame's V and nothing the view owns.

    Asking the view's own helper where its moments should fall would only assert that the code
    agrees with itself. This is the rule as `docs/plans/2026-09-two-hot-span.md` states it —
    squared coordinates, the two largest holding at least 95 % of the norm, a zero column
    excluded — spelled out a second time so the two implementations can disagree.
    """
    matrix = np.asarray(spanning_set, dtype=float)
    count = 0
    pairs = []
    for column in range(matrix.shape[1]):
        entries = matrix[:, column]
        norm = float(np.sum(entries**2))
        if norm <= 0.0:
            continue
        ranked = sorted(range(len(entries)), key=lambda row: (-abs(entries[row]), row))
        largest, second = ranked[0], ranked[1]
        if entries[largest] ** 2 + entries[second] ** 2 < TWO_HOT_SHARE * norm:
            continue
        count += 1
        pairs.append((min(largest, second), max(largest, second)))
    return count, int(matrix.shape[1]), pairs


def test_a_recording_is_dispatched_to_the_view_its_problem_kind_names(two_hot_recording):
    view = view_for(two_hot_recording)

    assert isinstance(view, View)
    assert view.kind == two_hot_recording.problem["kind"] == "two_hot_span"


def test_the_view_javascript_registers_itself_under_its_own_kind(two_hot_recording):
    view = two_hot_span_view(two_hot_recording)

    assert "window.walkthroughViews[KIND] = function" in view.javascript
    assert f'var KIND = "{view.kind}";' in view.javascript


def test_the_layout_is_json_serialisable_so_it_can_ride_in_the_page(two_hot_recording):
    layout = two_hot_span_view(two_hot_recording).layout

    assert json.loads(json.dumps(layout)) == layout


def test_the_layout_places_every_node_the_recording_positioned(two_hot_recording):
    layout = two_hot_span_view(two_hot_recording).layout
    embedded = two_hot_recording.problem["layout"]

    assert layout["node_count"] == two_hot_recording.problem["n"] == len(embedded["positions"])
    assert len(layout["positions"]) == len(embedded["positions"])
    assert layout["edges"] == embedded["edges"]


def test_every_position_lands_inside_the_graph_panel(two_hot_recording):
    # The recording's coordinates are the instance's own (a roach ladder spans x = 0..9, y = 0..1);
    # what the page needs is those coordinates inside the box the panel actually draws.
    layout = two_hot_span_view(two_hot_recording).layout
    box = layout["graph"]

    for x, y in layout["positions"]:
        assert box["x"] <= x <= box["x"] + box["width"]
        assert box["y"] <= y <= box["y"] + box["height"]


def test_the_ladder_keeps_its_two_rows_apart(two_hot_recording):
    # The roach's twenty nodes sit on exactly two heights; scaling each axis on its own extent is
    # what keeps that readable, so exactly two distinct y values must survive the mapping.
    layout = two_hot_span_view(two_hot_recording).layout

    assert len({y for _, y in layout["positions"]}) == 2


def test_the_heatmap_scale_covers_every_entry_of_every_recorded_v(two_hot_recording):
    layout = two_hot_span_view(two_hot_recording).layout
    entries = [
        abs(value)
        for frame in _full_frames(two_hot_recording)
        for row in frame["spanning_set"]
        for value in row
    ]

    assert layout["value_max"] == max(entries)
    assert layout["column_count"] == len(_full_frames(two_hot_recording)[0]["spanning_set"][0])


def test_the_cut_curve_range_covers_every_rounded_cut_and_the_floor(two_hot_recording):
    layout = two_hot_span_view(two_hot_recording).layout
    cuts = [frame["rounded_cut"] for frame in _full_frames(two_hot_recording)]
    floor = two_hot_recording.problem["spectral_floor"]

    assert layout["spectral_floor"] == floor
    assert layout["cut_min"] == min([*cuts, floor])
    assert layout["cut_max"] == max([*cuts, floor])


def test_the_loss_curve_range_covers_every_step_including_the_light_ones(two_hot_recording):
    layout = two_hot_span_view(two_hot_recording).layout
    losses = [frame["training_loss"] for frame in two_hot_recording.frames if not frame["end"]]

    assert len(losses) == len(two_hot_recording.frames) - 1
    assert layout["loss_min"] == min(losses)
    assert layout["loss_max"] == max(losses)


def test_the_end_frames_restated_loss_is_not_a_point_of_the_loss_series(two_hot_recording):
    # The end frame carries the loss of the step it follows, at that step's own x. It is the same
    # number at the same place, so it belongs to neither the range nor the curve.
    frames = [dict(frame) for frame in two_hot_recording.frames]
    assert frames[-1]["end"] is True
    frames[-1]["training_loss"] = 1.0e6
    stretched = replace(two_hot_recording, frames=frames)

    assert (
        two_hot_span_view(stretched).layout["loss_max"]
        == (two_hot_span_view(two_hot_recording).layout["loss_max"])
    )


def test_the_run_wide_counts_reach_the_last_step_and_the_largest_component_count(
    two_hot_recording,
):
    layout = two_hot_span_view(two_hot_recording).layout

    assert layout["step_count"] == max(frame["step"] for frame in two_hot_recording.frames) + 1
    assert layout["component_max"] == max(
        frame["component_count"] for frame in _full_frames(two_hot_recording)
    )


def test_the_panels_stay_inside_the_canvas_the_page_is_given(two_hot_recording):
    layout = two_hot_span_view(two_hot_recording).layout

    assert (layout["width"], layout["height"]) == (CANVAS_WIDTH, CANVAS_HEIGHT)
    for name in ("banner", "graph", "heatmap", "loss_curve", "cut_curve"):
        box = layout[name]
        assert box["x"] + box["width"] <= CANVAS_WIDTH, name
        assert box["y"] + box["height"] <= CANVAS_HEIGHT, name


def test_the_view_titles_itself_with_the_graph_and_the_run_it_drew(two_hot_recording):
    view = two_hot_span_view(two_hot_recording)

    assert "roach_g5" in view.title
    assert "20 nodes" in view.title
    assert "λ = 10.0" in view.title
    assert "spectral init" in view.title


# --- The explanation: the words the page says about the run ------------------------------------
#
# Every sentence below is written out rather than recomputed from the frames. Recomputing it would
# only assert that the code agrees with itself; written out, a narration that drifts from what the
# run did shows up as a diff a reader can judge (plan § 4, "narration drifts from the recording").
# What *is* recomputed from the fixture is where the moments fall and which numbers are present,
# because those are claims about the frames rather than about the prose.


def test_the_opening_names_the_instance_the_recording_holds(two_hot_recording):
    explanation = explain(two_hot_recording)

    assert len(explanation.opening) == 4
    assert explanation.opening[0] == (
        "Instance: roach_g5, n = 20 vertices, K = 2 clusters asked, λ = 10.0 on the collision "
        "measure, spectral initialisation."
    )


def test_there_is_exactly_one_sentence_per_frame(two_hot_recording):
    explanation = explain(two_hot_recording)

    assert len(explanation.narration) == len(two_hot_recording.frames)
    assert all(line.endswith(".") for line in explanation.narration)


def test_the_first_full_frame_is_narrated_against_the_floor_it_starts_on(two_hot_recording):
    assert explain(two_hot_recording).narration[0] == (
        "Step 0: E* 0.1658 against a floor of 0.0713, Ê 6.5000, 4 components of K = 2; "
        "0 of 18 columns are 2-hot; mean R(v_j) 0.0813."
    )


def test_a_light_frame_carries_its_training_loss_and_says_whose_picture_is_showing(
    two_hot_recording,
):
    # Frame 4 is a light frame four steps after the last full one: the only number it has is the
    # loss, and the panels above it belong to step 0.
    assert two_hot_recording.frames[4]["full"] is False
    assert explain(two_hot_recording).narration[4] == (
        "Step 4: training loss -21.1456 (down by 1.8756 since step 3); the picture is from step 0."
    )


def test_a_later_full_frame_is_narrated_against_the_previous_full_frame(two_hot_recording):
    # Step 20 is the frame the first 2-hot columns appear on, and its deltas are measured against
    # step 15 — the previous *full* frame — not against the light frame before it.
    assert explain(two_hot_recording).narration[20] == (
        "Step 20: E* 0.8475 (rose by 0.3098 since step 15), Ê 6.6619 (fell by 1.3631), "
        "4 components (more than K = 2); 2 of 18 columns are 2-hot (+2 since step 15); "
        "mean R(v_j) 0.1986."
    )


def test_the_end_frame_is_narrated_from_the_numbers_the_run_reported(two_hot_recording):
    assert explain(two_hot_recording).narration[-1] == (
        "Run ended after 30 steps: E* 0.4056, \u00ca 2.9000, \u00ca \u2212 \u03a3\u03bb 2.8287, "
        "3 components; "
        "2 of 18 columns 2-hot on 2 distinct pairs, 0 of them edges of the graph."
    )


def test_the_ending_says_why_the_run_stopped_and_what_it_cannot_say(two_hot_recording):
    ending = explain(two_hot_recording).ending

    assert ending[0] == (
        "Stopped because the step budget of 30 steps was spent; the optimizer never stops early."
    )
    assert ending[-1] == "The datum this run is judged against is not in the recording."
    # A drift of 6.66e-16 printed to four decimals is a drift indistinguishable from none.
    assert "6.66e-16" in ending[-2]


def test_every_moment_sits_on_the_frame_the_frames_themselves_name(two_hot_recording):
    cluster_count = two_hot_recording.problem["cluster_count"]
    stepping = _stepping_frames(two_hot_recording)
    moments = {moment.label: moment.frame_index for moment in explain(two_hot_recording).moments}

    reached = [index for index, frame in stepping if frame["component_count"] == cluster_count]
    counted = [(index, _two_hot_by_hand(frame["spanning_set"])) for index, frame in stepping]
    first_two_hot = [index for index, (count, _columns, _pairs) in counted if count > 0]
    all_two_hot = [index for index, (count, columns, _pairs) in counted if count == columns]
    falls = [
        (before["rounded_cut"] - frame["rounded_cut"], index)
        for (_, before), (index, frame) in pairwise(stepping)
    ]

    # The roach at λ = 10 never gets to K = 2 components and never rounds every column, so those
    # two moments are absent rather than misplaced — which is the claim worth asserting.
    assert reached == [] and "K reached" not in moments
    assert all_two_hot == [] and "all 2-hot" not in moments
    assert moments["first 2-hot"] == first_two_hot[0] == 20
    assert moments["Ê drop"] == max(falls)[1] == 29
    # Ê last changed on the last full frame the run took, so nothing settled before the end.
    assert "Ê settled" not in moments


def test_moments_are_sorted_and_never_two_to_a_frame(two_hot_recording):
    moments = explain(two_hot_recording).moments
    indices = [moment.frame_index for moment in moments]

    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert all(len(moment.label.split()) <= 3 for moment in moments)


def test_every_legend_entry_is_a_thing_the_drawing_actually_tags(two_hot_recording):
    # The legend is the page's key to the picture; an entry nobody draws is a lie and a drawn
    # element nobody explains is the thing the legend exists to prevent.
    drawn = set(LEGEND_TAG.findall(_view_javascript()))
    explained = {entry.key for entry in explain(two_hot_recording).legend}

    assert drawn
    assert drawn == explained


def test_every_quantity_carries_one_value_per_frame(two_hot_recording):
    explanation = explain(two_hot_recording)

    assert {quantity.key for quantity in explanation.quantities} == {
        "relaxed_objective",
        "rounded_cut",
        "cut_minus_floor",
        "component_count",
        "two_hot_columns",
        "mean_collision",
        "training_loss",
    }
    for quantity in explanation.quantities:
        assert len(quantity.values) == len(two_hot_recording.frames), quantity.key


def test_a_light_frame_reports_only_the_number_the_recorder_took(two_hot_recording):
    # Carrying the last full frame's Ê onto a light frame would report a measurement nobody made.
    quantities = {
        quantity.key: quantity.values for quantity in explain(two_hot_recording).quantities
    }

    for index, frame in enumerate(two_hot_recording.frames):
        for key, values in quantities.items():
            if key == "training_loss":
                assert values[index] is not None
            else:
                assert (values[index] is None) == (not frame["full"]), (key, index)


def test_the_two_hot_column_count_reads_as_the_share_of_the_columns_it_is(two_hot_recording):
    # Every full frame, against the rule written out a second time in this module.
    values = {q.key: q.values for q in explain(two_hot_recording).quantities}["two_hot_columns"]

    for index, frame in enumerate(two_hot_recording.frames):
        if not frame["full"]:
            continue
        count, columns, _pairs = _two_hot_by_hand(frame["spanning_set"])
        assert values[index] == f"{count} of {columns}", index
    assert values[0] == "0 of 18"
    assert values[20] == "2 of 18"


def test_the_end_frame_counts_the_pairs_the_two_hot_columns_actually_name(two_hot_recording):
    # The distinct pairs and the ones that are edges, recomputed here from V and the instance's
    # own edge list, then assembled into the sentence the page prints.
    frame = two_hot_recording.frames[-1]
    edges = {tuple(edge) for edge in two_hot_recording.problem["layout"]["edges"]}
    count, columns, pairs = _two_hot_by_hand(frame["spanning_set"])
    distinct = sorted(set(pairs))
    on_edges = [pair for pair in distinct if pair in edges]

    assert (count, columns) == (2, 18)
    # Both pairs cross the roach's antennae, where the ladder has no rungs, so neither is an edge:
    # the relaxation is free to pair any two vertices, which is why the drawing dashes those.
    assert distinct == [(2, 12), (3, 13)]
    assert on_edges == []
    assert (
        explain(two_hot_recording)
        .narration[-1]
        .endswith(
            f"{count} of {columns} columns 2-hot on {len(distinct)} distinct pairs, "
            f"{len(on_edges)} of them edges of the graph."
        )
    )


def test_a_two_hot_columns_pair_is_the_one_the_recorder_rounded_it_to(two_hot_recording):
    # The view derives the pair from V; the recorder wrote one down at the time. When every column
    # rounded to a pair the two lists are index-aligned, so they can be compared column by column.
    for frame in _full_frames(two_hot_recording):
        recorded = frame["rounded_pairs"]
        matrix = np.asarray(frame["spanning_set"], dtype=float)
        if len(recorded) != matrix.shape[1]:
            continue
        for column in range(matrix.shape[1]):
            entries = matrix[:, column]
            norm = float(np.sum(entries**2))
            ranked = sorted(range(len(entries)), key=lambda row: (-abs(entries[row]), row))
            if norm <= 0.0 or entries[ranked[0]] ** 2 + entries[ranked[1]] ** 2 < (
                TWO_HOT_SHARE * norm
            ):
                continue
            assert (min(ranked[0], ranked[1]), max(ranked[0], ranked[1])) == (
                min(recorded[column]),
                max(recorded[column]),
            ), (frame["step"], column)


def test_a_column_is_two_hot_when_its_two_largest_squared_entries_hold_the_share():
    # A pure pair, a column that is nearly one, a column that is not, and a column the run switched
    # off. The share is on squared entries because the energy is what the projector sees.
    assert TWO_HOT_SHARE == 0.95

    pure = [[1.0], [-1.0], [0.0]]
    assert _two_hot_columns(pure) == (1, 1, [(0, 1)])

    # 90 % of the squared norm on two coordinates is below the 95 % line, so it is not yet a pair.
    held = 0.9
    nearly = [[(held / 2) ** 0.5], [-((held / 2) ** 0.5)], [(1 - held) ** 0.5]]
    assert _two_hot_columns(nearly) == (0, 1, [])

    # A column the run switched off holds nothing; rounding it would invent a pair.
    assert _two_hot_columns([[0.0], [0.0], [0.0]]) == (0, 1, [])


def test_the_page_carries_a_definition_for_every_term_it_uses_and_no_others(two_hot_recording):
    explanation = explain(two_hot_recording)
    text = "\n".join(
        list(explanation.opening)
        + list(explanation.narration)
        + list(explanation.ending)
        + [entry.name for entry in explanation.legend]
        + [entry.meaning for entry in explanation.legend]
        + [quantity.definition for quantity in explanation.quantities]
        + [quantity.term for quantity in explanation.quantities]
    )

    assert explanation.glossary
    assert set(explanation.glossary) <= set(GLOSSARY)
    for term, definition in explanation.glossary.items():
        assert definition == GLOSSARY[term]
        assert pattern_for(term).search(text), term


def test_the_two_hot_vocabulary_reaches_the_page_that_narrates_in_it(two_hot_recording):
    # The nine words this view was given a glossary for; a page that used none of them would mean
    # the narration had found a friendlier dialect.
    assert {
        "2-hot vector",
        "Collision measure",
        "Component count",
        "Relaxed objective",
        "Roundability",
        "Rounded cut",
        "Rounded pair",
        "Spanning set",
        "Spectral floor",
    } <= set(explain(two_hot_recording).glossary)


def test_the_explanation_rides_in_the_layout_the_page_embeds(two_hot_recording):
    layout = two_hot_span_view(two_hot_recording).layout

    assert layout["explain"] == explain(two_hot_recording).to_dict()
    assert json.loads(json.dumps(layout)) == layout
