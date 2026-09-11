"""The two-hot span recorder against the real optimizer, on a seeded nine-node graph.

Every claim here is either about the run being unchanged by being watched, or about what the frames
say being exactly what the numpy `pinv` path says about the V the frame carries. The numbers
themselves are the optimizer's and the harness's business
(`tests/math/algorithms/test_two_hot_span_optimizer.py`).
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("torch")

from mllib.math.algorithms.two_hot_span_optimizer import (
    TwoHotSpanConfig,
    fit_two_hot_span,
)
from mllib.math.graph.two_hot_span_problem import (
    incidence_matrix,
    rounded_cut,
    spectral_floor,
)
from mllib.math.projector import ExactProjector
from mllib.math.recorder import NullRecorder
from mllib.visualization.recorders.two_hot_span import TwoHotSpanRecorder

NODE_TOTAL = 9
CLUSTER_COUNT = 2
STEP_COUNT = 20
SEED = 0


def _seeded_graph(node_total: int = NODE_TOTAL, seed: int = SEED) -> nx.Graph:
    """A connected seeded random graph with unit weights, as the optimizer's own tests build one."""
    graph = nx.gnp_random_graph(node_total, 0.55, seed=seed)
    if not nx.is_connected(graph):
        graph = nx.connected_watts_strogatz_graph(node_total, 4, 0.3, seed=seed)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def _config(step_count: int = STEP_COUNT) -> TwoHotSpanConfig:
    return TwoHotSpanConfig(step_count=step_count, collision_weight=1.0, seed=SEED)


def _watched(frame_every: int = 1, step_count: int = STEP_COUNT):
    """One run with a recorder attached; returns the graph, X, the run and the recorder."""
    graph = _seeded_graph()
    X = incidence_matrix(graph)
    recorder = TwoHotSpanRecorder(X, graph, CLUSTER_COUNT, step_count, frame_every=frame_every)
    run = fit_two_hot_span(X, CLUSTER_COUNT, _config(step_count), recorder=recorder)
    return graph, X, run, recorder


def test_an_explicit_null_recorder_leaves_the_run_bit_for_bit_unchanged():
    # The whole point of the guard: observation that is off costs one attribute read per step and
    # changes nothing the run computes.
    graph = _seeded_graph()
    X = incidence_matrix(graph)

    unwatched = fit_two_hot_span(X, CLUSTER_COUNT, _config())
    nulled = fit_two_hot_span(X, CLUSTER_COUNT, _config(), recorder=NullRecorder())

    assert np.array_equal(unwatched.spanning_set, nulled.spanning_set)
    assert np.array_equal(unwatched.loss_history, nulled.loss_history)
    assert unwatched.relaxed_objective == nulled.relaxed_objective
    assert unwatched.rounded_cut == nulled.rounded_cut


def test_the_null_recorder_raises_rather_than_swallowing_a_step():
    # A null recorder handed a frame means a missing guard, which is a bug and not a no-op.
    with pytest.raises(RuntimeError, match="guard is missing"):
        NullRecorder().record_step(0, 1.0, np.zeros((3, 2)))


def test_a_verbose_recorder_leaves_one_frame_per_step_and_one_at_the_end():
    _, _, run, recorder = _watched(frame_every=1)

    assert len(recorder.frames) == STEP_COUNT + 1
    assert [frame.step for frame in recorder.frames] == [*range(STEP_COUNT), STEP_COUNT - 1]
    assert [frame.index for frame in recorder.frames] == list(range(STEP_COUNT + 1))
    assert [frame.end for frame in recorder.frames] == [False] * STEP_COUNT + [True]
    assert run.loss_history.size == STEP_COUNT


def test_every_frames_training_loss_is_the_runs_loss_at_that_step():
    _, _, run, recorder = _watched(frame_every=1)

    for frame in recorder.frames[:STEP_COUNT]:
        assert frame.training_loss == float(run.loss_history[frame.step])
    # The end frame restates the last step's training loss, which is the run's final one.
    assert recorder.frames[-1].training_loss == float(run.loss_history[-1])


def test_every_full_frames_numbers_are_the_pinv_numbers_of_the_v_it_carries():
    # Exact equality, not a tolerance: the frame's V is the V the functions are called on, so a
    # difference of any size would mean the recorder computed something else.
    _, X, _, recorder = _watched(frame_every=1)

    full = [frame for frame in recorder.frames if frame.full and not frame.end]
    assert full
    for frame in full:
        spanning_set = np.asarray(frame.spanning_set, dtype=float)
        assert frame.relaxed_objective == ExactProjector().residual(X, spanning_set)
        assert frame.rounded_cut == rounded_cut(X, spanning_set)
        assert frame.rounded_cut_minus_relaxed == frame.rounded_cut - frame.relaxed_objective


def test_the_end_frame_restates_the_runs_own_reported_numbers():
    graph, _, run, recorder = _watched(frame_every=1)
    floor = spectral_floor(graph, CLUSTER_COUNT)

    end = recorder.frames[-1]
    assert end.end is True
    assert end.full is True
    assert end.relaxed_objective == run.relaxed_objective
    assert end.rounded_cut == run.rounded_cut
    assert list(end.labels) == [int(label) for label in run.labels]
    assert end.component_count == int(np.unique(run.labels).size)
    assert end.rounded_cut_minus_floor == run.rounded_cut - floor


def test_frame_every_five_makes_full_frames_at_the_cycle_and_at_the_last_step():
    _, _, _, recorder = _watched(frame_every=5)

    steps = recorder.frames[:STEP_COUNT]
    assert [frame.step for frame in steps if frame.full] == [0, 5, 10, 15, 19]
    light = [frame for frame in steps if not frame.full]
    assert [frame.step for frame in light] == [
        step for step in range(STEP_COUNT) if step not in {0, 5, 10, 15, 19}
    ]
    for frame in light:
        # A light frame carries the curve and nothing else; every derived field is absent.
        assert frame.spanning_set is None
        assert frame.collision_measures is None
        assert frame.rounded_pairs is None
        assert frame.labels is None
        assert frame.component_count is None
        assert frame.relaxed_objective is None
        assert frame.rounded_cut is None


def test_every_caption_is_a_sentence_that_names_its_step_and_never_calls_the_loss_e():
    _, _, _, recorder = _watched(frame_every=5)

    for frame in recorder.frames[:STEP_COUNT]:
        assert frame.caption.startswith(f"Step {frame.step}: training loss ")
        assert frame.caption.endswith(".")
        if frame.full:
            assert "E* " in frame.caption
            assert "components" in frame.caption or "component" in frame.caption
            assert "mean R(v_j)" in frame.caption
    end = recorder.frames[-1]
    assert end.caption.startswith(f"Run ended after {STEP_COUNT} steps: ")
    assert end.caption.endswith(".")
    # The training loss is the ridge one and is never reported as E, in any frame's caption.
    assert not any("training loss E" in frame.caption for frame in recorder.frames)


def test_every_frame_and_the_described_result_serialise_without_a_default():
    # json.dumps with no ``default`` is the test that no numpy scalar survived the conversion: a
    # np.float64 would raise here rather than at write time in a different module.
    _, _, run, recorder = _watched(frame_every=5)

    for frame_dict in recorder.frame_dicts():
        json.dumps(frame_dict)
    described = recorder.describe_result(run)
    json.dumps(described)
    assert described["spectral_floor"] == recorder.spectral_floor
    assert described["relaxed_objective"] == run.relaxed_objective
    assert described["config"]["step_count"] == STEP_COUNT


def test_a_frame_every_below_one_is_refused():
    graph = _seeded_graph()
    with pytest.raises(ValueError, match="frame_every"):
        TwoHotSpanRecorder(incidence_matrix(graph), graph, CLUSTER_COUNT, STEP_COUNT, frame_every=0)
