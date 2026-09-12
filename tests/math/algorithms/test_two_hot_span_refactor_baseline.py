"""The two-hot optimizer's arithmetic, pinned to the last bit across the BL-48 refactor.

`CONTRIBUTING.md` asks for a snapshot of the real pipeline before code that trains is restructured.
This is that snapshot for the optimizer: three seeded runs on the roach G₅ instance, each exercising
a different set of training knobs, with every float the run produced written out exactly. Slices 1
to 4 of `docs/plans/2026-09-optimizer-object-model.md` must leave it byte-identical; a slice that
means to change a number regenerates it with ``BASELINE_UPDATE=1`` and says so in its PR. A missing
snapshot is a failure, never a skip: the only claim this file makes is byte-identity, and a guard
that regenerates itself when its reference is gone has no reference.

The three cells are chosen so that every term and every schedule the refactor moves is on somewhere:
the brief's objective alone; every penalty at once under a cosine schedule and the edge-product
form; the laplacian form under warm-up without column normalisation. Comparison is exact equality on
the JSON round-trip, which is lossless for doubles, on the platform that wrote the snapshot; on any
other platform the BLAS differs in its last digits and the comparison is to 1e-9.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from mllib.math.algorithms.two_hot_span.optimizer import (
    TwoHotSpanOptimizer,
    TwoHotSpanSettings,
    training_cost,
)
from mllib.math.algorithms.two_hot_span.penalties import (
    CollisionPenalty,
    DiversityPenalty,
    EdgeProductAdjacencyPenalty,
    LaplacianAdjacencyPenalty,
)
from mllib.math.algorithms.two_hot_span.projectors import RidgeProjector
from mllib.math.algorithms.two_hot_span.step_rules import AdamStepRule
from mllib.math.graph.two_hot_span_problem import TwoHotSpanProblem, roach_graph
from mllib.math.learning_rate_schedule import CosineSchedule, WarmupCosineSchedule
from mllib.math.recorder import AbstractStepRecorder

from .two_hot_span_support import HistoryRecorder

SNAPSHOT_PATH = Path(__file__).with_name("two_hot_span_refactor_snapshot.json")
# The snapshot records the platform that wrote it. On that platform the comparison is exact; on any
# other the BLAS differs in its last digits (see the walkthrough fixture's tolerance, commit
# b51d905), so the comparison is to 1e-9 there and the exact claim is the writing platform's.
PLATFORM_KEY = "platform"
CROSS_PLATFORM_TOLERANCE = 1e-9


def platform_tag() -> str:
    return f"{platform.system()}-{platform.machine()}"


CLUSTER_COUNT = 2

PROBLEM = TwoHotSpanProblem(roach_graph(5), CLUSTER_COUNT, name="roach_g5")
STEP_COUNT = 30


def objective_only(recorder: AbstractStepRecorder | None = None) -> TwoHotSpanOptimizer:
    return TwoHotSpanOptimizer(
        PROBLEM,
        training_cost(PROBLEM),
        [CollisionPenalty(weight=10.0)],
        settings=TwoHotSpanSettings(step_count=STEP_COUNT, seed=0),
        recorder=recorder,
    )


def every_penalty_cosine_edge_product(
    recorder: AbstractStepRecorder | None = None,
) -> TwoHotSpanOptimizer:
    return TwoHotSpanOptimizer(
        PROBLEM,
        training_cost(PROBLEM),
        [
            CollisionPenalty(weight=10.0),
            EdgeProductAdjacencyPenalty(PROBLEM.adjacency, weight=0.3),
            DiversityPenalty(weight=10.0),
        ],
        AdamStepRule(0.05, CosineSchedule(final_fraction=0.1)),
        settings=TwoHotSpanSettings(step_count=STEP_COUNT, seed=1),
        recorder=recorder,
    )


def laplacian_warmup_unnormalised(
    recorder: AbstractStepRecorder | None = None,
) -> TwoHotSpanOptimizer:
    return TwoHotSpanOptimizer(
        PROBLEM,
        training_cost(PROBLEM, RidgeProjector(1e-6)),
        [CollisionPenalty(weight=1.0), LaplacianAdjacencyPenalty(PROBLEM.laplacian, weight=0.5)],
        AdamStepRule(0.05, WarmupCosineSchedule(warmup_steps=5, final_fraction=0.05)),
        settings=TwoHotSpanSettings(step_count=STEP_COUNT, seed=2, normalize_columns=False),
        recorder=recorder,
    )


# Each cell is a factory (D-35 (5)): a fresh optimizer, a fresh step rule, per run.
CELLS = {
    "objective_only": objective_only,
    "every_penalty_cosine_edge_product": every_penalty_cosine_edge_product,
    "laplacian_warmup_unnormalised": laplacian_warmup_unnormalised,
}


def run_cells() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for name, build in CELLS.items():
        history = HistoryRecorder()
        run = build(recorder=history).run()
        results[name] = {
            "loss_history": history.losses,
            "learning_rate_history": history.learning_rates,
            "spanning_set": np.asarray(run.spanning_set, dtype=np.float64).tolist(),
            "relaxed_objective": float(run.relaxed_objective),
            "rounded_cut": float(run.rounded_cut),
            "collision_measures": [float(value) for value in run.collision_measures],
            "labels": [int(label) for label in run.labels],
            "max_zero_sum_violation": float(run.max_zero_sum_violation),
        }
    return results


@pytest.fixture(scope="module")
def results() -> dict[str, dict[str, object]]:
    return run_cells()


def test_every_cell_ran_its_step_budget(results):
    for name in CELLS:
        assert len(results[name]["loss_history"]) == STEP_COUNT, name


def test_snapshot_matches_to_the_last_bit(results):
    if os.environ.get("BASELINE_UPDATE") == "1":
        document = {PLATFORM_KEY: platform_tag(), "cells": results}
        SNAPSHOT_PATH.write_text(json.dumps(document, indent=2) + "\n")
        pytest.skip(f"refactor snapshot written to {SNAPSHOT_PATH.name}; re-run to compare")
    if not SNAPSHOT_PATH.exists():
        pytest.fail("refactor snapshot missing: regenerate deliberately with BASELINE_UPDATE=1")

    document = json.loads(SNAPSHOT_PATH.read_text())
    expected = document["cells"]
    exact = document[PLATFORM_KEY] == platform_tag()
    assert set(results) == set(expected)
    for name in CELLS:
        for key, value in expected[name].items():
            actual = results[name][key]
            if exact or key == "labels":
                assert actual == value, f"{name}.{key} moved"
            else:
                np.testing.assert_allclose(
                    np.asarray(actual, dtype=np.float64),
                    np.asarray(value, dtype=np.float64),
                    rtol=0.0,
                    atol=CROSS_PLATFORM_TOLERANCE,
                    err_msg=f"{name}.{key} moved by more than {CROSS_PLATFORM_TOLERANCE}",
                )
