"""Composing a two-hot span optimizer inside `tests/math`, from math objects alone.

The math tests are composition roots (D-35 (8)) and may name concrete classes, but they must not
reach up into `mllib.ml` for the name-to-object mapping that lives there. This helper is the
one place the math tests wire a problem, a cost, its penalties and a step rule together; a zero
weight builds no penalty, as the loss has always had it, and every call returns fresh objects.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from mllib.math.algorithms.two_hot_span.optimizer import (
    TwoHotSpanOptimizer,
    TwoHotSpanResult,
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
from mllib.math.graph.two_hot_span_problem import TwoHotSpanProblem
from mllib.math.learning_rate_schedule import AbstractLearningRateSchedule
from mllib.math.recorder import AbstractStepRecorder
from mllib.math.regularization_function import AbstractRegularizationFunction


class HistoryRecorder(AbstractStepRecorder):
    """The per-step training loss and learning rate, off the recorder where they now live."""

    def __init__(self) -> None:
        super().__init__()
        self.losses: list[float] = []
        self.learning_rates: list[float] = []

    def record_step(self, step, training_loss, spanning_set, **extras) -> None:
        self.losses.append(float(training_loss))
        self.learning_rates.append(float(extras["learning_rate"]))

    def record_end(self, run, **extras) -> None:
        pass

    def describe_result(self, result) -> dict[str, object]:
        return {}

    @property
    def loss_history(self) -> np.ndarray:
        return np.array(self.losses, dtype=np.float64)

    @property
    def learning_rate_history(self) -> np.ndarray:
        return np.array(self.learning_rates, dtype=np.float64)


def problem_for(graph: nx.Graph, cluster_count: int) -> TwoHotSpanProblem:
    return TwoHotSpanProblem(graph, cluster_count, name="test")


def optimizer_for(
    graph: nx.Graph,
    cluster_count: int,
    *,
    step_count: int,
    seed: int = 0,
    learning_rate: float = 0.05,
    epsilon: float = 1e-6,
    normalize_columns: bool = True,
    zero_sum_tolerance: float = 1e-12,
    schedule: AbstractLearningRateSchedule | None = None,
    collision_weight: float = 0.0,
    adjacency_weight: float = 0.0,
    adjacency_form: str = "laplacian",
    diversity_weight: float = 0.0,
    initial: np.ndarray | None = None,
    recorder: AbstractStepRecorder | None = None,
) -> TwoHotSpanOptimizer:
    """A fresh optimizer over the graph, the knobs spelled out as objects."""
    problem = problem_for(graph, cluster_count)
    penalties: list[AbstractRegularizationFunction] = []
    if collision_weight != 0.0:
        penalties.append(CollisionPenalty(weight=collision_weight))
    if adjacency_weight != 0.0:
        if adjacency_form == "edge_product":
            penalties.append(
                EdgeProductAdjacencyPenalty(problem.adjacency, weight=adjacency_weight)
            )
        else:
            penalties.append(LaplacianAdjacencyPenalty(problem.laplacian, weight=adjacency_weight))
    if diversity_weight != 0.0:
        penalties.append(DiversityPenalty(weight=diversity_weight))
    return TwoHotSpanOptimizer(
        problem,
        training_cost(problem, RidgeProjector(epsilon)),
        penalties,
        AdamStepRule(learning_rate, schedule),
        settings=TwoHotSpanSettings(
            step_count=step_count,
            seed=seed,
            normalize_columns=normalize_columns,
            zero_sum_tolerance=zero_sum_tolerance,
        ),
        initial_spanning_set=initial,
        recorder=recorder,
    )


def run(graph: nx.Graph, cluster_count: int, **knobs) -> TwoHotSpanResult:
    """One run, its result."""
    return optimizer_for(graph, cluster_count, **knobs).run()


def run_with_history(
    graph: nx.Graph, cluster_count: int, **knobs
) -> tuple[TwoHotSpanResult, HistoryRecorder]:
    """One run, its result and the per-step histories a recorder collected."""
    history = HistoryRecorder()
    return optimizer_for(graph, cluster_count, recorder=history, **knobs).run(), history
