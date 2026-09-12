"""Composing a two-hot span optimizer from names and knobs: the composition roots' one home.

A harness, an example or a stress cell arrives with a schedule *name*, an adjacency *form* and a
handful of weights off a command line. Turning those into math objects is a composition root's
job (D-35 (8)), and it is done once, here, rather than in every script that runs the optimizer.
Nothing below `mllib.ml` knows these names: the math objects take objects, not strings. A zero
weight builds no penalty (`penalties.py`); a fresh optimizer and a fresh step rule come back on
every call, so a grid is a grid over calls (D-35 (5)).
"""

from __future__ import annotations

import numpy as np

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
from mllib.math.graph.two_hot_span_problem import TwoHotSpanProblem
from mllib.math.learning_rate_schedule import (
    AbstractLearningRateSchedule,
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    WarmupCosineSchedule,
)
from mllib.math.recorder import AbstractStepRecorder
from mllib.math.regularization_function import AbstractRegularizationFunction

SCHEDULE_NAMES = ("constant", "cosine", "warmup_cosine", "linear")
ADJACENCY_FORM_NAMES = ("laplacian", "edge_product")

DEFAULT_STEP_COUNT = 200
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EPSILON = 1e-6


def schedule_from_name(
    name: str, *, final_fraction: float = 0.0, warmup_steps: int = 0
) -> AbstractLearningRateSchedule:
    """The schedule object a command-line name asks for, with the shape knobs it takes."""
    if name == "constant":
        return ConstantSchedule()
    if name == "linear":
        return LinearSchedule(final_fraction=float(final_fraction))
    if name == "cosine":
        return CosineSchedule(final_fraction=float(final_fraction))
    if name == "warmup_cosine":
        return WarmupCosineSchedule(
            warmup_steps=int(warmup_steps), final_fraction=float(final_fraction)
        )
    raise ValueError(f"unknown schedule {name!r}; expected one of {list(SCHEDULE_NAMES)}")


def adjacency_penalty_from_name(
    form: str, problem: TwoHotSpanProblem, weight: float
) -> AbstractRegularizationFunction:
    """The graph penalty a form name asks for, over the problem's own Laplacian or adjacency."""
    if form == "laplacian":
        return LaplacianAdjacencyPenalty(problem.laplacian, weight=float(weight))
    if form == "edge_product":
        return EdgeProductAdjacencyPenalty(problem.adjacency, weight=float(weight))
    raise ValueError(
        f"unknown adjacency_form {form!r}; expected one of {list(ADJACENCY_FORM_NAMES)}"
    )


def penalties_from_weights(
    problem: TwoHotSpanProblem,
    *,
    collision_weight: float = 0.0,
    adjacency_weight: float = 0.0,
    adjacency_form: str = "laplacian",
    diversity_weight: float = 0.0,
) -> tuple[AbstractRegularizationFunction, ...]:
    """The penalties the weights ask for, in the order the loss adds them; a zero weight is absent."""
    if adjacency_form not in ADJACENCY_FORM_NAMES:
        raise ValueError(
            f"unknown adjacency_form {adjacency_form!r}; expected one of {list(ADJACENCY_FORM_NAMES)}"
        )
    for name, weight in (
        ("collision_weight", collision_weight),
        ("adjacency_weight", adjacency_weight),
        ("diversity_weight", diversity_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must not be negative; got {weight!r}")
    penalties: list[AbstractRegularizationFunction] = []
    if collision_weight != 0.0:
        penalties.append(CollisionPenalty(weight=float(collision_weight)))
    if adjacency_weight != 0.0:
        penalties.append(adjacency_penalty_from_name(adjacency_form, problem, adjacency_weight))
    if diversity_weight != 0.0:
        penalties.append(DiversityPenalty(weight=float(diversity_weight)))
    return tuple(penalties)


def compose_two_hot_span(
    problem: TwoHotSpanProblem,
    *,
    step_count: int = DEFAULT_STEP_COUNT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon: float = DEFAULT_EPSILON,
    seed: int = 0,
    normalize_columns: bool = True,
    zero_sum_tolerance: float = 1e-12,
    schedule: str = "constant",
    warmup_steps: int = 0,
    final_learning_rate_fraction: float = 0.0,
    collision_weight: float = 0.0,
    adjacency_weight: float = 0.0,
    adjacency_form: str = "laplacian",
    diversity_weight: float = 0.0,
    initial_spanning_set: np.ndarray | None = None,
    recorder: AbstractStepRecorder | None = None,
) -> TwoHotSpanOptimizer:
    """A fresh optimizer from names and knobs: the one place a script needs to know the classes."""
    if learning_rate < 0.0:
        raise ValueError(f"learning_rate must not be negative; got {learning_rate!r}")
    return TwoHotSpanOptimizer(
        problem,
        training_cost(problem, RidgeProjector(epsilon=float(epsilon))),
        penalties_from_weights(
            problem,
            collision_weight=collision_weight,
            adjacency_weight=adjacency_weight,
            adjacency_form=adjacency_form,
            diversity_weight=diversity_weight,
        ),
        AdamStepRule(
            float(learning_rate),
            schedule_from_name(
                schedule, final_fraction=final_learning_rate_fraction, warmup_steps=warmup_steps
            ),
        ),
        settings=TwoHotSpanSettings(
            step_count=int(step_count),
            seed=int(seed),
            normalize_columns=bool(normalize_columns),
            zero_sum_tolerance=float(zero_sum_tolerance),
        ),
        initial_spanning_set=initial_spanning_set,
        recorder=recorder,
    )
