"""The two-hot span optimizer: Adam on a training loss over a spanning set, as injected objects.

The optimizer of the rcut relaxation (`docs/plans/2026-09-two-hot-span.md`) on `AbstractOptimizer`:
it is handed a problem, a cost, its penalties, a step rule and a recorder, and owns only what is
its own — the seeded start, the projection every forward pass applies, the zero-sum and finiteness
checks, and the assembly of the result (D-35 (3), (9)). The trainable parameter is W (n x r); every
forward pass projects it to V, orthogonal to the problem's constraint vector, optionally with unit
columns. The training loss is the injected cost, the ridge projector for training, plus the
penalties (D-31); every delivered number comes back through the problem's exact arithmetic.
VᵀV = I is never imposed: nothing here orthogonalises. rcut only; ncut is BL-41.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mllib.describe import configuration_of
from mllib.math.algorithms.abstract_optimizer import (
    AbstractOptimizer,
    OptimizerResult,
    Stop,
    StopReason,
)
from mllib.math.algorithms.two_hot_span.projectors import RidgeProjector
from mllib.math.algorithms.two_hot_span.step_rules import AdamStepRule
from mllib.math.cost_function import AbstractCostFunction
from mllib.math.graph.two_hot_span_problem import SpanCost, TwoHotSpanProblem
from mllib.math.projector import AbstractProjector
from mllib.math.recorder import AbstractStepRecorder
from mllib.math.regularization_function import AbstractRegularizationFunction
from mllib.math.step_rule import AbstractStepRule


def training_cost(
    problem: TwoHotSpanProblem, projector: AbstractProjector | None = None
) -> SpanCost:
    """The span cost of the problem's incidence matrix in the training arithmetic.

    A composition root's one-liner: X as a float64 tensor under the ridge projector by default.
    """
    projector = RidgeProjector() if projector is None else projector
    return SpanCost(projector, torch.tensor(np.asarray(problem.X, dtype=np.float64)))


def training_loss(
    cost: AbstractCostFunction,
    spanning_set_t: torch.Tensor,
    penalties: Sequence[AbstractRegularizationFunction] = (),
) -> torch.Tensor:
    """The injected cost plus each injected penalty in turn, a float64 tensor on the gradient path.

    A cost in the reporting arithmetic is refused here rather than failing at `backward`. Each
    penalty carries its own weight and sign (`penalties.py`), so this is a plain left-to-right sum
    and the order of ``penalties`` is the order of the additions.
    """
    loss = cost.compute_cost(spanning_set_t)
    if not torch.is_tensor(loss):
        raise TypeError(
            f"the cost must be in the training arithmetic (a torch tensor); got {type(loss).__name__}"
        )
    for penalty in penalties:
        loss = loss + penalty.compute_penalty(spanning_set_t)
    return loss


def project_orthogonal(
    weights: torch.Tensor, constraint: torch.Tensor, normalize_columns: bool
) -> torch.Tensor:
    """V = W - c (cᵀW)/(cᵀc): every column orthogonal to ``constraint``, optionally unit length."""
    spanning_set_t = weights - torch.outer(constraint, constraint @ weights) / (
        constraint @ constraint
    )
    if normalize_columns:
        spanning_set_t = spanning_set_t / torch.linalg.vector_norm(spanning_set_t, dim=0)
    return spanning_set_t


@dataclass(frozen=True, slots=True)
class TwoHotSpanSettings:
    """The loop's own knobs (D-35 (4)); everything else is a knob of an injected object."""

    step_count: int = 200
    seed: int = 0
    normalize_columns: bool = True
    zero_sum_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.step_count < 0:
            raise ValueError(f"step_count must not be negative; got {self.step_count!r}")
        if self.zero_sum_tolerance < 0.0:
            raise ValueError(
                f"zero_sum_tolerance must not be negative; got {self.zero_sum_tolerance!r}"
            )


@dataclass(frozen=True, slots=True)
class TwoHotSpanResult(OptimizerResult):
    """The run's result: the spanning set it delivered and the numbers read off it (D-35 (9))."""

    relaxed_objective: float
    rounded_cut: float
    collision_measures: np.ndarray
    labels: np.ndarray
    max_zero_sum_violation: float

    @property
    def spanning_set(self) -> np.ndarray:
        """The delivered V: ``parameters`` under the name the domain uses."""
        return self.parameters


class TwoHotSpanOptimizer(AbstractOptimizer):
    """Adam on the ridge training loss of a spanning set; every reported field through numpy.

    ``initial_spanning_set`` is the start, or ``None`` for the seeded ``randn`` start. The step
    rule defaults to a fresh ``AdamStepRule()``; it is bound to this run's parameters at
    ``_begin`` and serves no other. A ``WarmupCosineSchedule`` whose warm-up reaches the step
    budget is refused at construction: its decay would never run.
    """

    def __init__(
        self,
        problem: TwoHotSpanProblem,
        cost: AbstractCostFunction,
        penalties: Sequence[AbstractRegularizationFunction] = (),
        step_rule: AbstractStepRule | None = None,
        *,
        settings: TwoHotSpanSettings | None = None,
        initial_spanning_set: np.ndarray | None = None,
        recorder: AbstractStepRecorder | None = None,
    ):
        self.settings = TwoHotSpanSettings() if settings is None else settings
        super().__init__(step_count=self.settings.step_count, recorder=recorder)
        self.problem = problem
        self.cost = cost
        self.penalties = tuple(penalties)
        expected = (problem.node_count, problem.node_count)
        for penalty in self.penalties:
            for matrix_name in ("laplacian", "adjacency"):
                matrix = getattr(penalty, matrix_name, None)
                if matrix is not None and tuple(matrix.shape) != expected:
                    raise ValueError(
                        f"{type(penalty).__name__}.{matrix_name} is {tuple(matrix.shape)}, not the "
                        f"problem's {expected}: the penalty is over another graph"
                    )
        self.step_rule: AbstractStepRule = AdamStepRule() if step_rule is None else step_rule
        self.initial_spanning_set = (
            None
            if initial_spanning_set is None
            else np.asarray(initial_spanning_set, dtype=np.float64)
        )
        warmup = int(getattr(getattr(self.step_rule, "schedule", None), "warmup_steps", 0))
        if warmup > 0 and warmup >= self.settings.step_count:
            raise ValueError(
                f"warmup_steps ({warmup}) must be shorter than step_count "
                f"({self.settings.step_count}), or the decay never runs"
            )
        self._constraint = torch.tensor(problem.constraint_vector, dtype=torch.float64)
        self._weights: torch.Tensor | None = None
        self._max_violation = 0.0
        self._learning_rate_in_force = 0.0

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            **{
                name: getattr(self.settings, name)
                for name in ("step_count", "seed", "normalize_columns", "zero_sum_tolerance")
            },
            "problem": self.problem.configuration,
            "cost": configuration_of(self.cost),
            "penalties": [configuration_of(penalty) for penalty in self.penalties],
            "step_rule": configuration_of(self.step_rule),
            "initial_spanning_set": "given" if self.initial_spanning_set is not None else "seeded",
        }

    # ----------------------------------------------------------------------------------------
    # The seams of AbstractOptimizer.
    # ----------------------------------------------------------------------------------------

    def _begin(self) -> Stop | None:
        node_total = self.problem.node_count
        spanning_count = self.problem.spanning_vector_count
        if self.initial_spanning_set is None:
            generator = torch.Generator().manual_seed(self.settings.seed)
            initial = torch.randn(
                node_total, spanning_count, dtype=torch.float64, generator=generator
            )
        else:
            initial = torch.tensor(self.initial_spanning_set)
            if initial.shape != (node_total, spanning_count):
                raise ValueError(
                    f"initial_spanning_set must be {(node_total, spanning_count)}; "
                    f"got {tuple(initial.shape)}"
                )
        self._weights = initial.clone().requires_grad_(True)
        if not bool(torch.isfinite(initial).all()):
            return Stop(StopReason.NON_FINITE, "the initial spanning set is not finite")
        self.step_rule.bind(self._weights, self.settings.step_count)
        return self._inspect("at initialisation")

    def _step(self, step: int) -> float | Stop:
        weights = self._parameters()
        self.step_rule.zero_gradient()
        self._learning_rate_in_force = self.step_rule.learning_rate_in_force()
        spanning_set_t = self._project(weights)
        loss = training_loss(self.cost, spanning_set_t, self.penalties)
        if not bool(torch.isfinite(loss)):
            return Stop(StopReason.NON_FINITE, f"the loss is not finite at step {step}")
        loss.backward()
        if weights.grad is None or not bool(torch.isfinite(weights.grad).all()):
            return Stop(StopReason.NON_FINITE, f"the gradient is not finite at step {step}")
        self.step_rule.step()
        stop = self._inspect(f"after step {step}")
        if stop is not None:
            return stop
        return float(loss.detach())

    def _iterate(self) -> np.ndarray:
        with torch.no_grad():
            current = self._project(self._parameters())
        return np.ascontiguousarray(current.numpy(), dtype=np.float64)

    def _recorder_extras(self) -> dict[str, Any]:
        return {"learning_rate": self._learning_rate_in_force}

    def _assemble(
        self, steps_taken: int, final_training_loss: float | None, stop: Stop
    ) -> TwoHotSpanResult:
        spanning_set = self._iterate()
        report = self.problem.report(spanning_set)
        return TwoHotSpanResult(
            parameters=spanning_set,
            final_training_loss=final_training_loss,
            steps_taken=steps_taken,
            stop_reason=stop.reason,
            stop_detail=stop.detail,
            configuration=self.configuration,
            relaxed_objective=report.relaxed_objective,
            rounded_cut=report.rounded_cut,
            collision_measures=report.collision_measures,
            labels=report.labels,
            max_zero_sum_violation=self._max_violation,
        )

    # ----------------------------------------------------------------------------------------
    # The optimizer's own arithmetic.
    # ----------------------------------------------------------------------------------------

    def _parameters(self) -> torch.Tensor:
        if self._weights is None:
            raise RuntimeError("the run has not begun; run() calls _begin() first")
        return self._weights

    def _project(self, weights: torch.Tensor) -> torch.Tensor:
        return project_orthogonal(weights, self._constraint, self.settings.normalize_columns)

    def _inspect(self, moment: str) -> Stop | None:
        """The constraint and finiteness checks on the current V, as a stop rather than a raise."""
        with torch.no_grad():
            current = self._project(self._parameters())
            if not bool(torch.isfinite(current).all()):
                return Stop(StopReason.NON_FINITE, f"the spanning set is not finite {moment}")
            # cᵀ v_j as a weighted column sum: for c = ones it is the plain column sum, to the
            # bit, and for any other constraint vector it is the right inner product (BL-41).
            weighted = current * self._constraint[:, None]
            violation = float(torch.max(torch.abs(weighted.sum(dim=0))))
            self._max_violation = max(self._max_violation, violation)
            if violation > self.settings.zero_sum_tolerance:
                return Stop(
                    StopReason.CONSTRAINT_VIOLATION,
                    f"max |cᵀ v_j| = {violation:.3e} exceeds "
                    f"{self.settings.zero_sum_tolerance:.3e} {moment}",
                )
        return None
