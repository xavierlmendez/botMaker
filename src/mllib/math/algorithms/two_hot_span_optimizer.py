"""The direct PyTorch optimizer for the two-hot span rcut relaxation.

Slice 2.2 of the prototype in `docs/plans/2026-09-two-hot-span.md`.

The trainable parameter is W (n x r). Every forward pass projects it to the zero-sum spanning set
V = W - 1 (1ᵀ W)/n, optionally normalizing each column by its 2-norm. The *training* loss uses the
ridge projector V (VᵀV + eps I)⁻¹ Vᵀ (D-31, P-2); every *reported* number is recomputed by the numpy
functions of `two_hot_span_problem` on the final V with the exact pinv projector, so `epsilon` never
reaches a report.

VᵀV = I is never imposed: there is no QR and no orthogonalisation anywhere in this module. This is
the gradient track only; the alternating analytical solver is parked. Ê (`rounded_cut`) is the
headline number and R(v_j) (`collision_measures`) is a per-column diagnostic. rcut only — the ncut
generalisation replaces the ones vector by c = sqrt(d) and is out of scope here (BL-41).

**Three experimental training knobs, all off by default (slices 2.4 and 2.5).** A learning-rate
schedule (`learning_rate_schedule`, `warmup_steps`, `final_learning_rate_fraction`), an optional
per-column graph term (`adjacency_weight`, `adjacency_form`), and an optional column-diversity term
(`diversity_weight`). With the shipped defaults — `"constant"`, `adjacency_weight = 0.0` and
`diversity_weight = 0.0` — none of them runs, and the objective is exactly the brief's §20
objective, `E_ridge(V) - λ Σ_j R(v_j)`. They are knobs of the *training* path in the sense D-31
fixes for `epsilon`: they change the trajectory Adam takes and never enter a reported number. Every
reported field still comes back through the exact numpy `pinv` projector of `two_hot_span_problem`.

The diversity term is the first one that **couples the columns**: the schedule and the graph term
both score a column on its own, and the span term is invariant to rotation inside the span, so
nothing before slice 2.5 could tell two columns apart from one column used twice.

The three terms are injected math objects (`two_hot_span/penalties.py`, D-35 (3)) and the training
loss is their plain sum over the ridge cost. Until slice 4 of
`docs/plans/2026-09-optimizer-object-model.md` hands the optimizer its penalties directly, this
module still builds them from the config's weights, in `_penalties_from_config`, so that every
caller's `TwoHotSpanConfig` keeps working; a zero weight builds no term (`penalties.py`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from mllib.math.algorithms.two_hot_span.penalties import (
    CollisionPenalty,
    DiversityPenalty,
    EdgeProductAdjacencyPenalty,
    LaplacianAdjacencyPenalty,
)
from mllib.math.graph.two_hot_span_problem import (
    clustering_from_pairs,
    collision_measures,
    graph_matrices,
    projector_residual,
    rounded_cut,
    rounded_pairs,
    spanning_vector_count,
)
from mllib.math.recorder import AbstractStepRecorder, NullRecorder
from mllib.math.regularization_function import AbstractRegularizationFunction

LEARNING_RATE_SCHEDULES = ("constant", "cosine", "warmup_cosine", "linear")
ADJACENCY_FORMS = ("laplacian", "edge_product")


@dataclass(frozen=True, slots=True)
class TwoHotSpanConfig:
    """The training configuration. Defaults are the brief's objective, unscheduled.

    `learning_rate_schedule`, `warmup_steps`, `final_learning_rate_fraction`, `adjacency_weight` and
    `adjacency_form` are slice 2.4's experiment and `diversity_weight` is slice 2.5's; at their
    defaults (`"constant"`, 0, 0.0, 0.0, `"laplacian"`, 0.0) the run is bit for bit the run slice
    2.2 shipped.
    """

    step_count: int = 200
    learning_rate: float = 0.05
    collision_weight: float = 0.0
    epsilon: float = 1e-6
    normalize_columns: bool = True
    seed: int = 0
    zero_sum_tolerance: float = 1e-12
    learning_rate_schedule: str = "constant"
    warmup_steps: int = 0
    final_learning_rate_fraction: float = 0.0
    adjacency_weight: float = 0.0
    adjacency_form: str = "laplacian"
    diversity_weight: float = 0.0

    def __post_init__(self) -> None:
        """Reject a misspelled name or a negative weight here, not 200 steps into a sweep."""
        if self.learning_rate_schedule not in LEARNING_RATE_SCHEDULES:
            raise ValueError(
                f"unknown learning_rate_schedule {self.learning_rate_schedule!r}; "
                f"expected one of {list(LEARNING_RATE_SCHEDULES)}"
            )
        if self.adjacency_form not in ADJACENCY_FORMS:
            raise ValueError(
                f"unknown adjacency_form {self.adjacency_form!r}; "
                f"expected one of {list(ADJACENCY_FORMS)}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must not be negative; got {self.warmup_steps!r}")
        if self.learning_rate_schedule == "warmup_cosine" and self.warmup_steps >= self.step_count:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be shorter than step_count "
                f"({self.step_count}) under warmup_cosine, or the decay never runs"
            )
        if self.adjacency_weight < 0.0:
            raise ValueError(
                f"adjacency_weight must not be negative; got {self.adjacency_weight!r}"
            )
        if self.diversity_weight < 0.0:
            raise ValueError(
                f"diversity_weight must not be negative; got {self.diversity_weight!r}"
            )
        if self.learning_rate < 0.0:
            raise ValueError(f"learning_rate must not be negative; got {self.learning_rate!r}")
        if not 0.0 <= self.final_learning_rate_fraction <= 1.0:
            raise ValueError(
                "final_learning_rate_fraction must lie in [0, 1]; "
                f"got {self.final_learning_rate_fraction!r}"
            )


class ZeroSumViolation(RuntimeError):  # noqa: N818 - names fixed by the plan
    """A column of V drifted off the zero-sum constraint. A stop, never a warning."""


class NonFiniteLoss(FloatingPointError):  # noqa: N818 - names fixed by the plan
    """NaN or inf appeared in the loss, the gradients or V. A stop, never a warning."""


@dataclass(frozen=True, slots=True)
class TwoHotSpanRun:
    spanning_set: np.ndarray
    loss_history: np.ndarray
    relaxed_objective: float
    rounded_cut: float
    collision_measures: np.ndarray
    labels: np.ndarray
    max_zero_sum_violation: float
    config: TwoHotSpanConfig
    learning_rate_history: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


def learning_rate_lambda(config: TwoHotSpanConfig) -> Callable[[int], float]:
    """The `LambdaLR` multiplier for step t, 1.0 at t = 0 for every schedule but `warmup_cosine`.

    One factor function, not four schedulers: the whole point of `LambdaLR` here is that a single
    Adam instance survives the run, so its moment estimates are never reset (which restarting an
    optimizer per phase would do, and which is a change to the *trajectory*, not to the step size).

    `final_learning_rate_fraction` is the fraction of `learning_rate` in force at the **last** step,
    so `cosine` and `linear` interpolate over `step_count - 1` steps. `warmup_cosine` starts at 0,
    rises linearly to 1 at step `warmup_steps`, and cosines down from there.
    """
    name = config.learning_rate_schedule
    fraction = float(config.final_learning_rate_fraction)
    last_step = max(config.step_count - 1, 1)
    warmup = int(config.warmup_steps)

    def factor(step: int) -> float:
        if name == "constant":
            return 1.0
        if name == "linear":
            return 1.0 + (fraction - 1.0) * (step / last_step)
        if name == "cosine":
            return fraction + (1.0 - fraction) * 0.5 * (1.0 + math.cos(math.pi * step / last_step))
        # warmup_cosine
        if warmup > 0 and step < warmup:
            return step / warmup
        remaining = config.step_count - 1 - warmup
        if remaining <= 0:
            return 1.0
        progress = (step - warmup) / remaining
        return fraction + (1.0 - fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def training_loss(
    X_t: torch.Tensor,
    spanning_set_t: torch.Tensor,
    epsilon: float,
    penalties: Sequence[AbstractRegularizationFunction] = (),
) -> torch.Tensor:
    """E_ridge(V) plus each injected penalty in turn, float64.

    The r x r system is solved with `torch.linalg.solve`; nothing is orthogonalised. Each penalty
    carries its own weight and sign (`two_hot_span/penalties.py`), so this is a plain left-to-right
    sum and the order of ``penalties`` is the order of the additions. With no penalties the value is
    the ridge cost alone, the arithmetic slice 2.2 shipped.
    """
    spanning_count = spanning_set_t.shape[1]
    gram = spanning_set_t.T @ spanning_set_t
    ridge = gram + epsilon * torch.eye(
        spanning_count, dtype=spanning_set_t.dtype, device=spanning_set_t.device
    )
    coefficients = torch.linalg.solve(ridge, spanning_set_t.T @ X_t)
    residual = X_t - spanning_set_t @ coefficients
    loss = torch.sum(residual * residual)
    for penalty in penalties:
        loss = loss + penalty.compute_penalty(spanning_set_t)
    return loss


def _penalties_from_config(
    config: TwoHotSpanConfig, X: np.ndarray
) -> tuple[AbstractRegularizationFunction, ...]:
    """The penalties a config's weights ask for, in the order the loss has always added them.

    A zero weight builds no term (`penalties.py`), so a default run is bit for bit the run slice
    2.2 shipped. This adapter is transitional: slice 4 of the BL-48 plan injects the penalties into
    the optimizer and the weights leave the config.
    """
    penalties: list[AbstractRegularizationFunction] = []
    if config.collision_weight != 0.0:
        penalties.append(CollisionPenalty(weight=config.collision_weight))
    if config.adjacency_weight != 0.0:
        laplacian, adjacency = graph_matrices(X)
        if config.adjacency_form == "edge_product":
            penalties.append(EdgeProductAdjacencyPenalty(adjacency, weight=config.adjacency_weight))
        else:
            penalties.append(LaplacianAdjacencyPenalty(laplacian, weight=config.adjacency_weight))
    if config.diversity_weight != 0.0:
        penalties.append(DiversityPenalty(weight=config.diversity_weight))
    return tuple(penalties)


def _project_zero_sum(weights: torch.Tensor, normalize_columns: bool) -> torch.Tensor:
    """V = W - 1 (1ᵀ W)/n. The ones vector is written out so ncut is a parameter change later."""
    node_total = weights.shape[0]
    ones = torch.ones(node_total, dtype=weights.dtype, device=weights.device)
    spanning_set_t = weights - torch.outer(ones, ones @ weights) / node_total
    if normalize_columns:
        spanning_set_t = spanning_set_t / torch.linalg.vector_norm(spanning_set_t, dim=0)
    return spanning_set_t


def fit_two_hot_span(
    X: np.ndarray,
    cluster_count: int,
    config: TwoHotSpanConfig,
    initial_spanning_set: np.ndarray | None = None,
    *,
    recorder: AbstractStepRecorder | None = None,
) -> TwoHotSpanRun:
    """Adam on the ridge-projector loss; every reported field comes back through numpy pinv.

    The schedule, the adjacency term and the diversity term are *training* knobs, in exactly the
    sense D-31 fixes for `epsilon`: they shape the trajectory and never appear in E\\*, Ê, R(v_j)
    or the clustering, all of which are recomputed after the loop by `two_hot_span_problem` through
    the exact `pinv` projector. At their defaults the loop is the one slice 2.2 shipped, arithmetic
    included.

    ``recorder`` is an optional collaborator, off unless one is passed (D-32). It is handed the
    step number, the training loss the step took a gradient of, and V as it stands — the objects the
    loop already holds — and nothing is computed here for its benefit: E\\*, Ê, R(v_j) and the
    rounding are the recorder's own work, through the numpy ``pinv`` path, in ``visualization``.
    With the default ``NullRecorder`` the guard is one attribute read per step and the run is
    identical.
    """
    recorder = recorder or NullRecorder()
    node_total = X.shape[0]
    spanning_count = spanning_vector_count(node_total, cluster_count)

    if initial_spanning_set is None:
        generator = torch.Generator().manual_seed(config.seed)
        initial = torch.randn(node_total, spanning_count, dtype=torch.float64, generator=generator)
    else:
        initial = torch.tensor(np.asarray(initial_spanning_set, dtype=np.float64))
        if initial.shape != (node_total, spanning_count):
            raise ValueError(
                f"initial_spanning_set must be {(node_total, spanning_count)}; "
                f"got {tuple(initial.shape)}"
            )
    if not bool(torch.isfinite(initial).all()):
        raise NonFiniteLoss("the initial spanning set is not finite")

    X_t = torch.tensor(np.asarray(X, dtype=np.float64))
    weights = initial.clone().requires_grad_(True)
    # One Adam for the whole run, its step size scaled by a LambdaLR: rebuilding the optimizer per
    # phase would reset the moment estimates, which changes the trajectory and not just the lr.
    optimizer = torch.optim.Adam([weights], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_lambda(config))

    penalties = _penalties_from_config(config, X)

    max_violation = 0.0
    losses: list[float] = []
    learning_rates: list[float] = []

    def inspect(step_label: str) -> torch.Tensor:
        nonlocal max_violation
        with torch.no_grad():
            current = _project_zero_sum(weights, config.normalize_columns)
            if not bool(torch.isfinite(current).all()):
                raise NonFiniteLoss(f"the spanning set is not finite {step_label}")
            violation = float(torch.max(torch.abs(current.sum(dim=0))))
            max_violation = max(max_violation, violation)
            if violation > config.zero_sum_tolerance:
                raise ZeroSumViolation(
                    f"max |1ᵀ v_j| = {violation:.3e} exceeds {config.zero_sum_tolerance:.3e} "
                    f"{step_label}"
                )
        return current

    inspect("at initialisation")

    for step in range(config.step_count):
        optimizer.zero_grad(set_to_none=True)
        # The lr in force *for this step*, read off the param group the step is about to use.
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        spanning_set_t = _project_zero_sum(weights, config.normalize_columns)
        loss = training_loss(X_t, spanning_set_t, config.epsilon, penalties)
        if not bool(torch.isfinite(loss)):
            raise NonFiniteLoss(f"the loss is not finite at step {step}")
        loss.backward()
        if weights.grad is None or not bool(torch.isfinite(weights.grad).all()):
            raise NonFiniteLoss(f"the gradient is not finite at step {step}")
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        current = inspect(f"after step {step}")
        if recorder.enabled:
            recorder.record_step(
                step, losses[-1], np.ascontiguousarray(current.numpy(), dtype=np.float64)
            )

    with torch.no_grad():
        final = _project_zero_sum(weights, config.normalize_columns)
    spanning_set = np.ascontiguousarray(final.numpy(), dtype=np.float64)

    pairs = rounded_pairs(spanning_set)
    run = TwoHotSpanRun(
        spanning_set=spanning_set,
        loss_history=np.array(losses, dtype=np.float64),
        relaxed_objective=projector_residual(X, spanning_set),
        rounded_cut=rounded_cut(X, spanning_set),
        collision_measures=collision_measures(spanning_set),
        labels=clustering_from_pairs(node_total, pairs),
        max_zero_sum_violation=max_violation,
        config=config,
        learning_rate_history=np.array(learning_rates, dtype=np.float64),
    )
    if recorder.enabled:
        recorder.record_end(run)
    return run
