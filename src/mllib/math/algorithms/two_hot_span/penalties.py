"""The two-hot span penalties: the terms the training loss adds to its cost, as injected objects.

Each class is one term of the brief's training loss, with its weight as a knob on the class
(D-35 (3)) and its sign fixed by what the term is for: a *reward* — the collision measure, the graph
term — comes back negative from ``compute_penalty`` so the loss is always a plain sum; a *penalty* —
the diversity term — comes back positive. ``term`` is the unweighted quantity a reader can check by
hand; ``compute_penalty`` is what the loss adds. All arithmetic is float64 torch, kept here and out
of the abstract base (D-35 (6)); every term is a training knob in D-31's sense and never enters a
reported number.

Off by default means absent. A term is injected when it is wanted; a weight of zero is not a way of
switching one off, because a zero-weighted term still builds its tensors and, on a column of zeros,
its NaNs. A composition root leaves it out instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from mllib.math.regularization_function import AbstractRegularizationFunction


def _as_matrix_tensor(matrix: np.ndarray | torch.Tensor) -> torch.Tensor:
    """A float64 tensor of a square matrix, whichever array library it arrived in."""
    return torch.as_tensor(np.asarray(matrix, dtype=np.float64))


def _same_graph_penalty(penalty: object, other: object, matrix_name: str) -> bool:
    """Equal when the same class, the same weight and the same matrix entry for entry."""
    if type(other) is not type(penalty):
        return NotImplemented  # type: ignore[return-value]
    mine, theirs = getattr(penalty, matrix_name), getattr(other, matrix_name)
    return penalty.weight == other.weight and torch.equal(mine, theirs)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class CollisionPenalty(AbstractRegularizationFunction):
    """-λ Σ_j R(v_j): the collision measure of every column, rewarded.

    R(v) = ‖v‖₂² / ‖v‖₁² is scale-invariant, equal to ½ exactly on a 2-hot column and smaller on
    any column that spreads over more coordinates; rewarding it pulls each column toward a vertex
    pair. It is a diagnostic of how 2-hot a column is, never the criterion a run is judged by.
    """

    weight: float = 0.0

    def term(self, spanning_set: torch.Tensor) -> torch.Tensor:
        """Σ_j R(v_j), unweighted."""
        squared = torch.sum(spanning_set * spanning_set, dim=0)
        absolute = torch.sum(torch.abs(spanning_set), dim=0)
        return torch.sum(squared / (absolute * absolute))

    def compute_penalty(self, parameters: torch.Tensor) -> torch.Tensor:
        return -(self.weight * self.term(parameters))


@dataclass(frozen=True, slots=True)
class LaplacianAdjacencyPenalty(AbstractRegularizationFunction):
    """-μ Σ_j v_jᵀ L v_j / ‖v_j‖₂²: the Laplacian graph term, rewarded.

    On a unit 2-hot column over the pair (i, j) this is (d_i + d_j + 2 w_ij)/2: larger on an edge
    than off it, and carrying a degree bias that has nothing to do with whether (i, j) is an edge.
    The division by ‖v_j‖₂² makes it scale-invariant, as R(v) is. The Laplacian is the graph's
    own, recovered from the incidence matrix by `two_hot_span_problem.graph_matrices`.
    """

    laplacian: torch.Tensor = field(compare=False, repr=False)
    weight: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "laplacian", _as_matrix_tensor(self.laplacian))

    # Value semantics on the graph too: the generated tuple comparison cannot compare tensors.
    def __eq__(self, other: object) -> bool:
        return _same_graph_penalty(self, other, "laplacian")

    def __hash__(self) -> int:
        return hash((type(self), self.weight, tuple(self.laplacian.shape)))

    def term(self, spanning_set: torch.Tensor) -> torch.Tensor:
        """Σ_j v_jᵀ L v_j / ‖v_j‖₂², unweighted."""
        squared = torch.sum(spanning_set * spanning_set, dim=0)
        numerator = torch.sum(spanning_set * (self.laplacian @ spanning_set), dim=0)
        return torch.sum(numerator / squared)

    def compute_penalty(self, parameters: torch.Tensor) -> torch.Tensor:
        return -(self.weight * self.term(parameters))


@dataclass(frozen=True, slots=True)
class EdgeProductAdjacencyPenalty(AbstractRegularizationFunction):
    """-μ Σ_j |v_j|ᵀ A |v_j| / ‖v_j‖₂²: the edge-product graph term, rewarded.

    On a unit 2-hot column over the pair (i, j) this is w_ij on an edge and exactly 0 off it: the
    same preference for edges as the Laplacian form with none of its degree bias. Scale-invariant
    by the same division. The adjacency is the graph's own, recovered by `two_hot_span_problem.graph_matrices`.
    """

    adjacency: torch.Tensor = field(compare=False, repr=False)
    weight: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "adjacency", _as_matrix_tensor(self.adjacency))

    # Value semantics on the graph too: the generated tuple comparison cannot compare tensors.
    def __eq__(self, other: object) -> bool:
        return _same_graph_penalty(self, other, "adjacency")

    def __hash__(self) -> int:
        return hash((type(self), self.weight, tuple(self.adjacency.shape)))

    def term(self, spanning_set: torch.Tensor) -> torch.Tensor:
        """Σ_j |v_j|ᵀ A |v_j| / ‖v_j‖₂², unweighted."""
        squared = torch.sum(spanning_set * spanning_set, dim=0)
        absolute = torch.abs(spanning_set)
        numerator = torch.sum(absolute * (self.adjacency @ absolute), dim=0)
        return torch.sum(numerator / squared)

    def compute_penalty(self, parameters: torch.Tensor) -> torch.Tensor:
        return -(self.weight * self.term(parameters))


@dataclass(frozen=True, slots=True)
class DiversityPenalty(AbstractRegularizationFunction):
    """+ν D(V), D(V) = Σ_i ℓ_i²: the squared vertex load, penalised.

    Each column gets the brief's §9 distribution p_j = |v_j| / ‖v_j‖₁, and the **vertex load** is
    ℓ_i = Σ_j p_ij — how much of the r units of mass the spanning set puts on vertex i. Σ_i ℓ_i = r
    always, so by Cauchy-Schwarz D(V) ≥ r²/n with equality exactly when every vertex carries the
    same load r/n. Minimising it spreads the columns over the vertices and penalises a spanning set
    that piles several columns onto the same vertices, or onto the same pair.

    For exact 2-hot columns it is a statement about the pair graph and nothing else: a 2-hot column
    on (i, j) has p = (½, ½), so ℓ_i = deg_i/2 with deg_i the number of columns incident to i, and
    D(V) = ¼ Σ_i deg_i². The identity worth carrying: Σ_{j<k} p_jᵀ p_k = ½(D(V) - Σ_j R(v_j)),
    since ‖p_j‖₂² = R(v_j). So D is the **pairwise column overlap** plus the collision sum — the
    one term that is not a sum of per-column scores, and therefore the one that couples the
    columns. The constant r²/n is deliberately not subtracted: it is constant in V, so it changes
    no gradient and no comparison between two runs on the same graph, and carrying it would only
    invite reading the number as a deficit rather than as the loss term it is.
    """

    weight: float = 0.0

    def term(self, spanning_set: torch.Tensor) -> torch.Tensor:
        """D(V) = Σ_i ℓ_i², unweighted."""
        absolute = torch.abs(spanning_set)
        distribution = absolute / torch.sum(absolute, dim=0)
        load = torch.sum(distribution, dim=1)
        return torch.sum(load * load)

    def compute_penalty(self, parameters: torch.Tensor) -> torch.Tensor:
        return self.weight * self.term(parameters)
