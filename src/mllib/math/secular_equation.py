"""Eigenvalues of a rank-one downdate of a diagonal matrix, through the secular equation.

Subtracting one outer product from a matrix whose spectrum is known does not need a fresh
decomposition. If ``D = diag(λ)`` with ``λ`` descending, the eigenvalues ``μ`` of ``D - w wᵀ`` are
the roots of the secular equation

    s(μ) = 1 - Σ_j w_j² / (λ_j - μ) = 0,

and they interlace the old ones: the i-th new eigenvalue lies in ``[λ_{i+1}, λ_i]``, the last in
``[λ_r - ‖w‖², λ_r]``. On each interval ``s`` is strictly decreasing (its derivative is
``-Σ w_j² / (λ_j - μ)²``), from ``+∞`` at the lower end to ``-∞`` at the upper end whenever both
bounding weights are nonzero, so bisection cannot miss the root; a zero bounding weight makes
that endpoint the root, and the bisection converges to it. When an interval has zero width
(a repeated eigenvalue) the root is that eigenvalue and nothing is solved.

This is the step that prices a search's children from its parent's spectrum (BL-27): the parent is
decomposed once, and every child is one downdate, vectorised here across all children at once.
Gragg's cubically convergent iteration would converge in a handful of steps instead of 64
bisections; it is not implemented because the parent decomposition, not this solve, dominates the
cost (plan P-5). [A1 §4, A3 eq. 10, L4]
"""

from __future__ import annotations

import numpy as np

# Bisection halves the bracket 64 times: ~1e-19 of the initial width, past double precision, so the
# result is accurate to rounding regardless of the scale of the spectrum.
BISECTION_ITERATIONS = 64
# An interlacing interval narrower than this multiple of the largest eigenvalue is a repeated
# eigenvalue in floating point, and its root is the eigenvalue itself.
COLLAPSED_INTERVAL_TOLERANCE = 1e-14


def top_eigenvalues_of_rank_one_downdate(
    eigenvalues: np.ndarray, weights: np.ndarray, count: int
) -> np.ndarray:
    """Return the ``count`` largest eigenvalues of ``diag(eigenvalues) - w wᵀ`` for each column of
    ``weights``.

    ``eigenvalues`` is the parent spectrum, shape ``(r,)``, descending and non-negative.
    ``weights`` has shape ``(r, m)``: one column per downdate, each in the parent's eigenbasis.
    The result has shape ``(count, m)``, row ``i`` holding the ``i``-th largest eigenvalue of every
    downdate, descending down the rows. ``count = 0`` returns an empty ``(0, m)`` array.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a 1-D array.")
    if weights.ndim != 2 or weights.shape[0] != eigenvalues.shape[0]:
        raise ValueError("weights must have shape (r, m) with r the number of eigenvalues.")
    if np.any(np.diff(eigenvalues) > 0.0):
        raise ValueError("eigenvalues must be in descending order.")
    if eigenvalues.size and eigenvalues[-1] < 0.0:
        raise ValueError("eigenvalues must be non-negative.")
    rank, downdate_count = weights.shape
    if count < 0 or count > rank:
        raise ValueError("count must lie between 0 and the number of eigenvalues.")
    if count == 0:
        return np.zeros((0, downdate_count))

    squared_weights = weights * weights
    eigenvalue_column = eigenvalues[:, None]
    largest = float(eigenvalues[0]) if rank else 0.0
    collapsed_width = COLLAPSED_INTERVAL_TOLERANCE * max(largest, np.finfo(float).tiny)

    result = np.empty((count, downdate_count))
    for index in range(count):
        upper = np.full(downdate_count, eigenvalues[index])
        if index + 1 < rank:
            lower = np.full(downdate_count, eigenvalues[index + 1])
        else:
            lower = eigenvalues[index] - squared_weights.sum(axis=0)
        collapsed = (upper - lower) <= collapsed_width

        for _ in range(BISECTION_ITERATIONS):
            midpoint = 0.5 * (lower + upper)
            # A denominator can vanish only where the interval has collapsed, and those columns
            # take the eigenvalue itself below; an overflowing term just signs s correctly.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                secular = 1.0 - np.sum(
                    squared_weights / (eigenvalue_column - midpoint[None, :]), axis=0
                )
            # s decreases through the root, so a positive value means the root is above the midpoint.
            root_is_above = secular > 0.0
            lower = np.where(root_is_above, midpoint, lower)
            upper = np.where(root_is_above, upper, midpoint)

        result[index] = np.where(collapsed, eigenvalues[index], 0.5 * (lower + upper))
    return result
