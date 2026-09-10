<!-- transcribed from the research repo's evidence-to-plan run 2026-09-09..10; §5 of that plan was the source -->
# Two-hot spanning sets — minimal rcut prototype

Status: **proposed** · Written 2026-09-09 · Owner: Xavier
Target: three stacked slices from `feat/two-hot-span-problem` into `main` (`main` @ `85f2205`).
Why the code exists: `~/develop/research/candidates/constrained-eigenvectors.md` and Prof.
Schweitzer's handoff brief §20 ("the exact next step").

---

## 0. Why

The handoff asks for "the smallest possible direct PyTorch implementation of this objective, not a
full autoencoder and not yet the alternating analytical solver". Two facts settled in the research
session make that implementable without guesswork: R(u) ≤ 1/2 for zero-sum u with equality exactly at
2-hot vectors (proved), and Ê of a rounded within-block spanning forest is exactly the RatioCut of the
partition, with the λ = 0 minimum of E over r = n − K vectors equal to the sum of the K smallest
eigenvalues of XXᵀ (proved, machine precision on five random weighted graphs). So the harness has a
floor to report against and a tiny-graph oracle that is a partition enumeration, not a search.

`main` @ `85f2205` has no torch: absent from `pyproject.toml` dependencies and groups, zero matches
in `uv.lock`, no import under `src/`. `networkx>=3.2` is already a core dependency, so the connected
components of the rounded-pair graph come free. The module triple is mirrored **by role** —
problem / optimizer / harness, as `nystrom_landmark_problem.py` (374 lines) /
`nystrom_landmark_selectors.py` (273) / `nystrom_uci_harness.py` (336) — not by base class:
this is a gradient-optimized continuous relaxation, so `AbstractGraphProblem` and
`SearchCostFunction` do not apply and are not subclassed.

## 1. Decisions and assumptions

| # | Decision |
|---|---|
| P-1 | torch enters as a PEP 735 `[dependency-groups] torch` group with an explicit CPU index (`docs/DECISIONS.md` **D-31**), not as an extra: extras are published metadata and this library publishes no torch consumer. |
| P-2 | The **training** loss uses the ridge projector `V(VᵀV + εI)⁻¹Vᵀ`; every **reported** E\* and Ê uses the exact `pinv` projector in numpy. The objective is never reported through a regularized projector. |
| P-3 | The rounded-cut oracle for n ≤ 10 is a brute-force set-partition enumeration, not the A\* engine. |
| P-4 | K = number of clusters, r = n − K = number of spanning vectors. `k` is not used in this module family. |

| # | Assumption | Why |
|---|---|---|
| A-1 | Adam on the unconstrained ratio penalty reaches a usable V on toy graphs. | No published result covers plain autodiff on an ℓ1/ℓ2 ratio penalty; the field uses ADMM, ℓ1−αL2 bisection, or moving-balls, all to critical points only. Slice 2.2's no-NaN test is the check; Hoyer's projection step is the named fallback. |
| A-2 | The CPU wheels run on the R620 (no AVX2). | PyTorch does not document an AVX2 requirement; issue #94021 shows some ops already fail without it. Recorded as **BL-42**; first runs are on the Mac, so nothing depends on it. |

## 2. Current state (evidence)

Captured 2026-09-09 at `85f2205`. No torch anywhere in the tree. `networkx>=3.2` core. Tests green.
No module in `math/` or `ml/` optimizes a continuous objective by gradient; this is the first.

## 3. Phases and slices

Each slice: branch off the previous slice's branch; `uv run ruff check . && uv run ruff format --check .`;
`uv run pytest`; reviewer-agent; Xavier commits and opens the PR. Each slice is ≤ ~300 changed lines
including tests, with its tests and records in the same PR. "Done when" is in addition to that.

**Acceptance, verbatim, on all three slices of phase 2:**

- "VᵀV = I is never imposed (source §3, §17)"
- "gradient track only; the alternating solver is parked (source §19)"
- "Ê is the headline number; R(v_j) is a per-column diagnostic (source §13)"
- "rcut first; ncut only in phase 4 (source §20)"

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 2.1 | `feat/two-hot-span-problem` | `math/graph/two_hot_span_problem.py`, numpy only: X = E for rcut from a networkx graph; `projector_residual` via `pinv`; `round_columns` → (argmax, argmin) pairs; `rounded_cut` = Ê; `spectral_floor` = Σλ; `brute_force_rcut` by partition enumeration; the four test graphs (roach G_5 with n = 20 and m = 23, karate, seeded planted partition, two-moons kNN). | Ê from a rounded within-block spanning forest equals RatioCut on ≥ 5 random weighted graphs (n = 7–11, K = 2–3) to 1e-10; `brute_force_rcut` matches an independent enumeration at n ≤ 8; roach G_5 has 4k vertices and 5k−2 edges for k = 5. |
| 2.2 | `feat/two-hot-span-optimizer` | `math/algorithms/two_hot_span_optimizer.py`: trainable W, V = W − c(cᵀW)/(cᵀc), optional column normalization, loss E_ridge(V) − λ Σ_j R(Cv_j), Adam. `pyproject.toml` gains the `torch` group + explicit CPU index; CI runs `uv sync --locked --dev --group torch`; torch-dependent tests use `pytest.importorskip`. **D-31** and **BL-42** in the same PR. | `cᵀv_j = 0` to 1e-12 at every step of a seeded run; a test asserts the reporting path never uses `epsilon`; a 20-step run on the roach graph is finite (no NaN) with finite gradients; on a seeded 200-step run on one n ≤ 8 graph, the reported Ê is ≥ the brute-force optimum from 2.1 and the report states both (S6). |
| 2.3 | `feat/two-hot-span-harness` | `ml/projects/two_hot_span_harness.py`: the four graphs × a λ sweep; the datum computed three ways; one JSON report per graph. | Each report carries E\*, Ê, Ê − E\*, R(v_j) per column, the clustering, Σλ, Ê − Σλ, and the datum's rounded rcut under `kmeans`, `discretize` and `cluster_qr`; at n ≤ 10 it also carries the brute-force optimum and asserts Ê ≥ it. |
| 4.1 | `feat/two-hot-span-ncut` | **Conditional** on 2.3's rcut behaving reliably (source §20). c_i = √d_i, C = diag(c), X = D^{-1/2}E threaded as parameters of the same three modules. | Ê = NCut on ≥ 5 random weighted graphs to 1e-10; the harness emits ncut rows beside rcut rows; the diff adds parameters, it does not copy modules. If this slice is not reached by 2026-09-25, it is **BL-41** and parked as P23 in the research repo. |

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Adam produces NaN at zero entries of ‖v‖₁. | Slice 2.2's finite-value test on the roach graph is the gate; the fallback is Hoyer's projected-gradient step, which never imposes orthogonality. |
| V goes rank-deficient and the projector's gradient breaks. | P-2: ridge in training, `pinv` in reporting. QR is excluded — reduced mode returns an n×r Q whatever the rank, so QQᵀ over-spans a rank-deficient V and is a different function. |
| The prototype's numbers get quoted before the identity test passes. | 2.1 lands the identity test before any optimizer exists; the harness asserts Ê ≥ the brute-force optimum at n ≤ 10 on every run. |
| torch bloats CI. | The group is opt-in; only the `--group torch` job installs it, and the CPU index keeps the Linux wheel off the GPU build. |

## 5. Immediate next actions

1. Xavier reviews § 1 and § 3; amends or accepts; plan status → `accepted`.
2. Slice 2.1 opens from `main`.
