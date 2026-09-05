# Nyström A* bound — rank-one downdate and spectrum truncation (BL-27)

Status: **draft for owner acceptance** · Written 2026-09-04 · Owner: Xavier · Implementer: a separate
session; this document is its brief. Review afterwards by the authoring session.
Builds on `docs/plans/2026-09-nystrom-landmark-selection.md` (the port, PRs #22) and on the research
specification `~/develop/research/nystrom/harness/BL-27-rank-one-downdate.md` (the math, hazards,
acceptance tests T1–T6). Where this plan and that spec disagree on *what*, the spec wins; on *how it
arrives in this repo*, this plan and `CONTRIBUTING.md` win.

---

## 0. Why

`NystromCssCostFunction.lower_bound` recomputes an SVD of the selected columns and an n×n
eigendecomposition of the deflated residual for **every child above goal depth**: O(n³) per child.
D-24 removed that cost for goal-depth children only. Measured on the 2026-09-04 grid (EXP-01 in the
research program): n = 80, k = 5 cells take a median 7 minutes of search over ~110,000 expansions;
n ≥ 500 is out of reach. Arai, Maung & Schweitzer (AAAI-15, §4) do one eigendecomposition per
**parent** and price every child as a rank-one downdate of the parent's spectrum through the secular
equation; the research reproduction (`repro/astar-css/astar_css.py`) implements it and matches the
paper's node counts. Porting that is BL-27.

A second change rides with it, decided 2026-09-04: the field-scale experiment (BL-28, EXP-09) needs
the root spectrum **truncated** for n ≥ 1,000, because for a full-rank kernel the per-parent
eigendecomposition is still n³. Truncation is admissible without any correction term, by monotonicity
of the Schur complement (spec §10.2); goal costs stay exact. It is a new backlog item and a new
decision, and it ships behind a parameter whose default leaves every existing test untouched.

Node counts and chosen subsets must not change. This is a change to the cost of the search, not to
the search — the same property slice 6.1 (batched siblings) was held to.

## 1. Decisions and assumptions

Decided 2026-09-04 (owner, in session):

| # | Decision |
|---|---|
| P-1 | Branch from `main` (**pull first**: local `main` is behind `origin/main` @ `e3e92d1`, the PR #22 merge). One PR per slice, stacked in order, squash-merged, as in the port. |
| P-2 | The reproduction's algorithm is ported, not its code: no bitmask closed sets, no counting conventions, no changes to `AStarSearch`. Parity is with the current `mllib` engine (node counts, subsets), never with the repro's counts. |
| P-3 | `lower_bound(state)` (the per-child path) is kept unchanged as the **oracle** every new bound is tested against. `goal_cost` is untouched. The D-24 goal-depth Schur batch is untouched and remains the k̄ = 1 case. |
| P-4 | Spectrum truncation is in scope (owner decision: EXP-09 needs it) as a constructor parameter `spectrum_mass_tolerance: float = 0.0` on `NystromLandmarkProblem`; δ = 0 reproduces today's behaviour exactly. Recorded as **D-26**. |
| P-5 | The secular solver is vectorised bisection (the repro's deviation from Gragg's method), 64 iterations, fixed. Gragg/Melman is not ported; revisit only if the solver, not the parent `eigh`, dominates a profile. |
| P-6 | The frozen reference set of grid cells for the engine-independence test is vendored from the research program when its snapshot S1 is frozen (`~/develop/research/nystrom/harness/BL-27-reference-cells.json`, ~60 cells with expected node counts and subsets). Until then the test runs on the bundled fixtures and seeded random kernels at n ≤ 12, and the vendored file is a follow-up test in the same slice family. |

Assumptions to confirm before slice A:

| # | Assumption | Why |
|---|---|---|
| A-1 | `NystromLandmarkProblem` may grow three attributes (`eigenvalues`, `eigenvectors`, `reduced_coordinates`) and one parameter without a contract change: nothing outside the Nyström modules constructs it. | ARCHITECTURE §2 contracts are the graph/cost ABCs; the problem class is a leaf. |
| A-2 | The secular solver belongs in `mllib/math/linear_algebra_helpers.py` beside `QuadraticFormHelper`, as a module-level function with a full docstring; it is a numerical routine, not a Nyström object. | Module already exists for exactly this kind of helper; BL-14 smoke test covers it. |
| A-3 | The `# TODO(BL-27)` marker on `_remaining_spectrum` is removed in slice C and the method is kept (it is the seam the clamp tests inject through, per the port's review). | CI rejects a TODO whose item is closed only if the id is missing; a stale marker on a closed item is a review finding. |
| A-4 | The learning-log entry for the downdate lands with slice C; the entry for truncation (with the four-line admissibility proof) with slice D; D-26 with slice D; BL-29 (truncation) is *added* in slice A and *closed* in slice D. | CONTRIBUTING: records in the same PR; "new work starts from an entry". |
| A-5 | `examples/nystrom_batched_bounds.py` gains a third column (downdated) rather than a new example file. | One benchmark script per topic; it already compares two bound paths. |

## 2. Current state (evidence)

`origin/main` @ `e3e92d1` is green: 183 tests, ruff clean, baseline untouched. Relevant files:

| File | What it holds now | Lines |
|---|---|---|
| `src/mllib/math/graph/nystrom_landmark_problem.py` | `NystromLandmarkProblem` (validation, `kernel_sqrt` via `eigh`, canonical successors); `NystromCssCostFunction` (`lower_bound` per child, `goal_cost`, D-24 `lower_bounds` for goal depth, `_remaining_spectrum` with `TODO(BL-27)`, `_residual_kernel`, `_project_out`, `_orthonormal_basis`) | 205 |
| `src/mllib/math/search_cost_function.py` | the contract; `lower_bounds` default = per-child | 45 |
| `src/mllib/math/algorithms/a_star_search.py` | calls `lower_bound` once at the root, `lower_bounds` once per expansion in generation order; FIFO ties | 98 |
| `src/mllib/math/linear_algebra_helpers.py` | `QuadraticFormHelper` only | 12 |
| `tests/math/graph/test_nystrom_landmark_problem.py` | 27 tests incl. clamp injection through `_remaining_spectrum`, batched-vs-per-child equivalence, "A* expands the same states whether batched or not" | 422 |
| `tests/math/fixtures/` | `identity_8x8`, `all_ones_8x8`, `rbf_chain_{4,6,8,10}x{…}` — repeated-eigenvalue and rank-one cases already present | — |
| `examples/nystrom_batched_bounds.py` | per-child vs batched timing on SPECTF; harness both ways | — |
| `docs/BACKLOG.md` BL-27 | the intent and the marker's location | — |

The reproduction's reference implementation: `~/develop/research/repro/astar-css/astar_css.py`
(`CSSProblem` for root decomposition + reduced coordinates; `top_eigs_rank1_downdate` for the secular
solver; the parent/children block inside `astar_css`). Read-only; nothing is imported from it.

## 3. Functional requirements

| FR | Requirement | Slice |
|---|---|---|
| FR-1 | `NystromLandmarkProblem` exposes `eigenvalues` (descending, clipped ≥ 0), `eigenvectors` (matching columns), `numeric_rank` r, and `reduced_coordinates` = D_r^{1/2} V_rᵀ of shape (r, n), with `reduced_coordinatesᵀ · reduced_coordinates ≈ K` when nothing is truncated and column norms² equal `diag(K)`. `kernel_sqrt` is unchanged. | A |
| FR-2 | `top_eigenvalues_of_rank_one_downdate(eigenvalues, W, count) → (count, m)`: for each column w of W, the `count` largest eigenvalues of `diag(eigenvalues) − w wᵀ`, by bisection on the secular function within interlacing intervals; collapsed intervals (repeated eigenvalues) return the shared value; the last interval's lower end is `λ_r − ‖w‖²`; `count = 0` returns shape (0, m). | B |
| FR-3 | `NystromCssCostFunction.lower_bounds(parent, successors)` prices **every** depth: goal-depth children by the existing Schur batch; non-goal children by one `eigh` of H = D − Z Zᵀ at the parent and FR-2 for the children, with g = tr(H) − ‖z_j‖², h = Σ top-(k̄−1) downdated eigenvalues, f = max(g − h, 0); columns already in the parent's span (‖residual‖ ≤ tolerance) get z = 0. Falls back to the per-child oracle when the successors do not extend the parent, as today. | C |
| FR-4 | Every value FR-3 returns equals the oracle `lower_bound` for that child to within 1e-8 · tr(K) (D-24's "up to rounding") at δ = 0, on every fixture and on seeded random kernels, at every depth, including rank-deficient and badly scaled kernels. | C |
| FR-5 | `AStarSearch` expands the same number of states and returns the same subset with FR-3 as with an oracle-only cost function, on all fixtures and seeded random kernels (n ≤ 12, k ≤ 4); on the vendored reference cells when available (P-6). Where the brute-force optimum set has more than one member, only membership is asserted. | C |
| FR-6 | `spectrum_mass_tolerance` δ on the problem: r is the smallest rank with Σ_{i>r} λ_i ≤ δ · tr(K) (and ≥ 1); δ = 0 falls back to the relative eigenvalue tolerance used today; `reduced_coordinates` and the non-goal bound use the truncated spectrum; `goal_cost`, the goal-depth Schur batch and `kernel_sqrt` remain exact on the full K. δ appears in the problem's `describe()` output via the constructor signature. | D |
| FR-7 | For δ > 0 the bound stays admissible: on brute-forceable fixtures and seeded kernels, for δ ∈ {1e-10, 1e-8, 1e-6, 1e-4}, every state's bound ≤ the true cost of every completion, and A* returns a member of the brute-force optimum set. r = n at δ = 0 on a full-rank kernel; r = 1 on the all-ones fixture for any δ > 0. | D |
| FR-8 | Records: BL-29 added (A) and closed (D); BL-27 closed with the commit (C); `TODO(BL-27)` marker removed (C); D-26 (D); learning-log entries "the parent-once, child-cheap structure, completed" (C) and "truncation is admissible by Schur-complement monotonicity" with the proof (D); D-24's *Consequences* paragraph is not edited (append-only) — D-26 cross-references it. | A, C, D |
| FR-9 | `examples/nystrom_batched_bounds.py` reports per-child, batched (goal-depth only) and downdated timings on the same rows, and asserts (prints) `same` for node counts across all three. | C |
| FR-10 | Gates on every slice: `uv run ruff check . && uv run ruff format --check .`, `uv run pytest` (183 + new), the training baseline untouched, no new per-file ignores (BL-24 counts do not rise), reviewer-agent run before the PR. | all |
| FR-11 | Not built, recorded: Gragg's method (P-5); parent-from-grandparent downdate (spec §7); the grid runner, Laplacian kernel, ridge-leverage selector and tie-aware oracle (BL-28 and the research program's H2–H5) — a follow-on plan after this one is reviewed. | — |

## 4. Phases and slices

Each slice: branch off the previous slice's branch; gates per FR-10; reviewer-agent; Xavier commits
and opens the PR. Test names describe the behaviour (`test_<what_is_true>`), one behaviour per test,
deterministic (seeded), per `.claude/agents/testing-agent.md`.

| Slice | Branch | Change | Tests (new) | Done when |
|---|---|---|---|---|
| 0 | `docs/nystrom-downdate-plan` | This plan marked `accepted` with A-1…A-5 confirmed or amended; BL-29 entry text drafted (added in A). | — | Plan merged. |
| A | `feat/nystrom-root-spectrum` | `NystromLandmarkProblem` keeps `eigenvalues`, `eigenvectors`, `numeric_rank`, `reduced_coordinates` (FR-1); BL-29 entry added to `BACKLOG.md` (intent: truncation, `backlog-only`, re-entry "slice D of this plan"). | reduced coordinates reproduce K; column norms equal the diagonal; identity has rank n and all-ones rank 1; a rank-deficient kernel drops its zero modes; existing 183 unchanged | FR-1; ~40 src / ~50 test. |
| B | `feat/rank-one-downdate-solver` | `top_eigenvalues_of_rank_one_downdate` in `linear_algebra_helpers.py` (FR-2), docstring stating interlacing, the secular function, the collapsed-interval rule, and the Gragg deviation. | matches `eigvalsh(diag(λ) − wwᵀ)` to rtol 1e-9 on seeded random spectra and on the identity/all-ones/rbf-chain fixtures' spectra; `w = 0` leaves the spectrum; `count = 0` and `count = r` shapes; a hypothesis property over random λ (descending, ≥ 0) and w | FR-2; ~60 src / ~90 test. |
| C | `feat/nystrom-downdate-bound` | `lower_bounds` for every depth (FR-3) using A and B; marker removed (A-3); `_remaining_spectrum` kept; BL-27 closed; learning-log entry; example gains the third column (FR-9). | equivalence with the oracle at every depth on every fixture and seeded kernels, incl. rank-deficient and badly scaled (FR-4); A* node counts and subsets unchanged vs an oracle-only subclass (FR-5), reusing the pattern of `test_a_star_expands_the_same_states_whether_bounds_are_batched_or_not`; the palindrome chain's tie handled via the optimum *set*; the existing clamp-injection tests still pass; columns already in the span priced as the parent | FR-3, 4, 5, 8 (C part), 9; ~120 src / ~150 test. |
| D | `feat/nystrom-spectrum-truncation` | `spectrum_mass_tolerance` on the problem (FR-6); bound uses the truncated spectrum for non-goal children; D-26 in `DECISIONS.md`; BL-29 closed; learning-log entry with the proof. | admissibility at each δ on brute-forceable cases (FR-7); optimum membership at each δ; rank rule (r = n at δ = 0 full rank; r = 1 all-ones); δ = 0 byte-identical to slice C on node counts; `describe()` shows the parameter | FR-6, 7, 8 (D part); ~50 src / ~90 test. |
| E | `test/nystrom-reference-cells` | When S1 is frozen (P-6): vendor `BL-27-reference-cells.json` under `tests/math/fixtures/`, one test asserting node counts and optimum membership for every cell at δ = 0 with the downdated engine. Runtime budget ≤ 10 s — pick the cell mix accordingly or mark `slow`. | the vendored cells | FR-5 at grid scale. |
| F | `docs/nystrom-downdate-close` | Plan status `complete` with a what-changed section (precedent: port plan §7, §8); timing table from the example on the reference machine, labelled as this machine's numbers; BL-24 counts unchanged. | — | Suite green; research program's `harness/CHANGES.md` H1/H1b marked done (Xavier's edit). |

Six or seven PRs. Estimated 1–2 days of implementation for A–D (matches the BL-27 entry), plus E when
S1 exists.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The downdated bound differs from the oracle in the last bits and A* breaks a tie the other way, changing a subset on a fixture. | FR-5 asserts membership in the brute-force optimum *set* where ties exist (the ten-point RBF chain is a palindrome); node counts must still match exactly. If a node count differs, the bound is wrong, not the tie — stop and diff per depth with FR-4's test. |
| Repeated eigenvalues (identity, all-ones fixtures) collapse interlacing intervals; a naive bisection returns garbage or NaN. | FR-2's collapsed-interval rule and its fixture tests in slice B, before the bound touches it. |
| `eigh(H)` at the parent returns slightly negative eigenvalues from rounding; the secular function's denominators cross zero. | Clip parent eigenvalues at 0 before the solve; the bound is clamped at 0 anyway (existing rule). Test on a badly scaled kernel (existing fixture pattern). |
| Platform rounding (Accelerate vs OpenBLAS) makes a bit-level assertion pass locally and fail in CI. | Compare clamped values with tolerances (FR-4); never assert the sign of a cancellation (the port's review finding 3). |
| The parent `eigh` is still n³ and the speedup at n = 80 disappoints. | The claim is "fewer decompositions", not a factor; FR-9 prints all three paths. Expected ≥ 10× on the n = 60, k = 5 row per the spec's estimate; if below 3×, profile before tuning. |
| Truncation changes node counts at δ > 0 and someone reads that as a bug. | FR-6/7 assert optimality and admissibility only; node counts at δ > 0 are reported by the research calibration (EXP-09a), not asserted here. δ = 0 is asserted byte-identical to slice C. |
| Scope creep into BL-28 (grid runner, Laplacian kernel, leverage selector). | FR-11: out of scope; a follow-on plan after this one is reviewed. |

## 6. Immediate next actions

1. Xavier reviews §1 (A-1…A-5) and §4; amends or accepts; slice 0 PR.
2. The implementing session starts at §7 below.

---

## 7. Handoff brief for the implementing session

**Start.**
```
cd ~/Desktop/BotMaker/botMaker
git checkout main && git pull            # local main is behind origin/main (e3e92d1)
uv sync && uv run pytest -q              # expect 183 passed
git checkout -b docs/nystrom-downdate-plan   # slice 0, then feat/nystrom-root-spectrum, …
```
Read, in order: `CLAUDE.md`; `docs/DECISIONS.md` D-22 to D-25; `docs/BACKLOG.md` BL-27, BL-28;
`docs/ARCHITECTURE.md` §2 and §4; `.claude/agents/testing-agent.md`; this plan; the research spec
`~/develop/research/nystrom/harness/BL-27-rank-one-downdate.md` §2, §5, §6, §10; the reproduction's
`top_eigs_rank1_downdate` and the parent/children block in `~/develop/research/repro/astar-css/astar_css.py`.

**The math to implement, in the repo's names** (Y = K^{1/2}, so Y_r = D^{1/2} Vᵀ):

```
root      : K = V D Vᵀ (eigh; clip D ≥ 0); r by tolerance; Y_r = sqrt(D_r)[:, None] * V_r.T     (r × n)
parent S  : Q = orthonormal basis of Y_r[:, S]      (SVD, RANK_TOLERANCE, as _orthonormal_basis does)
            Z = sqrt(D_r)[:, None] * Q ;  H = diag(D_r) − Z Zᵀ ;  λ, U = eigh(H) (descending)
            trace_parent = Σ λ                      (== trace of _residual_kernel(S), test it)
children  : R = Y_r − Q (Qᵀ Y_r);  norms = ‖R[:, j]‖ ;  tiny = norms ≤ 1e-12 · sqrt(tr K)
            q_j = R[:, j]/norms_j (0 where tiny) ; z_j = sqrt(D_r) * q_j ;  W = Uᵀ Z_children
            g_j = trace_parent − ‖z_j‖²
            h_j = Σ top (k̄−1) eigenvalues of diag(λ) − w_j w_jᵀ        (slice B, vectorised over j)
            f_j = max(g_j − h_j, 0)
goal depth: unchanged D-24 Schur batch on the full K
```

**Do not:** change `AStarSearch`, `SearchCostFunction`, `goal_cost`, tie-breaking, or the D-24
goal-depth path; import anything from the research repo; add per-file ruff ignores; assert the sign of
a cancellation; quote a timing as a property of the method; commit (Xavier commits).

**Do:** keep `lower_bound` as the oracle and test the new path against it at every depth; use the
existing fixtures (identity and all-ones are the repeated-eigenvalue and rank-one cases); reuse the
"same states whether batched or not" test pattern for FR-5 with an oracle-only subclass
(`lower_bounds = SearchCostFunction.lower_bounds`, as `examples/nystrom_batched_bounds.py` does);
write the learning-log entries in the log's voice (what / where / design / what was confusing /
reference); run the reviewer-agent before each PR; leave the plan's §4 "Done when" verifiable.

**Numbers the review will check.** Node counts for the harness's default run must be identical to
the committed output (SPECTF n = 40, k = 3: 363 / 501 / 668 / 780 / 781 across the five scales; the
port's plan §8 table lists 668 / 1,650 / 2,837 / 5,645 / 15,740 for the example's rows). The
`examples/nystrom_batched_bounds.py` "same" column must read True for all three paths.

**Reporting back.** For the review session: the PR list with per-slice test counts, the example's
timing table, any place where the spec and the code had to diverge and why, and any number that
moved.
