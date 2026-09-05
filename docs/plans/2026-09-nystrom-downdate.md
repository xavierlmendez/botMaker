# Nyström A* bound — rank-one downdate and spectrum truncation (BL-27, BL-29)

Status: **accepted** · Drafted 2026-09-04 · Grilled and rewritten 2026-09-04 (§8) · Owner: Xavier ·
Implementer: a separate session; §7 is its brief. Review afterwards by the authoring session.
Builds on `docs/plans/2026-09-nystrom-landmark-selection.md` (the port, PR #22) and on the research
specification `~/develop/research/nystrom/harness/BL-27-rank-one-downdate.md` (the math, hazards,
acceptance tests T1–T6). Where this plan and that spec disagree on *what*, the spec wins; on *how it
arrives in this repo*, this plan, `CONTRIBUTING.md` and `docs/DECISIONS.md` win. Vocabulary is
`CONTEXT.md`; citations are `docs/REFERENCES.md`.

---

## 0. Why

`NystromCssCostFunction.lower_bound` recomputes an SVD of the selected columns and an n×n
eigendecomposition of the deflated residual for **every child above goal depth**: O(n³) per child.
D-24 removed that cost for goal-depth children only. Measured on the 2026-09-04 grid (EXP-01 in the
research program): n = 80, k = 5 cells take a median 7 minutes of search over ~110,000 expansions;
n ≥ 500 is out of reach. Arai, Maung & Schweitzer [A1, §4] do one eigendecomposition per **parent**
and price every child as a rank-one downdate of the parent spectrum through the secular equation;
the research reproduction (`repro/astar-css/astar_css.py`) implements it and matches the paper's
node counts. Porting that is BL-27.

A second change rides with it, decided 2026-09-04: the field-scale experiment (BL-28, EXP-09) needs
the root spectrum **truncated** for n ≥ 1,000, because for a full-rank kernel the per-parent
eigendecomposition is still n³. Truncation is admissible without any correction term, by
monotonicity of the Schur complement (spec §10.2); goal costs stay exact. It is BL-29 and D-26, and
it ships behind a parameter whose default leaves every existing test untouched.

Expansion counts and chosen subsets must not change at δ = 0. This is a change to the cost of the
search, not to the search, the property slice 6.1 (batched siblings) was held to, and this time it
is pinned by a committed baseline before any code moves (P-6).

## 1. Decisions and assumptions

Decided by the owner in the grilling session of 2026-09-04:

| # | Decision |
|---|---|
| P-1 | This plan arrives by its own PR from `docs/nystrom-downdate-plan` off `main` @ `e3e92d1`; the skills tree it was first committed with stays on `feat/add-skills`. One PR per slice, stacked in order, squash-merged, as in the port. |
| P-2 | The reproduction's algorithm is ported, not its code: no bitmask closed sets, no counting conventions, no changes to `AStarSearch`. Parity is with the current `mllib` engine (expansion counts, subsets), never with the repro's counts. |
| P-3 | `lower_bound(state)` (the per-child path) is kept as the **oracle** every new bound is tested against; its docstring and the `_remaining_spectrum` comment say so, and the `TODO(BL-27)` marker goes. `goal_cost` is untouched. The D-24 goal-depth Schur batch is untouched and remains the k̄ = 1 case. |
| P-4 | The kept root decomposition is not a slice of its own (it would be a layer nothing consumes, which D-25 and the no-unstarted-code rule forbid). It ships inside the bound's slice. Order: solver (B), downdated bound (A+C), truncation (D), reference cells (E, conditional), close (F). |
| P-5 | The secular solver is its own module, `mllib/math/secular_equation.py`, exposing `top_eigenvalues_of_rank_one_downdate(eigenvalues, W, count)`: vectorised bisection, 64 iterations, fixed (the repro's deviation from Gragg's method). Gragg/Melman is not ported; revisit only if the solver, not the parent `eigh`, dominates a profile. ARCHITECTURE §1 gains the module. |
| P-6 | A **search baseline** is committed before any code slice (CONTRIBUTING § Behavioural baseline): `tests/ml/test_nystrom_search_baseline.py` with `nystrom_search_snapshot.json` beside it, pinning expansion count, chosen subset and residual trace for the six fixtures at their known-answer k, SPECTF n = 40, k = 3 at bandwidth scales 0.25 / 1 / 4, and the example's larger rows (60,3), (80,3), (40,4), (60,4). Regenerated only with `BASELINE_UPDATE=1` in a PR that says why. |
| P-7 | `examples/nystrom_batched_bounds.py` gains a `--no-timings` flag and a committed `nystrom_batched_bounds.example_output.txt` generated on `main` with it, plus a block that runs SPECTF n = 40, k = 3 at scales 0.25 / 0.5 / 1 / 2 / 4 printing expansion count and subset per scale. Every code slice's PR shows the file byte-identical (timings off). The draft's "363 / 501 / 668 / 780 / 781" figures are superseded by whatever that run records. |
| P-8 | Spectrum truncation is in scope as `NystromLandmarkProblem(kernel, landmark_count, spectrum_mass_tolerance=0.0)`; δ = 0 reproduces today's behaviour exactly. When δ > 0, **`kernel_sqrt` is built from the truncated kernel too**, so the oracle and the fast path compute the same truncated bound and D-24's "same value both paths" holds at every δ. `kernel_matrix`, `goal_cost` and the goal-depth Schur batch stay exact. Recorded as **D-26**. |
| P-9 | Rank rule: the **numeric rank** is the count of eigenvalues of K above `EIGENVALUE_TOLERANCE = 1e-12` times the largest; the **retained rank** (`retained_rank` attribute) is the smaller of the numeric rank and the smallest rank whose dropped mass is within δ·tr K, never below 1. The repro's rule (1e-10 on singular values of K^{1/2}) is not adopted. |
| P-10 | The new path treats a column as **explained** when its residual norm is at or below `1e-12 · √tr K` (spec §5.3). The D-24 goal-depth path keeps its absolute `PIVOT_TOLERANCE` on the residual diagonal, so the baseline stays byte-identical. Unifying the two rules is a backlog line under BL-29's close, not a TODO. |
| P-11 | The frozen reference set (spec §9) does not exist yet. Engine independence is asserted in-suite on fixtures and seeded kernels (FR-5). Slice E vendors the reference cells only if the chosen mix runs in ≤ 3 s with the downdated engine; otherwise the file stays research-side and EXP-09a runs it against botMaker. No `slow` marker is introduced. |
| P-12 | Vocabulary: a single root `CONTEXT.md` (glossary only), seeded with the search, Nyström, spectral and reporting terms and the rulings of §8; `docs/REFERENCES.md` holds the papers code or terms rest on, with keys matching the research reading list. `CLAUDE.md`'s reading list gains `CONTEXT.md`. |

Assumptions confirmed in the same session:

| # | Assumption | Why |
|---|---|---|
| A-1 | `NystromLandmarkProblem` may grow `eigenvalues` (descending, clipped ≥ 0), `eigenvectors`, `retained_rank`, `reduced_coordinates` and one constructor parameter without a contract change: nothing outside the Nyström modules, the harness, the example and the tests constructs it. | ARCHITECTURE §2 contracts are the graph/cost ABCs; the problem class is a leaf. `describe()` reads the constructor signature, so δ appears in descriptors automatically. |
| A-2 | The admissibility test at δ > 0 allows the same rounding slack as the equivalence test: `1e-8 · tr(K)`. | D-24's "up to rounding"; the port's review finding that a sign of a cancellation is never asserted. |
| A-3 | D-26 covers truncation only. The downdate is an implementation under D-24's extension point and gets a learning-log entry, not a decision. D-24 is not edited (DECISIONS.md is append-only; the spec's "update D-24" is superseded). | DECISIONS.md header: "Supersede rather than edit." |
| A-4 | Records land with their slice: BL-29 opened here (0a) and closed in D; BL-27 closed in A+C; learning-log entries for the downdate (A+C) and for the admissibility proof (D). | CONTRIBUTING: records in the same PR; "new work starts from an entry". |
| A-5 | Hypothesis properties are acceptable where an invariant exists (existing precedent: `max_examples=30, deadline=None`); no timing is asserted anywhere in the suite. | `.claude/agents/testing-agent.md`. |

## 2. Current state (evidence)

`main` @ `e3e92d1` (PR #22 merged) is green: 183 tests, ruff clean, baseline untouched. Relevant files:

| File | What it holds now | Lines |
|---|---|---|
| `src/mllib/math/graph/nystrom_landmark_problem.py` | `NystromLandmarkProblem` (validation, `kernel_sqrt` via `eigh` with the factors discarded, canonical successors); `NystromCssCostFunction` (`lower_bound` per child, `goal_cost`, D-24 `lower_bounds` for goal depth, `_remaining_spectrum` with `TODO(BL-27)`, `_residual_kernel`, `_project_out`, `_orthonormal_basis`) | 205 |
| `src/mllib/math/search_cost_function.py` | the contract; `lower_bounds` default = per-child | 45 |
| `src/mllib/math/algorithms/a_star_search.py` | calls `lower_bound` once at the root, `lower_bounds` once per expansion in generation order; FIFO ties | 98 |
| `src/mllib/math/linear_algebra_helpers.py` | `QuadraticFormHelper` only (BL-14 legacy); untouched by this plan | 12 |
| `tests/math/graph/test_nystrom_landmark_problem.py` | 27 tests incl. clamp injection through `_remaining_spectrum`, batched-vs-per-child equivalence, "A* expands the same states whether batched or not" | 422 |
| `tests/math/fixtures/` | `identity_8x8`, `all_ones_8x8`, `rbf_chain_{4,6,8,10}` — repeated-eigenvalue and rank-one cases already present | — |
| `examples/nystrom_batched_bounds.py` | per-child vs batched timing on SPECTF; harness both ways; no committed output | — |
| `tests/ml/test_training_baseline.py` | the snapshot pattern (`BASELINE_UPDATE=1`) P-6 copies | — |
| `docs/BACKLOG.md` BL-27 | the intent and the marker's location | — |

Not in the repo, contrary to the first draft: any pinned A* expansion count (the harness test pins
only selector evaluation counts), a five-scale harness run (the harness runs three scales), or a
root rank tolerance (the constructor only clips at zero; `RANK_TOLERANCE` applies to column blocks).

The reproduction's reference implementation: `~/develop/research/repro/astar-css/astar_css.py`
(`CSSProblem` for root decomposition + reduced coordinates; `top_eigs_rank1_downdate` for the secular
solver; the parent/children block inside `astar_css`). Read-only; nothing is imported from it.

## 3. Functional requirements

| FR | Requirement | Slice |
|---|---|---|
| FR-1 | `NystromLandmarkProblem` exposes `eigenvalues` (descending, clipped ≥ 0), `eigenvectors` (matching columns), `retained_rank` r (P-9), and `reduced_coordinates` = D_r^{1/2} V_rᵀ of shape (r, n), with `reduced_coordinatesᵀ · reduced_coordinates ≈ K` at δ = 0 on full-rank kernels and column norms² equal to `diag(K)`. | A+C |
| FR-2 | `top_eigenvalues_of_rank_one_downdate(eigenvalues, W, count) → (count, m)`: for each column w of W, the `count` largest eigenvalues of `diag(eigenvalues) − w wᵀ`, by bisection on the secular function within interlacing intervals; collapsed intervals (repeated eigenvalues) return the shared value; the last interval's lower end is `λ_r − ‖w‖²`; `count = 0` returns shape (0, m). | B |
| FR-3 | `NystromCssCostFunction.lower_bounds(parent, successors)` prices **every** depth: goal-depth children by the existing Schur batch; non-goal children by one `eigh` of H = D − Z Zᵀ at the parent and FR-2 for the children, with g = tr(H) − ‖z_j‖², h = Σ top-(k̄−1) downdated eigenvalues, f = max(g − h, 0); explained columns (P-10) get z = 0. Falls back to the oracle when the successors do not extend the parent, as today. | A+C |
| FR-4 | Every value FR-3 returns equals the oracle `lower_bound` for that child to within `1e-8 · tr(K)` at **every δ** (P-8), on every fixture and on seeded random kernels, at every depth, including rank-deficient and badly scaled kernels. | A+C, D |
| FR-5 | `AStarSearch` expands the same number of states and returns the same subset with FR-3 as with an oracle-only cost function, on all fixtures and seeded random kernels (n ≤ 12, k ≤ 4). Where the brute-force optimum set has more than one member, only membership is asserted. | A+C |
| FR-6 | `spectrum_mass_tolerance` δ on the problem: r per P-9; `reduced_coordinates`, `kernel_sqrt` and the non-goal bound use the truncated spectrum; `goal_cost` and the goal-depth Schur batch remain exact on the full K. | D |
| FR-7 | For δ > 0 the bound stays admissible: on brute-forceable fixtures and seeded kernels, for δ ∈ {1e-10, 1e-8, 1e-6, 1e-4}, every state's bound ≤ the true residual trace of every completion (+ A-2 slack), and A* returns a member of the brute-force optimum set. r = n at δ = 0 on a full-rank kernel; r = 1 on the all-ones fixture for any δ > 0. | D |
| FR-8 | Records: BL-29 opened (0a) and closed (D); BL-27 closed (A+C); `TODO(BL-27)` marker removed (A+C); D-26 (D); learning-log entries "the parent-once, child-cheap structure, completed" (A+C) and "truncation is admissible by Schur-complement monotonicity" with the proof (D); ARCHITECTURE §1 mentions `secular_equation.py` (B); the P-10 unification line under BL-29's close (D). | 0a, B, A+C, D |
| FR-9 | The search baseline of P-6 exists and is byte-identical after every code slice at δ = 0. | 0b, all |
| FR-10 | The example of P-7 prints per-child, batched (goal-depth only) and downdated timings on the same rows plus the five-scale block, and `--no-timings` output is committed and byte-identical after every code slice. | 0b (flag, output), A+C (third column) |
| FR-11 | Gates on every slice: `uv run ruff check . && uv run ruff format --check .`, `uv run pytest` (183 + new), the training baseline untouched, no new per-file ignores (BL-24 counts do not rise), reviewer-agent run before the PR. | all |
| FR-12 | `CONTEXT.md` and `docs/REFERENCES.md` exist; every new public name in this plan uses the glossary's canonical term (state, expansion, frontier, landmark, residual trace, retained rank, dropped mass, explained column, oracle). | 0a, all |
| FR-13 | Not built, recorded: Gragg's method (P-5); parent-from-grandparent downdate (spec §7); the grid runner, Laplacian kernel, ridge-leverage selector and tie-aware oracle (BL-28 and the research program's H2–H5) — a follow-on plan after this one is reviewed. | — |

## 4. Phases and slices

Each slice: branch off the previous slice's branch; gates per FR-11; reviewer-agent; Xavier commits
and opens the PR. Test names describe the behaviour (`test_<what_is_true>`), one behaviour per test,
deterministic (seeded), per `.claude/agents/testing-agent.md`.

| Slice | Branch | Change | Tests (new) | Done when |
|---|---|---|---|---|
| 0a | `docs/nystrom-downdate-plan` | This plan; `CONTEXT.md`; `docs/REFERENCES.md`; `CLAUDE.md` reading-list line; BL-29 entry (`backlog-only`, re-entry "slice D of this plan"); learning-log "fringe" → "frontier". | — | Merged. |
| 0b | `test/nystrom-search-baseline` | The search baseline (P-6) generated on `main`; the example's `--no-timings` flag, five-scale block and committed output (P-7). | the baseline test (fixtures + SPECTF cells + example rows); a test that the example's five scales are the ones printed | FR-9, FR-10 (flag, output); suite ≤ ~48 s. |
| B | `feat/secular-equation` | `mllib/math/secular_equation.py` with FR-2, module docstring stating the secular function, interlacing, the collapsed-interval rule and the Gragg deviation; ARCHITECTURE §1 line. | matches `eigvalsh(diag(λ) − wwᵀ)` to rtol 1e-9 on seeded random spectra and on the identity / all-ones / rbf-chain fixtures' spectra; `w = 0` leaves the spectrum; `count = 0` and `count = r` shapes; a hypothesis property over random descending λ ≥ 0 and w | FR-2; ~70 src / ~90 test. |
| A+C | `feat/nystrom-downdated-bound` | `NystromLandmarkProblem` keeps the root decomposition (FR-1); `lower_bounds` for every depth (FR-3) using B; marker removed, oracle docstrings per P-3; example's third column; BL-27 closed; learning-log entry. | reduced coordinates reproduce K and column norms equal the diagonal; identity has rank n and all-ones rank 1; a rank-deficient kernel drops its zero modes; equivalence with the oracle at every depth on every fixture and seeded kernels, incl. rank-deficient and badly scaled (FR-4); expansion counts and subsets unchanged vs an oracle-only subclass (FR-5), reusing the pattern of `test_a_star_expands_the_same_states_whether_bounds_are_batched_or_not`; the palindrome chain's tie handled via the optimum set; existing clamp-injection tests pass; explained columns priced as the parent; the search baseline and example output byte-identical | FR-1, 3, 4, 5, 8 (its part), 10 (column); ~160 src / ~200 test. |
| D | `feat/nystrom-spectrum-truncation` | `spectrum_mass_tolerance` (FR-6) with P-8's `kernel_sqrt` rule and P-9's rank rule; D-26; BL-29 closed with the P-10 line; learning-log entry with the proof. | admissibility and optimum membership at each δ (FR-7); the rank rules; FR-4 at δ > 0; δ = 0 leaves the search baseline byte-identical; `describe()` shows the parameter | FR-6, 7, 8 (its part); ~60 src / ~110 test. |
| E | `test/nystrom-reference-cells` | Only if P-11's 3 s budget holds once S1 is frozen: vendor `BL-27-reference-cells.json` under `tests/math/fixtures/`; one test asserting expansion counts and optimum membership per cell at δ = 0. | the vendored cells | FR-5 at grid scale, or a plan note saying the check stays research-side. |
| F | `docs/nystrom-downdate-close` | Plan status `complete` with a what-changed section (precedent: port plan §7, §8); the example's timing table labelled as this machine's; BL-24 counts unchanged. | — | Suite green; research `harness/CHANGES.md` H1/H1b marked done (Xavier's edit). |

Six or seven PRs. Estimated 1–2 days of implementation for B, A+C, D (matches the BL-27 entry), plus
E when S1 exists.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The downdated bound differs from the oracle in the last bits and A* breaks a tie the other way, changing a subset on a fixture. | FR-5 asserts membership in the optimum set where ties exist (the ten-point RBF chain is a palindrome); expansion counts must still match exactly. If a count differs, the bound is wrong, not the tie: stop and diff per depth with FR-4's test. |
| Repeated eigenvalues (identity, all-ones) collapse interlacing intervals; a naive bisection returns garbage or NaN. | FR-2's collapsed-interval rule and its fixture tests in slice B, before the bound touches it. |
| `eigh(H)` at the parent returns slightly negative eigenvalues from rounding; secular denominators cross zero. | Clip parent eigenvalues at 0 before the solve; the bound is clamped at 0 anyway. Test on a badly scaled kernel (existing fixture pattern). |
| Platform rounding (Accelerate vs OpenBLAS) makes an assertion pass locally and fail in CI. | Compare clamped values with tolerances (FR-4, A-2); never assert the sign of a cancellation. The baseline snapshot pins integers (expansion counts, subsets) and a residual trace with a tolerance. |
| The parent `eigh` is still n³ and the speedup at n = 80 disappoints. | The claim is "fewer decompositions", not a factor; FR-10 prints all three paths. Expected ≥ 10× on the n = 60, k = 5 row per the spec's estimate; if below 3×, profile before tuning. |
| Truncation changes expansion counts at δ > 0 and someone reads that as a bug. | FR-7 asserts optimality and admissibility only; counts at δ > 0 are reported by the research calibration (EXP-09a), not asserted here. δ = 0 is pinned by the baseline. |
| The baseline slice (0b) takes ~8 s of suite time on the current engine. | Accepted by the owner; it shrinks once A+C lands, and the rows are the same ones the example measures. |
| Scope creep into BL-28 (grid runner, Laplacian kernel, leverage selector). | FR-13: out of scope; a follow-on plan after this one is reviewed. |

## 6. Immediate next actions

1. Xavier reviews this rewrite and merges slice 0a.
2. Slice 0b on `test/nystrom-search-baseline` (any session; it is generation, not design).
3. The implementing session starts at §7 for B, A+C, D.

---

## 7. Handoff brief for the implementing session

**Start.**
```
cd ~/Desktop/BotMaker/botMaker
git checkout main && git pull                 # expect e3e92d1 or later with 0a and 0b merged
uv sync && uv run pytest -q                   # expect 183 + the baseline tests, all passing
git checkout -b feat/secular-equation         # slice B; then feat/nystrom-downdated-bound, …
```
Read, in order: `CLAUDE.md`; `CONTEXT.md`; `docs/DECISIONS.md` D-22 to D-25; `docs/BACKLOG.md`
BL-27, BL-28, BL-29; `docs/ARCHITECTURE.md` §2 and §4; `.claude/agents/testing-agent.md`; this plan;
the research spec `~/develop/research/nystrom/harness/BL-27-rank-one-downdate.md` §2, §5, §6, §10;
the reproduction's `top_eigs_rank1_downdate` and the parent/children block in
`~/develop/research/repro/astar-css/astar_css.py`.

**The math to implement, in the repo's names** (Y = K^{1/2}, so Y_r = D^{1/2} Vᵀ):

```
root      : K = V D Vᵀ (eigh; clip D ≥ 0); r = retained rank (P-9); Y_r = sqrt(D_r)[:, None] * V_r.T   (r × n)
            at δ > 0: kernel_sqrt = V_r sqrt(D_r) V_rᵀ as well (P-8); kernel_matrix stays K
parent S  : Q = orthonormal basis of Y_r[:, S]      (SVD, RANK_TOLERANCE, as _orthonormal_basis does)
            Z = sqrt(D_r)[:, None] * Q ;  H = diag(D_r) − Z Zᵀ ;  λ, U = eigh(H) (descending, clip ≥ 0)
            trace_parent = Σ λ                      (== trace of _residual_kernel(S) at δ = 0; test it)
children  : R = Y_r − Q (Qᵀ Y_r);  norms = ‖R[:, j]‖ ;  explained = norms ≤ 1e-12 · sqrt(tr K)
            q_j = R[:, j]/norms_j (0 where explained) ; z_j = sqrt(D_r) * q_j ;  W = Uᵀ Z_children
            g_j = trace_parent − ‖z_j‖²
            h_j = Σ top (k̄−1) eigenvalues of diag(λ) − w_j w_jᵀ        (slice B, vectorised over j)
            f_j = max(g_j − h_j, 0)
goal depth: unchanged D-24 Schur batch on the full K
```

**Do not:** change `AStarSearch`, `SearchCostFunction`, `goal_cost`, tie-breaking, or the D-24
goal-depth path; touch `PIVOT_TOLERANCE` (P-10); import anything from the research repo; add per-file
ruff ignores; assert the sign of a cancellation; assert a timing; regenerate the search baseline
(if it moves at δ = 0, the code is wrong); introduce a word the glossary lists under *Avoid*;
commit (Xavier commits).

**Do:** keep `lower_bound` as the oracle and test the new path against it at every depth and every δ;
use the existing fixtures (identity and all-ones are the repeated-eigenvalue and rank-one cases);
reuse the "same states whether batched or not" pattern for FR-5 with an oracle-only subclass
(`lower_bounds = SearchCostFunction.lower_bounds`, as the example does); write the learning-log
entries in the log's voice (what / where / design / what was confusing / reference, citing
`docs/REFERENCES.md` keys); run the reviewer-agent before each PR; leave §4 "Done when" verifiable.

**Numbers the review will check.** The search baseline (`tests/ml/nystrom_search_snapshot.json`)
and the example output (`examples/nystrom_batched_bounds.example_output.txt`) are byte-identical
before and after every slice at δ = 0. The example's "same" column reads True for all three paths.
Nothing else is quoted from memory.

**Reporting back.** For the review session: the PR list with per-slice test counts, the example's
timing table, any place where the spec and the code had to diverge and why, and any number that
moved.

---

## 8. What the grilling changed (2026-09-04)

The first draft was interviewed against the repo before acceptance. Corrections, all owner-decided:

| # | Draft said | Now | Why |
|---|---|---|---|
| 1 | Slice A keeps the root decomposition on its own. | Folded into A+C (P-4). | A layer nothing consumes violates D-25 and the no-unstarted-code rule. |
| 2 | Solver in `linear_algebra_helpers.py`. | Own module `secular_equation.py` (P-5). | One idea per module; the helper file is BL-14 legacy. |
| 3 | `kernel_sqrt` stays exact at δ > 0. | Built from the truncated kernel (P-8). | Otherwise the oracle and the fast path disagree at δ > 0 and D-24's contract silently narrows to δ = 0. |
| 4 | "The relative eigenvalue tolerance used today." | No such tolerance existed; P-9 defines it, and names the attribute `retained_rank`. | The constructor only clipped at zero; `numeric_rank` was the wrong name at δ > 0. |
| 5 | "363 / 501 / 668 / 780 / 781 across the five scales." | Removed; P-6/P-7 pin real numbers from a run on `main`. | Not in the repo; the harness has three scales and no test pins an A* count. |
| 6 | "Pull first, local main is behind." | Removed. | `main` was at `e3e92d1`, level with origin. |
| 7 | Slice E with a ≤ 10 s test or a `slow` marker. | ≤ 3 s or research-side (P-11). | Suite budget; no marker convention exists and none is introduced for a test that may never ship. |
| 8 | No glossary. | `CONTEXT.md`, `docs/REFERENCES.md`, eight term rulings (P-12); "frontier" over "fringe" and "open list" (fourth-edition Russell & Norvig usage). | Terms were drifting between the code (`still_needed`, `gamma_scale`), the spec (k̄, bandwidth) and the research plan (gap 1 / gap 2, H1 meaning two things). |
| 9 | Two rules for "already explained". | Recorded as debt under BL-29 (P-10). | Unifying now would touch the D-24 path before the baseline exists to prove it moves nothing. |
