# Nyström landmark selection — port plan

Status: **complete** · Written 2026-09-04 · Accepted and executed 2026-09-04 · Owner: Xavier
See §7 for what changed against the plan.
Reference implementation: branch `chore/nystrom-wip-snapshot` @ `7470c34` (never merges).
Target: stacked slices from `feat/nystrom-landmark-selection` into `main`.
Builds on `docs/plans/2026-08-refactor.md` (process precedent) and
`~/develop/research/candidates/nystrom-certified-landmarks.md` (why the code exists).

---

## 0. Why

On 2026-09-03 a research sprint produced, directly on `main`, a working A* column-subset-selection
solver for Nyström landmarks plus baselines and a UCI harness: 22 files, 2,675 lines, one uncommitted
lump. It works (A* matches brute force on every bundled instance) but it violates the framework the
repo adopted on 2026-08-28: it is on `main`, it is nine times the slice ceiling, `AStarSearch`
overrides `run` against `ARCHITECTURE.md` §4, the O(n³) bound has no `TODO(BL-27)` marker, and the
tests reach data by a cwd-relative path.

The research side (`research/SPRINT.md` § Where artifacts live) says exactly what to do: sprint code
is exploratory; "if it earns a longer life, port it into BotMaker's mllib under that repo's
contribution standards." It has earned it — the candidate is the sprint's anchor finalist and the
harness is its measurement instrument. This plan is that port: the same functionality, rebuilt as
reviewable slices, each with its records.

## 1. Decisions and assumptions

Decided 2026-09-04 (owner, in session):

| # | Decision |
|---|---|
| P-1 | Snapshot branch `chore/nystrom-wip-snapshot` holds the working tree verbatim; the clean branch `feat/nystrom-landmark-selection` starts from `main`. Claude committed the snapshot on explicit instruction (D-8 exception, one-off). |
| P-2 | Source of truth is the working tree. BL-27 (rank-one downdate) stays a backlog item; the research repro's `astar_css.py` is not ported here. |
| P-3 | One PR per slice, stacked in order, squash-merged. |
| P-4 | The three UCI files are committed under `data/uci/` (all < 1 MB, D-19). The six dummy kernel CSVs become test fixtures, not `data/`. |

Assumptions, all confirmed by the owner on 2026-09-04:

| # | Assumption | Why |
|---|---|---|
| A-1 | **`AStarSearch` conforms to `AbstractGraphAlgorithm`**: implements `_search`, does not override `run`. The base's `run` accepts `context: SearchContext \| None` and its `graph` may be an `AbstractGraphProblem`. Recorded as **D-23** (implicit-graph problems are a second input type for graph algorithms) and as a new §4 extension-point line. | ARCHITECTURE §4: "never override `run`". The snapshot overrides it to pass `None`. |
| A-2 | **The cost contract is terminal-objective only**: `SearchCostFunction` keeps `lower_bound(state)` (admissible bound on the best completed solution through `state`, i.e. the f-value) and `goal_cost(state)`. `edge_cost` and the carried `path_cost` are dropped: both are always zero and never affect ordering in the snapshot. Tie-break is FIFO by insertion. | No unstarted code (CONTRIBUTING). The repro found tie-breaking irrelevant outside the power-set regime (results.md D10). |
| A-3 | **The dummy-kernel project module (`ml/projects/nystrom_landmark_selection.py`) is not rebuilt.** Its six kernels move to `tests/math/fixtures/` as known-answer cases, contents verbatim and renamed for what they are (`rbf_chain_4x4.csv`, `identity_8x8.csv`, …), (identity 8×8, k=3 → cost 5.0; all-ones 8×8 → 0.0; 1-D RBF chains). The UCI harness is the project's composition root. | The module is three lines of composition around data that only exercises the solver; precedent is `baseline_snapshot.json` living beside its test. |
| A-4 | **`data/uci/README.md`** records source URL, licence (UCI, CC BY 4.0), row/column counts and the loader convention for each file. Line endings are normalised to LF on commit (pre-commit warned on two files). | D-19 precedent records provenance; the harness's label conventions are otherwise only in code. |
| A-5 | **D-22 — reporting rule for optimality-gap harnesses**: published baselines are named as such; randomised selectors are quoted as a mean and median over independent seeds; best-of-N is named with its N; the SVD rank-k residual and subset/SVD ratio are reported next to every algorithm-vs-optimum ratio; A* node counts and C(n,k) are printed together. | This is the learning-log's "what was confusing": the first harness inflated the gap by labelling best-of-32 as "random" and a lookahead greedy as "greedy". It must not happen again, in this repo or the paper. |
| A-6 | **The learning-log entry is split by technique and lands with its slice**: A* with a spectral bound (1.1/2.1), greedy Nyström and pivoted Cholesky (3.1), RPCholesky and seed-summarised randomised selectors (3.2), the harness's confusion note (4.2). The 2026-09-03 block on the snapshot is the source text. | CONTRIBUTING: a technique implemented by hand → entry in the same PR. |
| A-7 | **BL-28** records the first-experiment needs the port does not build: Laplacian kernel, ridge-leverage baseline, n = 500–3000, k ≤ 10, five bandwidths, RPCholesky over 10 draws. Interfaces are not pre-shaped for it. | Candidate one-pager § First experiment; no unstarted code. That experiment runs on `research/repro` until BL-27 is done. |
| A-8 | **Test data path** comes from a `conftest.py` fixture resolving the repository root, not `Path("data/uci")`. | The snapshot tests pass only when pytest runs from the repo root. |

## 2. Current state (evidence)

Captured 2026-09-04. `main` @ `10bdc90` is green. Snapshot `7470c34`:

| Area | Files | Lines (src / test) |
|---|---|---|
| Search contracts + A* | `search_cost_function.py`, `graph/abstract_graph_problem.py`, `algorithms/a_star_search.py` | 140 / 66 |
| Nyström problem + cost | `graph/nystrom_landmark_problem.py` | 140 / (in the above) |
| Selectors | `algorithms/nystrom_landmark_selectors.py` | 373 / 160 |
| Dummy project | `ml/projects/nystrom_landmark_selection.py` | 66 / 34 |
| UCI harness | `ml/projects/nystrom_uci_harness.py` | 331 / 167 |
| Data | 6 dummy kernels, 3 UCI files (412 KB) | — |
| Docs | BL-27 entry, learning-log entry | — |

Review findings against the standards (see § 0): on `main`; ~1,800 lines of code in one change;
`run` override; no `TODO(BL-27)` marker; cwd-relative test data; no `DECISIONS.md` entry; the repo's
`CONTRIBUTING.md` is missing the "Behavioural baseline" section back-ported to the standards on
2026-08-29 (fix in slice 0.1). Pre-commit passed on the snapshot after ruff-format and end-of-file
fixes to four files. The training baseline is unaffected by any of this code.

## 3. Functional requirements

Pulled from the snapshot code, the engineering standards, the repo's docs, and the research notes.
Each requirement names the slice that delivers it.

### From the code (what must exist again)

| FR | Requirement | Slice |
|---|---|---|
| FR-1 | `AbstractGraphProblem[State, Action]`: `initial_state`, `is_goal`, `successors` over an implicit graph. | 1.1 |
| FR-2 | `SearchCostFunction[State]`: `lower_bound` (admissible) and `goal_cost`; `SearchResult(state, cost, optimal, nodes_expanded)` frozen dataclass. | 1.1 |
| FR-3 | `AStarSearch`: best-first by lower bound, FIFO ties, returns `optimal=True` on the first goal popped, raises when no goal is reachable; counts expansions. | 1.1 |
| FR-4 | `NystromLandmarkProblem`: validates 2-D, square, symmetric, `0 < k ≤ n`; states are ascending index tuples so each subset is one state; successors generated in canonical order; `K^{1/2}` via `eigh` with negative eigenvalues clipped. | 2.1 |
| FR-5 | `NystromCssCostFunction`: `goal_cost = tr(K − K_S K_SS⁺ K_S^T)` clamped ≥ 0; `lower_bound` = residual energy after projecting out the selected columns of `K^{1/2}` minus the top-r eigenvalues of the deflated residual Gram; equals `goal_cost` at goal depth; root bound equals the SVD rank-k residual. O(n³) per child carries `# TODO(BL-27)`. | 2.1 |
| FR-6 | Selector contract (`name`, `select(problem, cost_function) → SearchResult`, `optimal=False` for heuristics, `cost` always the true residual); `run_selector_suite` keyed by name. | 3.1 |
| FR-7 | Published baselines: A* wrapper, greedy residual-trace (Farahat 2011), pivoted Cholesky (largest residual diagonal). | 3.1 |
| FR-8 | Randomised and instrumented selectors: RPCholesky (seeded, diagonal-proportional pivot), uniform single draw (seeded), best-of-N (named `best_of_N_random`, `nodes_expanded = N`), greedy-on-lower-bound (instrumented); `summarize_randomized_selector` → mean/median/min/max over `trials` seeds from `base_seed`. | 3.2 |
| FR-9 | UCI loading: `UciDatasetSpec` (delimiter, label position first/last/none, `wdbc` loader dropping id and diagnosis), `standardize_columns` (zero std → 1), `downsample_rows` (seeded, sorted indices), `build_rbf_kernel` (median-distance heuristic × `gamma_scale`, unit diagonal, explicit `gamma` override), `svd_rank_k_residual`. | 4.1 |
| FR-10 | `run_nystrom_on_uci_dataset` → `UciHarnessResult` with: all selector results, ratios to optimum (safe at zero), randomised summaries and their mean ratios, SVD residual, subset/SVD ratio, `subset_count = C(n,k)`, gamma used; `run_small_uci_suite` over specs × gamma scales; `format_run` labelling optimum / baseline / instrumented; `main` over scales (0.25, 1, 4). | 4.2 |
| FR-11 | Known-answer fixtures: the six dummy kernels with their expected optimal costs. | 2.1 |

### From the engineering standards and repo docs (how it must arrive)

| FR | Requirement | Slice |
|---|---|---|
| FR-12 | Every slice is one end-to-end piece of functionality with its tests and records (D-25 retired the ~300-line ceiling this originally cited), branch `<type>/<slug>`, PR title one sentence; reviewer-agent run before each PR. | all |
| FR-13 | Deterministic tests, one behaviour each, named for the behaviour; hypothesis properties where an invariant exists: `lower_bound(s) ≤ goal_cost(t)` for every completion `t` of `s`; `goal_cost ≥ 0`; every heuristic's cost ≥ A*'s; A* equals brute force for `n ≤ 9`. | 1.1–3.2 |
| FR-14 | `ruff check`, `ruff format --check`, full `pytest`, and the training baseline pass on every slice; no new per-file ignores (BL-24 counts do not rise). | all |
| FR-15 | Records in the same PR: D-22, D-23 (0.1); BL-27 with the marker (2.1); BL-28 (4.2); learning-log entries per A-6; ARCHITECTURE §4 line (1.1) and §1 table row (5.1). | as listed |
| FR-16 | `ARCHITECTURE.md` §4 rule honoured: A* subclasses the base, implements `_search`, does not override `run`. | 1.1 |
| FR-17 | Datasets under 1 MB each with provenance recorded (D-19, A-4). | 4.1 |

### From the research notation (what the port must preserve for the candidate)

| FR | Requirement | Slice |
|---|---|---|
| FR-18 | The two gaps are reported separately: algorithm-vs-best-subset and best-subset-vs-SVD (candidate § The combination; D-22). | 4.2 |
| FR-19 | The certificate's cost is visible: A* `nodes_expanded` printed next to `C(n,k)` (BL-27 text; repro lesson 1). | 4.2 |
| FR-20 | Baselines are the published ones the candidate will be compared against: greedy trace, pivoted Cholesky, RPCholesky, uniform (candidate § First experiment; READING B1). | 3.1–3.2 |
| FR-21 | Numerical robustness carried over from the repro: bounds clamped ≥ 0 (results.md D13); eigenvalues clipped; pinv with `hermitian=True`. | 2.1 |
| FR-22 | Bandwidth is a first-class sweep parameter, since flat spectra (wide bandwidth) are where the candidate expects A* to struggle and greedy to err (candidate § The gap). | 4.2 |
| FR-23 | Not built, recorded: rank-one downdate (BL-27) and the n ≥ 500 experiment grid (BL-28, A-7). | 2.1, 4.2 |

## 4. Phases and slices

Each slice: branch off the previous slice's branch; `uv run ruff check . && uv run ruff format --check .`;
`uv run pytest`; reviewer-agent; Xavier commits and opens the PR. "Done when" is in addition to that.

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 0.1 | `docs/nystrom-plan` | This plan; D-22, D-23 in `DECISIONS.md`; `CONTRIBUTING.md` regains the behavioural-baseline section from the standards @ current commit. | Plan status `accepted`; assumptions A-1…A-8 confirmed or amended in § 1. |
| 1.1 | `feat/implicit-graph-search` | `AbstractGraphProblem`, `SearchCostFunction`, `SearchResult`, `AStarSearch` under the base contract (A-1, A-2); base `run` accepts `None` context. Tests on a toy pick-k-of-n problem with diagonal weights and on an unreachable-goal problem; hypothesis: A* equals brute force on random small terminal-objective problems. ARCHITECTURE §4 line. Learning-log: A* over ordered subsets. | FR-1–3, FR-16; ~150 src / ~100 test. |
| 2.1 | `feat/nystrom-css-problem` | `NystromLandmarkProblem`, `NystromCssCostFunction` with `# TODO(BL-27)`; BL-27 entry; six kernel fixtures under `tests/math/fixtures/` with known answers. Tests: validation errors, canonical successors, goal cost vs direct formula, root bound = SVD residual, hypothesis admissibility, A* = brute force on random RBF kernels (n ≤ 9). Learning-log: the trace-residual = Frobenius-CSS reduction and the spectral bound. | FR-4, 5, 11, 13, 21; ~150 src / ~130 test. |
| 3.1 | `feat/nystrom-greedy-selectors` | Selector ABC, `AStarLandmarkSelector`, `GreedyResidualTraceLandmarkSelector`, `PivotedCholeskyLandmarkSelector`, `_pivoted_cholesky_step`, `run_selector_suite`. Tests: Schur-complement check of greedy picks, diagonal-kernel picks, suite keys, heuristics never beat A*. Learning-log: greedy Nyström and pivoted Cholesky. | FR-6, 7, 20; ~170 src / ~90 test. |
| 3.2 | `feat/nystrom-randomized-selectors` | `RandomlyPivotedCholeskyLandmarkSelector`, `RandomSamplingLandmarkSelector`, `BestOfRandomSamplingLandmarkSelector`, `GreedyLowerBoundLandmarkSelector`, `RandomizedSelectorSummary`, `summarize_randomized_selector`. Tests: seed determinism, single draw = 1 evaluation and best-of-N = N, summary ordering, validation. Learning-log: RPCholesky and why randomised selectors are seed-averaged. | FR-8, 20; ~200 src / ~90 test. |
| 4.1 | `feat/nystrom-uci-data` | `data/uci/` three files + README (A-4); `UciDatasetSpec`, `load_feature_matrix`, `standardize_columns`, `downsample_rows`, `build_rbf_kernel`, `svd_rank_k_residual`; root-path fixture in `tests/conftest.py` (A-8). Tests on `tmp_path` files and small arrays. | FR-9, 17; ~170 src / ~100 test (+412 KB data, not counted). |
| 4.2 | `feat/nystrom-uci-harness` | `UciHarnessResult`, `run_nystrom_on_uci_dataset`, `run_small_uci_suite`, `format_run`, `main`, `PUBLISHED_BASELINES`; BL-28 entry (A-7). Tests: one dataset at n=14, k=2 checking both gaps, summaries, labels; suite over two scales. Learning-log: the confusion note. | FR-10, 18, 19, 22, 23; ~170 src / ~100 test. |
| 5.1 | `docs/nystrom-close` | Plan status `complete` with a what-changed section (precedent: refactor §7); ARCHITECTURE §1 row for `graph/` and `algorithms/` implicit-search additions; README run line for the harness; research-side pointers are Xavier's (candidate log, orchestrator `next`). | Suite green; `chore/nystrom-wip-snapshot` deleted or left as reference by owner's choice. |

Eight PRs. Line estimates come from the snapshot and will shrink where the rebuild simplifies.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The rebuild silently changes a number the research notes already quote (1.000–1.034× greedy, 1.21–1.27 subset/SVD on the three 40-row slices). | Before slice 4.2 merges, run the snapshot's `main()` and the rebuilt `main()` at the default settings and diff the printed blocks; record the diff in the PR. This is the behavioural-baseline rule applied to a port. |
| A-2 (dropping `edge_cost`) is later regretted by a path-cost problem. | Reintroduce it then, with the problem that needs it; a docstring on `SearchCostFunction` states the terminal-objective scope. |
| Suite runtime grows past the ~40 s the guide promises. | Keep harness tests at n ≤ 14, k = 2, `randomized_trials ≤ 8`; hypothesis examples capped at n ≤ 7. |
| The candidate's first experiment is due runnable by 2026-09-17 and this port does not reach n ≥ 500. | Decoupled by A-7: the experiment runs on `research/repro/astar-css` with Y = K^{1/2}; the port is the reference implementation, not the experiment rig, until BL-27. |
| Stacked PRs conflict when an earlier slice is amended in review. | Rebase the stack in order after each merge; each slice touches disjoint files by construction. |

## 6. Immediate next actions

1. Xavier reviews § 1 assumptions A-1…A-8 and § 4; amends or accepts.
2. Slice 0.1: mark this plan `accepted`, add D-22 and D-23, restore the CONTRIBUTING section; PR.
3. Slice 1.1 begins from the 0.1 branch.

---

## 7. What changed against the plan

The port landed as planned in eight slices, with three deviations, all recorded here rather than
folded in silently.

| # | Deviation | Why |
|---|---|---|
| 1 | **The selectors are two modules, not one.** `nystrom_landmark_selectors.py` holds the deterministic rules (A*, greedy trace, pivoted Cholesky, greedy-on-bound) and `nystrom_randomized_selectors.py` holds the seeded ones plus `summarize_randomized_selector`. | Slices 3.1 and 3.2 were specified as separate PRs, and one module would have put both in the same file, so the stack could not be reviewed independently. The split also matches D-22: the randomized selectors are exactly the ones that must never be quoted from a single run, and the machinery that averages them now lives with them. |
| 2 | **The UCI data layer is its own module.** `nystrom_uci_data.py` (loading, standardizing, subsampling, kernel building, SVD residual) is separate from `nystrom_uci_harness.py` (the experiment). | Same reason for slices 4.1 and 4.2. It also makes the choices that decide *what is being measured* readable and testable without running a search. |
| 3 | **`_pivoted_cholesky_step` and `_safe_cost_ratio` are public** as `pivoted_cholesky_step` and `cost_ratio`, and the snapshot's inline label logic is extracted as `selector_kind` / `selector_label`. | The first two are imported across module boundaries; `cost_ratio` is worth testing directly, since its zero-optimum behaviour is a decision rather than an accident; the labels are the mechanism D-22 rests on and must be testable without rendering a whole block. |
| 4 | **Six of eight slices exceeded the ≤ ~300 changed-line guideline** that CONTRIBUTING carried at the time, at roughly 385 (1.1), 470 (2.1), 400 (3.1), 380 (3.2), 325 (4.1) and 430 (4.2). | Splitting further would have separated a cost function from the tests that pin its numerics. The owner retired the ceiling instead: a slice is a vertical, end-to-end piece of functionality (D-25), which each of these already was. No longer a deviation. |
| 5 | **`GreedyLowerBoundLandmarkSelector` ships in slice 3.1, not 3.2** as FR-8's mapping says. | It is deterministic, so it belongs with the deterministic module; only its "instrumented rather than published" status connected it to the randomized group. The §3 FR table is otherwise accurate. |

Everything else is as specified. All 23 functional requirements are delivered.

### The number check

The plan's leading risk was that the rebuild would silently change a number the research notes already
quote. Both harnesses were run at their defaults — three datasets, k = 3, n = 40, bandwidth scales
0.25/1/4, 50 seeds — and the printed output was **byte-identical** between `chore/nystrom-wip-snapshot`
and this branch. The rebuild is a restructuring, not a change of result.

Review then changed the printed labels deliberately (see below), so the blocks are no longer
byte-identical. **Every numeric field still is**, which is the property that mattered.

### What review changed

The reviewer-agent ran against the finished tree. Five findings were acted on:

1. **D-22 was violated by the harness's own output.** `format_run` printed a single-draw ratio for
   `rpcholesky` and `random_single_draw` tagged `[baseline]`, in the same column and with the same
   label as the deterministic `greedy_trace`. That is precisely the confusion D-22 exists to prevent,
   in the one artefact a reader copies numbers out of. Those lines now read `[baseline, 1 seed]`.
2. **D-23 claimed a guarantee the code did not provide.** The decision says algorithms over a
   materialized `Graph` still require a context, but `run()` with no context reached
   `BreadthFirstSearch._search` and failed with an `AttributeError` on `None`. The base class now
   raises a clear `ValueError`, and a test pins it.
3. **The numerical clamps had no test.** FR-21 names them and the learning log calls them
   load-bearing, yet deleting either `max(..., 0.0)` left the suite green. Two tests now construct
   badly scaled rank-deficient kernels where the unclamped value is genuinely negative.
   *Amended after the first CI run:* one of those tests failed on Linux, where OpenBLAS rounds the
   same cancellation to exactly zero rather than below it. Asserting the sign of rounding noise is
   not deterministic across platforms. The cost function now exposes the spectrum of what remains
   as `_remaining_spectrum` (the exact method BL-27's downdate replaces), the clamp tests inject the
   overshoot through that seam and through `_residual_kernel`, and a third test asserts on real
   instances only that no clamped value is ever negative.
4. **`main` and its bandwidth sweep were unpinned**, though FR-10 requires them. A test now asserts
   the three scales it passes.
5. **A learning-log figure was wrong**: the single-draw range was 1.22–1.42, written as 1.21–1.39.
   Corrected, with a note, in the entry that exists because of exactly this kind of error.

One optional finding was also taken: an unreachable pivot guard in `PivotedCholeskyLandmarkSelector`
was removed, since the problem already rejects the only condition that could trigger it.

### State at close

| Gate | Result |
|---|---|
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 98 files already formatted |
| `uv run pytest` | 167 passed (161 at first close; six added by review) |
| Training baseline | passes, untouched by this work |
| BL-24 lint debt | unchanged; no new per-file ignores |

Open items this work created: **BL-27** (the O(n³)-per-child bound, marked in the code) and **BL-28**
(the first-experiment grid). Neither is started, both are recorded. The research-side pointers — the
candidate log in `research/candidates/nystrom-certified-landmarks.md` and the orchestrator's `next` —
are the owner's to update.

---

## 8. Slice 6.1 — batched sibling scoring (added 2026-09-04)

Added after close, on the owner's request, from a performance assessment. Branch `feat/batched-bounds`.

**Why.** Profiling one harness run (SPECTF, n = 40, k = 3) put 91% of A*'s time in scoring goal-depth
children one at a time, 8,042 pseudo-inverses of 3×3 blocks at about 70 µs each. That is call overhead,
not arithmetic, so the fix is fewer calls, not faster ones. A GPU was assessed and rejected on evidence
(D-24): no CUDA on this machine, no Metal eigendecomposition in any available library, NumPy already on
Accelerate, and matrices too small to amortize a kernel launch.

**What.** `SearchCostFunction.lower_bounds(parent, successors)` with a per-child default; `AStarSearch`
and the greedy-on-bound selector call it once per expansion in generation order; the Nyström override
prices every goal-depth child from one Schur complement of the parent. Non-goal children fall back:
what they share is the eigendecomposition, which is BL-27. Nine tests pin batched against per-child at
every depth, on rank-deficient and badly scaled kernels, and pin that A* expands the same states either way.

**Measured with the project's own scripts** (`examples/nystrom_batched_bounds.py`, Apple Silicon,
NumPy 2.5 on Accelerate):

| n | k | per-child | batched | speedup | states expanded |
|---|---|---|---|---|---|
| 40 | 3 | 0.36 s | 0.10 s | 3.6× | 668 |
| 60 | 3 | 1.54 s | 0.33 s | 4.7× | 1,650 |
| 80 | 3 | 3.61 s | 0.98 s | 3.7× | 2,837 |
| 40 | 4 | 2.53 s | 1.05 s | 2.4× | 5,645 |
| 60 | 4 | 13.63 s | 5.48 s | 2.5× | 15,740 |

| Full harness, 3 datasets × 3 bandwidths, n = 40, k = 3, 50 seeds | |
|---|---|
| per-child | 2.24 s |
| batched | 0.85 s |
| speedup | 2.6× |

The gain shrinks as k grows because a larger share of the search is non-goal children, whose bound is
still the O(n³) path. That remaining share is exactly BL-27's target, and `lower_bounds` is now where it
plugs in. The harness output is numerically identical to the snapshot on all 549 fields, including every
A* node count and chosen subset: batching changed the cost of the search, not the search.

**One thing it broke, correctly.** The ten-point RBF chain fixture is a palindrome, so two mirror-image
subsets tie exactly; batched and per-child arithmetic differ in the last bit and return different
mirrors. The brute-force oracle now returns the set of optima rather than the first one found.
