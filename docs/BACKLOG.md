# Backlog — stripped initiatives and code-level TODOs

Registry of everything that was removed from the tree, or left in with a `TODO(BL-nn)` tag, under the strip rule in
`docs/plans/2026-08-refactor.md` (D-4). Phase-level milestones live in the orchestrator repo (`projects/botmaker.md`); this file is the
code-adjacent detail. When an item is done, move it to the **Closed** section with the commit that closed it.

Every in-code `TODO` must reference an id here: `# TODO(BL-nn): …`. CI rejects any other form.

| Field | Meaning |
|---|---|
| Verdict | `deleted` (gone from tree, restore from this entry + git history) · `kept` (in tree, tagged) · `moved` · `fix` · `backlog-only` |
| Re-entry | Estimated effort to rebuild from this entry |
| Phase | Where the refactor plan schedules it, if scheduled |

## Open

### BL-01 — FastAPI contract mock · `deleted` · re-entry 2–3 h

Removed 2026-08-28 (slice 0.3): `fastapi_app/` (`main.py`, `models.py`, `services.py`, `requirements.txt`, `api_tests.http`),
`Dockerfile`, `buildspec.yaml` (AWS CodeBuild → ECR us-east-2, dormant), `initDummyDataForStrategy.json`, root `__init__.py`.

Why removed: no consumers; `services.py` returned a hardcoded payload; the deploy pipeline had not run in months (D-9).
Why it matters later: the response shape was the intended join point to tradePlatform (see BL-19). Saved contract:

```
GET  /                                   → str
GET  /api/alerts                         → list
GET  /api/alerts/{alert_id}
POST /api/alerts                 201     ← CreateAlertRequest
GET  /api/strategybacktester             → StrategyBacktestResponse
GET  /api/strategybacktester/{strategy_id}
POST /api/strategybacktester     201     ← CreateStrategyBacktestRequest
```

```python
class CreateAlertRequest(BaseModel):
    name: str
    threshold: int


class CreateStrategyBacktestRequest(BaseModel):
    name: str
    threshold: int


class TickerAllocation(BaseModel):
    symbol: str
    displayName: str
    allocation: float


class Timeframe(BaseModel):
    start: str
    end: str  # ISO 8601


class StrategySettings(BaseModel):
    initialCapital: int
    rebalanceFrequency: str
    benchmark: str


class StrategyIndicator(BaseModel):
    name: str
    parameters: dict[str, Any]


class RiskManagement(BaseModel):
    stopLoss: float
    takeProfit: float
    positionSizing: str


class StrategyAdditionalProperties(BaseModel):
    indicators: list[StrategyIndicator]
    riskManagement: RiskManagement
    notes: str


class StrategyBacktestResponse(BaseModel):
    strategyId: str
    strategyName: str
    description: str
    tickers: list[TickerAllocation]
    timeframe: Timeframe
    settings: StrategySettings
    additionalProperties: StrategyAdditionalProperties
```

Example payload (was `initDummyDataForStrategy.json`): `STRAT-001` "Moving Average Crossover", AAPL 0.4 / MSFT 0.6,
2024-01-01 → 2024-12-31, initialCapital 100000, monthly rebalance, benchmark SPY. Stack: fastapi ≥ 0.110, uvicorn ≥ 0.24,
pydantic ≥ 2.6, python 3.11-slim image on port 8000. Full source: `git show e29cd03:fastapi_app/main.py` etc.

Re-entry path: when tradePlatform needs MlLib models, build the seam there (it already has a FastAPI app) and expose MlLib as a
plugin — do **not** resurrect a second web app here unless a decision record says otherwise.






### BL-07 — Probability placeholders: sum rule, product rule, Bayes rule, Prior, Gaussian Prior · `kept` · CS 6344 pairing
`MlLib/math/probability/sum_rule.py`, `product_rule.py` — `pass` bodies, but imported by `test_probability.py`.
Implement alongside `bayes_rule.py` / `Prior.py` when probability is covered.

### BL-08 — InformationGain / ChiSquare split criteria · `kept`
`MlLib/math/graph/split_function.py` — placeholder subclasses beside a working `Gini`. Removing cleanly touches the
`SplitFunction` hierarchy and `DecisionTree` injection (> 1 h). Implement when decision trees are revisited (BL-13).

### BL-11 — Trained-model exporter · `kept` · re-entry 1–2 h
`MlLib/ml/projects/ad_click_logistic_regression.py:29,52` — `self.exporter = None`. Intent: persist fitted weights
+ hypothesis config + evaluator record so a trained model can be reloaded without re-running the grid.

### BL-13 — Decision tree debt · `kept`
`MlLib/ml/decision_tree.py:23` refactor onto `graphBased` utilities · `:115` missing no-split error handling (fixed in
slice 3.4) · `:133` "abstracted later".

### BL-14 — Formerly never-imported modules · smoke-tested in slice 3.6
Survey (2026-08-28) called these "real implementations"; on inspection only two were: `graphBased/visualizer.py`
(matplotlib/networkx animation) and `linear_algebra_helpers.py` (`QuadraticFormHelper.compute_q`). Both now have smoke
tests. `probabilityBased/bayes_rule.py`, `Prior.py`, `gaussian_prior.py` are placeholders → folded into BL-07;
`regularization_function.py` is a placeholder → folded into BL-10. `gaussian_prior.py` assigned the *type* `float`
to `variance` — fixed. Kept by owner decision (D-4 named set), tagged in slice 3.7.

**Amended 2026-09-10 (BL-43 slice 4).** `math/graph/visualizer.py` is deleted, and with it the smoke test
that was its only importer. The animation it offered — one matplotlib redraw per visited node, watchable once
at the speed of its own loop — is now a recording plus a walkthrough page: `TraversalRecorder`
(`visualization/recorders/graph_traversal.py`) watches BFS and DFS through the D-32 seam, and
`examples/graph_search_vs_networkx.py` writes `bfs.json` and `dfs.json` instead of animating. What the old
example produced was fingerprinted before the deletion and is now pinned by
`tests/math/algorithms/test_graph_search_example_fingerprint.py` (CONTRIBUTING § "Behavioural baseline
before a refactor"). Re-entry: none — the module is not coming back, and the D-4 named set loses it.
`linear_algebra_helpers.py` and the placeholders are unaffected and keep their smoke tests.

### BL-19 — tradePlatform plugin seam · `backlog-only`
North star (D-2): MlLib models plug into tradePlatform as strategy plugins with descriptors introspected at the boundary
(`docs/reviews/2026-07-18-architecture-review.md` §Cross-codebase). Prerequisites: BL-16 (introspected metadata), BL-11 (exporter).

### BL-24 — Lint debt behind temporary per-file ignores · `fix`
`pyproject.toml [tool.ruff.lint.per-file-ignores]` lists rules the code still violates so the ruff gate stays on.
Counts at close-out 2026-08-29 (src+examples): E501 ×149, F841 ×2, B007 ×2, E711 ×3, B905 ×1, E402 ×3. Cleared during
the refactor: TD004, RUF012, the N naming rules (216 identifiers). Rule: a slice that touches a file fixes that
file's ignored violations; a code leaves the list when its count reaches zero; nothing is added without an entry.

### BL-26 — Perceptron and SVM onto the descent base · `kept`
`ml/perceptron.py` and `ml/svm.py` are identical except for the update sign and use the three-argument
sub-gradient (`compute_gradient(actual, predicted, data_values)`) plus `compute_bias`. Aligning the loss
gradient signature (return the weight gradient given the design matrix for every loss) would let both sit on
`GradientDescentModel` with `predict_method = "compute_classification"`. Re-entry ≈ 1 h; needs the margin
tests in `tests/ml/test_perceptron_svm.py` as the guard.

### BL-28 — Nyström first-experiment grid · `backlog-only` · re-entry 1 day, after EXP-09a calibrates δ

The harness measures the two gaps on three 40-row slices at k = 3, which is a smoke test, not
evidence. The `nystrom-certified-landmarks` first experiment needs: a Laplacian kernel beside the RBF
one, a ridge-leverage-score baseline, n = 500–3000, k up to 10, five bandwidths per kernel, and
RPCholesky averaged over ten draws. None of it is built and no interface here is pre-shaped for it.
The engine is no longer the blocker: BL-27 made each child a rank-one downdate and BL-29 truncates
the spectrum (D-26). What remains is choosing δ, which is the research-side calibration EXP-09a;
until it runs, n ≥ 1,000 results have no tolerance to quote.

### BL-30 — One rule for an explained column · `backlog-only` · re-entry 1 hour

The goal-depth batch treats a column as explained when its residual diagonal is at or below an
absolute `PIVOT_TOLERANCE`; the downdated path uses a relative rule, residual norm at or below
`1e-12 · sqrt(tr K)`. Same concept, two thresholds (plan P-10). Unify on the relative rule and show
the search baseline unmoved; if it moves on a badly scaled kernel, that is the decision to record.

### BL-31 — Nyström search engine performance · `backlog-only` · re-entry ½ day per item

The downdated engine (BL-27) made an expansion cheap enough at n = 1,000 to hit the memory wall that
stopped EXP-01: the A* frontier stores every priced child at 124 bytes each, and a flat kernel fills
it in minutes. At full rank the secular solve also costs more than the parent decomposition, which is
the trigger plan P-5 named for revisiting Gragg's method. Options, measurements and a suggested order
are in `docs/reviews/2026-09-05-nystrom-engine-performance.md`; first items: prune children above a
greedy incumbent at push time (M1), Gragg/Melman iteration for the secular solve (T2), an
array-backed frontier (M2). Each keeps the search baseline byte-identical.
Slice G (`docs/plans/2026-09-nystrom-downdate.md` §10, 2026-09-05) delivered, as `PrunedAStarSearch`, a
variant of the unchanged exact `AStarSearch`: the
goal-sibling filter; M1 in its exact form (incumbent pruning, `incumbent_seed` for the greedy residual trace,
`IncumbentBelowOptimum` when a seed is below the optimum, a 1e-9 relative slack plus a caller-stated absolute
`incumbent_slack` — 1e-12 of the trace — so a seed priced on another arithmetic path cannot prune the optimum by
rounding, even when the optimum is zero); M4 in-library (`max_frontier`, `FrontierLimitExceeded`,
`frontier_peak` on the instance). Measured on this machine at SPECTF scale 4: n = 60, k = 5 frontier peak
3,794,117 → 452,830; n = 80, k = 5 at 1,488,433 (the plan's before-estimate was 24 million); expansions and
landmarks unchanged everywhere. Remaining: T2 (specified research-side,
`~/develop/research/nystrom/harness/T2-gragg-secular-solve.md`, next); M2 (array-backed frontier, only if
EXP-09a's `frontier_peak` column demands it); M5 (dropping the expanded set behind a problem-level tree
declaration); the (k̄+1)·f Deshpande–Rademacher rule, only as D-27 allows (δ = 0, or through the untruncated
root bound). Then BL-30.
Re-ranked 2026-09-06 after EXP-09a (research `nystrom/FINDINGS.md` F-5, F-6): on decaying-spectrum kernels the
untruncated search certifies n = 1,000 in 29–45 min and δ = 1e-4 cuts that 3–4× with the same optimum, so T2 is the
next gain there; on flat-spectrum kernels the frontier fills 60 M entries (~7.4 GB) at ~200k expansions before the
clock matters, so memory binds and M2 (array-backed frontier) and the bounded variant precede T2 for those cells.
Exact certification of flat spectra at n = 1,000 is not an engineering target on this machine.
Harness pass-through (PR #34, 2026-09-06, D-28): the A* selector takes a `search_factory`, the UCI runner
takes a `selectors` list, and `frontier_peak` rides out on `SearchResult` into `UciHarnessResult.frontier_peaks`,
so M2 and any bounded variant can be measured on the harness's own cells beside the certified reference
instead of through a bespoke script. Defaults unchanged: both baselines byte-identical, no snapshot regenerated.
Anytime A\* with the a-posteriori gap (pass-2 entry ticket, 2026-09-07; research seed
`sessions/2026-09-28-entry-ticket-slice.md`): `AnytimeAStarSearch(PrunedAStarSearch)` adds `max_expansions` beside
the inherited `max_frontier` and turns both stops into a result — the incumbent, `optimal=False`, and the
certified gap `incumbent − frontier_min` — instead of `FrontierLimitExceeded`. Uncapped it is exact A\* on every
snapshot cell and both toy problems (same state, cost, expansions), so the F-6 cells that today end with nothing
report a bracket `[frontier_min, incumbent]`. The bounded variant BL-31 named above is this one; M2 and T2 remain.

### BL-34 — External-memory best-first search for the Nyström engine · `backlog-only` · large · reward R1

The F-6 frontier (60 M entries, 7.4 GB) is the binding resource on flat-spectrum kernels at
n = 1,000, and compute per expansion (~150 ms) exceeds the I/O to spill 300 entries (~36 KB) by
five orders of magnitude, which inverts the premise of the external-memory-search literature
(Edelkamp et al. External A\*; Korf 2004 DDD). Because states are canonical tuples the DAG is a
tree and *no delayed duplicate detection is needed*, removing that literature's hardest part. Open
design question: bucket granularity for a real-valued f (too wide expands above OPT; too narrow
means many tiny files). First diagnostic, no code: dump the in-memory frontier's f-histogram at
several points of an isolet5 run. Unlocks every X2.1 cell (A1, E1, F1, F2, F7 comparisons at the
field's scale). Re-gate when a frontier abstraction exists (BL-31 M2 is the same seam). Origin:
`research/candidates/_field-2.md` cell F5.

### BL-35 — Removal-direction search with an incremental removal bound · `backlog-only` · large · reward R3

Start from all n columns and remove (the Narendra–Fukunaga 1977 tree); prune layer k from both
sides. Prerequisite: the removal bound as a rank-one *downdate of the complement*, or the tree has
depth n − k ≈ 990 with near-full eigendecompositions at every node; plausible first at n ≤ 240
with k near n. The bidirectional B&B shape for subset selection with monotone matrix criteria is
Cao & Kariwala's (Comput. Chem. Eng. 2008–2010, min-singular-value criteria), so the claim is a
*direction selector by k/n on this bound*, never the shape. Unlocks W7 (safe column elimination)
as a by-product and the removal bound's use as a second bound (parked P7). Origin: cell G1.

### BL-36 — Column-cluster abstraction as an admissible hierarchical bound · `backlog-only` · large · reward R1

Cluster columns so every member is within ε of its representative; search over clusters with a
bound inflated by a Wedin/Davis–Kahan slack 2ε√k‖Y‖₂/σ_min(Y_S); refine within chosen clusters
(hierarchical A\*, Holte et al. 1996). The crux is admissibility, undetectable at the fp64 floor
by comparing against known optima, and flat-spectrum columns have no cluster structure. First
diagnostic, no engine change: cluster by kernel-column cosine on the S1 cells and check the
cluster bound at the root against the known optima. A projection-cost-preserving sketch gives a
stronger bound with no conditioning factor but touches pass-1 S25 (parked P8). Unlocks X2.5
(per-expansion cost) at n = 1,000 if the abstraction prunes. Origin: cell H1.

### BL-37 — SMA\* on the Nyström frontier (expected negative) · `backlog-only` · large · reward R1

Delete the worst leaf at the cap and back up its f. Costs: a double-ended heap and parent
pointers raise the per-entry size from ~120 B toward 200 B; regeneration re-prices a whole sibling
batch (110 ms) per forgotten child; on plateaus max-f and min-f leaves differ by less than the
rounding tolerance, so deletion thrashes. First diagnostic: simulate the deletion policy on a
logged frontier of one F-6 cell and count regeneration events. Worth a row because "SMA\*'s
regeneration accounting fails under batched, expensive evaluation" is a publishable negative and
closes the family. Unlocks nothing until BL-34 fails. Origin: cell F4.

### BL-40 — Harness the R620 for grid runs · `backlog-only` · re-entry 2–4 h

The Dell R620 (2 × E5-2670, 110 GiB usable, Proxmox VE 9.2 at 192.168.2.10 over a direct cable to the
Mac) is set up but carries no workload yet. Docs, credentials and the standing rules live in
`~/develop/r620/` (read `OPERATIONS.md` first; §3 lists the undecided points — topology, where work is
picked up, local LLM, backups). First step when this opens: create the `compute-01` guest and run one
n ≥ 80 cell of the Nyström grid there from a botMaker checkout, comparing wall time against the Mac.

### BL-41 — ncut generalization of the two-hot span modules · `backlog-only` · re-entry 4–6 h

The rcut prototype (`docs/plans/2026-09-two-hot-span.md` phase 2) is built for c = 1, C = I. The
generalization is c_i = √d_i, C = diag(c), X = D^{-1/2}E threaded as parameters of the same three
modules — the rounding condition becomes c_i v_i + c_j v_j = 0 and the rounded objective Ê becomes
NCut, which was verified to machine precision in the research session. The source makes it
conditional: "Only after this works reliably for rcut should the implementation be generalized."
Open this when slice 2.3's harness shows a single λ giving exactly K components on at least three of
the four **original** test graphs (`roach_g5`, `karate`, `planted_partition`, `two_moons_knn`);
`roach_g20` (slice 2.6, K = 3, He-Gu-Zhang 2012) is an added instance reported alongside and outside
that criterion. Parked on the research side as P23. Do not copy the modules; add parameters.

### BL-42 — torch on the R620: AVX2 requirement undocumented · `backlog-only` · re-entry 1–2 h

The Dell R620's E5-2670 CPUs predate AVX2. PyTorch does not document an instruction-set requirement
for its CPU wheels (pytorch.org/get-started/locally carries no AVX/AVX2/AVX-512 text as of
2026-09-09), and issue #94021 (https://github.com/pytorch/pytorch/issues/94021 · issue "Set AVX2 is
minimum supported instruction set for Linux X86", open, 2023-02-03 · retrieved 2026-09-09) shows some
ops already hard-fail without AVX2 with the "Your CPU does not support FBGEMM" message. First step
when this opens: install the `torch` group inside
`compute-01` and run slice 2.3's harness on the roach graph; if it fails, record the failing op and
decide between an AVX-less build and keeping these runs on the Mac. Blocks nothing today — the
prototype's runs are minutes on the Mac (BL-40 is the wider R620 harnessing item).

### BL-43 — Interactive walkthroughs from injected recorders · `backlog-only` · re-entry 6–10 h

Opened 2026-09-10 from a grilling session; work started 2026-09-10 (plan
`docs/plans/2026-09-visualization.md`, D-32). Neither of the repo's two engines can be watched: a
`SearchResult` says what was proved and nothing about how the frontier moved or when the incumbent
improved, and the two-hot optimizer's V is only ever seen at the last step. Observation enters as an
*injected recorder* — an algorithm takes `recorder: AbstractRecorder` defaulting to a fresh
`NullRecorder`, with one guarded call site at the moment its state advances, so a run with the default
is byte-identical to today's and nothing joins a result type (D-28). The abstract recorder lives in
`math/recorder.py`; the per-problem children, which do the extracting and the captioning, live in the
new `mllib.visualization` package beside the renderer that turns a run's frames into one offline HTML
page with a stepper (vanilla JS, inline SVG, no CDN).

Four slices, by branch: `feat/visualization-recorder` (recorder contract, A\* instrumentation, records,
process edits) → `feat/visualization-walkthrough` (versioned JSON recording, renderer, template and JS,
A\* view, example, fixture, CLI) → `feat/visualization-two-hot` (two-hot recorder and view; stacked on
`feat/two-hot-span-harness`) → `feat/visualization-graph-search` (BFS and DFS take the recorder,
`math/graph/visualizer.py` deleted with BL-14 amended, the networkx example switched to a recording
plus a page).

**Phase 2 (2026-09-10).** Explainability layer: plan § 6, D-34, branches
`feat/visualization-explain{,-two-hot,-traversal}`.

First step: `math/recorder.py` with `AbstractRecorder`, a frozen slotted `Frame` and `NullRecorder`,
then the single guarded call site in `AStarSearch._search`, with the variants' extras coming through a
`_recorder_extras()` hook — both baselines run before and after.

### BL-44 — Spectral-bound admissibility slack on near-duplicate points (hypothesis falsifying example) · `backlog-only` · re-entry 1–2 h

Found 2026-09-10 while running the suite in a fresh worktree: hypothesis produced a falsifying example for
`tests/math/graph/test_nystrom_landmark_problem.py::test_the_bound_never_exceeds_any_completion_it_bounds`
— points `[0.0, 1.0, 0.0625, 0.03125, 1e-05]`, k = 3 — where the lower bound is 9.9e-08 and the best
completion is exactly 0.0, so the bound overshoots by ~1e-7 against the test's 1e-9 slack. The kernel is
numerically rank-deficient (near-duplicate points), and the code under test
(`src/mllib/math/graph/nystrom_landmark_problem.py`) is untouched by any open slice. The property test is not
derandomized, so `main` can hit this on any run. First step when this opens: reproduce with the example
above, decide whether the slack should scale with the retained spectrum (D-26/D-27 territory) or the test
should exclude kernels below numeric rank k, and add `derandomize=True` so the suite's verdict is stable.
Nothing else depends on it; the two behavioural baselines are unaffected.

### BL-45 — Two-hot span stress ladder (rungs 0–3) · `in-progress` · re-entry 2–4 h

Opened 2026-09-10 from the research seed `~/develop/research/sessions/2026-09-18-two-hot-stress.md`; built as
slice 2.7 of `docs/plans/2026-09-two-hot-span.md` on `feat/two-hot-stress`. The prototype's one tuned cell
(roach G₅, λ = 10, ν = 10, μ = 0.3, Ê = 4/15) is run against a **pre-registered** ladder: rung 0 the
prototype's three graphs at three seeds, rung 1 the planted partition at (100, 2)/(201, 3)/(500, 5) × clear,
moderate and weak cross-block degree × three seeds, rung 2 the same generator at n = 1000 and 2000 (cloud
only, BL-42 keeps it off the R620), rung 3 polbooks / football / email-Eu-core — 592 cells in all, every one
recorded including the ones that reach fewer than K components, time out or exceed the memory budget.

The runner (`src/mllib/ml/projects/two_hot_span_stress.py`, CLI `examples/two_hot_span_stress.py`) is
**harness-tier**: it composes the engine and computes no mathematics of its own, and the three engine modules
(problem, optimizer, harness) are byte-identical to what slices 2.1–2.6 shipped. The rung-0 oracle is what
proves that — the prototype's frozen JSON reports are reproduced to 1e-8 before rungs 1–3 are allowed to run,
and rungs 1–3 refuse to start without a passing oracle in the same results file. Live results are gitignored
under `examples/two_hot_span_stress_results/`; the **frozen** result is not a botMaker artefact at all — it
lives in the research repo as `nystrom/data/two-hot-stress-v1.jsonl` with its sha256 and a `PROVENANCE.md`
row naming the botMaker commit, which every cell record also carries beside the three engine files' digests.

Closes when the ladder has run and the seed's three questions have numbers; the analysis, the findings and
the kill-criteria re-read belong to the research repo, not here.

### BL-46 — Two-hot: restarts and the joint (W, H) factorization as an optimizer variant · `backlog-only` · re-entry 2–4 h

Opened 2026-09-10 from the `two_triangles` side-by-side (slice 2.8, `docs/LEARNING_LOG.md` "λ is not
scale-free"). Two things the prototype does not do and Xavier's own minimal implementation of the same
objective does: **restarts** — best of ten random initialisations, selected by training loss — and a **joint
(W, H) factorization** in place of the single moved V. On the six-node triangles the two together are worth
5/10 restarts against this engine's 2/10 seeds, and selection by training loss happened to pick the best Ê
because the losing restarts sat at a column with R ≈ 0.43–0.47, visibly short of ½. Neither observation is a
decision: one instance, and the selection rule only coincides with Ê selection while the failure stays that
legible.

First step: pair the two formulations on roach G₅ over 10 seeds at λ scaled by Σλ/r, and report by Ê — never
by training loss, which is the two formulations' one incomparable quantity. Closes when the pairing says
whether restarts alone explain the gap or the joint factorization buys something restarts do not.

### BL-47 — Walkthrough pages cost bytes per step, not per full frame · `backlog-only` · re-entry 2–3 h

Opened 2026-09-10 from slice 2.8, where the `two_triangles` pages had to be rendered at 1500 steps rather
than the instance's own 5000 to clear the 1024 kB `check-added-large-files` limit. `--frame-every` thins
full frames only, so a page's size tracks the **step count**: at 5000 steps the page is 2.9 MB, of which
1.79 MB is 4799 light frames (≈ 374 B each, and roughly 230 B of that is the nine null-valued derived keys
a light frame writes out because `TwoHotSpanFrame.to_dict` keeps one dict shape for every frame) and
0.79 MB is the explain layer's one narration sentence per frame. Measured: dropping the null keys alone
leaves 1.84 MB, still over; a light frame every ten steps with the nulls dropped and narration thinned to
match lands near 460 KB at the full 5000.

First step: decide whether the one-shape-per-frame rule (`visualization/recorders/two_hot_span.py` module
docstring) is worth its cost, since the view already branches on `full`. Then a `light_every` on
`AbstractStepRecorder` and narration derived only for retained frames. Closes when a 5000-step
`two_triangles` page is committed under the limit and the roach G₅ fixture is regenerated deliberately.

### BL-49 — Descent stack breaks injection · `backlog-only` · re-entry 3–5 h

Opened 2026-09-11 from the D-35 survey; out of BL-48's scope by decision. The descent stack is the
code the injection stance was written about, and it breaks it in four places.
`ml/logistic_regression.py:31-43` constructs its own `HypothesisFunction`,
`PolynomialRegressionExpander` and `MSE` with a hardcoded seed of 10 instead of receiving them, and
`grid_fit` (`:45-74`) mutates `loss_function`, `epochs` and `learning_rate` on the instance and
rebuilds the hypothesis per permutation, the mutation D-35 (5) forbids. `math/loss_function.py` has
two gradient contracts: the regression losses take `compute_gradient(actual, predicted)` (`:37`,
`:50`) while `PerceptronLoss` and `HingeLoss` take a third `data_values` argument (`:62`, `:80`),
and the descent base calls the two-argument form (`ml/gradient_descent.py:60`).
`math/hypothesis.py:20` overwrites the injected expander's degree with its own. The old bases —
`LossFunction`, `CostFunction`, `RegularizationFunction`, `HypothesisExpander` — express
abstractness by `raise NotImplementedError` on plain classes, or not at all, rather than by
`abc.ABC` (D-35 (2)); `CostFunction` and `RegularizationFunction` are redefined and made abstract by
BL-48 slices 1 and 2, the other two are not.

Relation to BL-26: that entry is the loop duplication (Perceptron and SVM off the descent base) and
names the gradient-signature alignment as its enabling step; this entry is the injection break and
the base-class form, and the signature alignment is done once, here or there, never twice. The
`math/hypothesis.py:51` call to an `expand` signature no expander defines is not this entry: it was a
bug, fixed in BL-48 slice 5 (2026-09-11) with a test that fails on `main`.

First step: a fresh-objects `grid_fit` — a cell is a configuration plus a constructor per injected
role, built per cell, never mutated (D-35 (5)) — with the training baseline byte-identical before
and after. Closes when `MyLogisticRegression` receives its hypothesis and loss, one gradient
contract serves every loss, the four bases are `abc.ABC`, and `hypothesis.py:20` no longer mutates
what it was handed.

### BL-50 — Spectral-start fixtures reproduce only on the platform that wrote them · `backlog-only` · re-entry 2–4 h

Opened 2026-09-11 when BL-48 slice 1 made `ci-torch` run the whole suite: on the ubuntu runner five
torch tests failed that had never run in CI before — the stress rung-0 oracles
(`tests/ml/test_two_hot_span_stress.py`, roach_g5, karate and roach_g20 at `ORACLE_TOLERANCE = 1e-8`)
and the two-hot walkthrough recording (`tests/visualization/test_two_hot_span_example.py`, roach_g5
at 1e-9), all of them runs from the **spectral start**. The random-start refactor snapshot
(`tests/math/algorithms/two_hot_span_refactor_snapshot.json`) passed on the same runner at 1e-9.

The mechanism, measured on the Mac that wrote the fixtures: the spectral spanning set is the
optimum of the span term, so its gradient there is rounding noise, and 20 of the 360 entries of the
first gradient on roach G₅ (λ = 10) are below 1e-8. Adam's first step is `lr · g / (|g| + 1e-8)`, a
sign function of `g`, so those 20 entries jump by ±lr according to which side of zero the BLAS put
them. A 1e-14 perturbation of the start moves V by 0.094 after one step and Ê from 6.50 to 6.30;
the same perturbation of a random start moves V by 8e-15. OpenBLAS on the runner against
Accelerate on the Mac is that perturbation (b51d905 saw its last-digit form on the search
fixtures). The numbers are not wrong on either platform; they are two legitimate runs from one
start, and no tolerance separates them.

Until closed, the five tests carry `only_on_fixture_platform` (`Darwin-arm64`) and skip elsewhere
with this id in the reason, so `ci-torch` runs everything else. First step: decide whether the
spectral start stays a reported cell given that it is a stationary point of the span term and its
first step is decided by rounding — if it stays, the fixtures are written on the CI platform and
compared at a stated tolerance there, and the Mac is the platform that skips; if it goes, the
walkthrough and the rung-0 oracles move to the random start (BL-45 owns the ladder). Closes when
no torch test is skipped by platform. The same decision settles the Laplacian's two derivations
(ARCHITECTURE §5): the fixture set that is regenerated deliberately takes the other's arithmetic.

## Closed

### BL-48 — Optimizer object model: the two-hot span optimizer onto injected math objects · closed 2026-09-11 (this PR: the stacked branch `refactor/optimizer-object-model-5`, slices 2–5, on top of #43 and #44)

Opened 2026-09-11 from D-35. The two-hot span optimizer
(`math/algorithms/two_hot_span_optimizer.py`) is one module with no class hierarchy: thirteen knobs
in one frozen config, free functions for the training loss and its penalties, Adam and its schedule
built inside the function, per-step arrays on the result, and two stop conditions that raise. The
search stack next to it is four injected objects and a result contract, and a variant touches none
of them. This initiative gives the optimizer the search stack's shape: an `AbstractOptimizer` in
`math/algorithms` owning `run()` with a `_step` seam; a problem object holding X and the cluster
count and the exact reporting arithmetic; `CostFunction` redefined as a scalar of the parameters
being optimized, with the ridge cost as its first implementation (shipped as `SpanCost` over an
injected projector, slice 2); `RegularizationFunction` made
abstract with the collision, adjacency and diversity terms as its first implementations, each weight
a knob on its class; step rules and schedules as injected objects; a result per D-35 (9) —
parameters, final training loss, steps taken, stop reason, configuration, delivered numbers — with
per-step frames on the recorder, so the two-hot recorder, view and stress harness read them from
there.

Acceptance test: BL-46 (restarts, the joint factorization) and BL-41 (ncut) each slot in without
editing an existing class. Plan: `docs/plans/2026-09-optimizer-object-model.md`, six slices, the
first docs-only. Closes when slice 5 has landed with both baselines byte-identical, the two-hot
fixtures regenerated deliberately and stated in the PR, and the acceptance test written down as a
sentence per item in the plan's §5 that the reviewer-agent can check against BL-46's first slice
when it opens.

Progress: slice 5 (2026-09-11) — `HypothesisFunction.expand_hypothesis` recomputes the constructor's
expansion of the initial weights; a test fails on `main` and passes here. All six slices shipped;
BL-46 and BL-41 open against the seams the plan's §5 names; the Laplacian dedupe waits on BL-50.
Progress: slice 4 (2026-09-11) — `AbstractOptimizer` with `run()`/`_begin`/`_step`/`_iterate`/`_assemble`;
`TwoHotSpanOptimizer` in the package on a `TwoHotSpanProblem` (X, K, the exact arithmetic, the
constraint vector), taking its cost, penalties, step rule and recorder; `TwoHotSpanResult` per D-35 (9)
with `StopReason` in place of the two exceptions; `configuration` assembled through `describe`;
`compose_two_hot_span` as the composition roots' one home; the harnesses and the recorder read
histories off the recorder; `two_hot_span_optimizer.py` deleted; walkthrough and stress fixtures
regenerated with every number unchanged; refactor snapshot byte-identical. The Laplacian dedupe
stays open (ARCHITECTURE §5, with BL-50): the two derivations differ by ulps the spectral start
amplifies, and this slice regenerated no number.
Progress: slice 3 (2026-09-11) — step rules and schedules as injected objects (`AbstractStepRule`,
`AdamStepRule`; `AbstractLearningRateSchedule` with four members); `learning_rate_lambda` deleted;
refactor snapshot byte-identical, after catching a one-ulp reordering in the cosine schedule.
Progress: slice 2 (2026-09-11) — `AbstractCostFunction`, the projector as one concept with two
arithmetics (`ExactProjector`, `RidgeProjector`), `SpanCost`; refactor snapshot byte-identical;
Laplacian dedupe moved to slice 4.
Progress: slice 1 (2026-09-11) — penalties as injected objects; `AbstractRegularizationFunction`;
refactor snapshot `tests/math/algorithms/two_hot_span_refactor_snapshot.json` byte-identical.

### BL-39 — Conditional solves: forced and forbidden columns in the Nyström problem · closed 2026-09-07 (this PR; research candidate E3 necessity margins, seed `sessions/2026-09-10-engine-slices.md`)

Necessity margins ask, for every column j, the cost of the best subset that must contain j and of the
best that must not: 2n conditional solves, each the same certified search on a modified ground set.
Closed by two opt-in knobs on `NystromLandmarkProblem`, `forced` and `forbidden`, with `landmark_count`
unchanged: the search starts at the forced columns and successors add one free column above the largest
free column already chosen, so every admissible subset is one canonical node. The bound is untouched
(removing candidates only raises the optimum; starting deeper is a subtree of the same tree); the
batched pricing now finds the added column wherever it sorts, so a forced column above it no longer
sends children to the per-child oracle. `constraints` states the knobs and the UCI harness records them
as `problem_constraints` (empty by default, printed only when set). The research driver
`nystrom/grid/run_margins.py` and the k = 4 Krause smoke result live research-side.

### BL-38 — First-in-first-out tie-breaking enumerates the plateau on nearly rank-k data · closed 2026-09-07 (this PR; research candidate C9, seed `sessions/2026-09-10-engine-slices.md`)

On nearly rank-k kernels (the S1 plateau cells, the reproduction's 2^k regime) many frontier entries
share a bound to within rounding, and the heap key `(bound, insertion_index)` works the plateau level
by level, so the search enumerates it before a goal pops — a property the reproduction recorded as
A\*'s. It is the tie-break's: preferring the deeper state reaches and certifies a goal after about k
expansions. Closed by two opt-in knobs on `AStarSearch`, inherited unchanged by the pruned and anytime
engines and stated by `configuration`: `tie_break="deepest"` (key `-len(state)` after the bound) and
`tie_tolerance` (absolute, trace-scaled like `incumbent_slack`; the bound key is quantised to that grid
so near-ties compare equal). Under a tolerance the popped goal is certified only when its bound is at or
below every remaining raw bound; otherwise `optimal=False` with the honest additive gap on
`certified_gap`, below the tolerance by construction (D-29). Defaults byte-identical on every snapshot
cell; on the plateau fixtures deeper-first expands k + 1 states where first-in-first-out expands the
plateau. The S1 experiment (ties versus bandwidth) is research-side.

### BL-33 — Is the spectral bound monotone along the tree? · closed 2026-09-07 (this PR; research session seed `sessions/2026-09-18-monotonicity-check.md`, run early)

A* with a terminal-only objective needs only admissibility (D-23), so nothing in the engine had ever
checked whether a child's bound can fall below its parent's — which decides how the anytime gap
(incumbent minus frontier minimum) behaves mid-run and whether root tightness predicts pruning, the
question the research side's parked idea P2 waits on. Closed by an opt-in counter on `AStarSearch`
(`count_bound_drops`, `bound_drop_slack`; `BoundDropCounter` left on the instance as `bound_drops`,
stated by `configuration`), off by default and with no effect on any expansion or result, and by a
grid run over the S1 cells whose result file and finding live research-side
(`nystrom/data/monotonicity-S1-v1.jsonl`, `method/13-search-landscape.md`). The bound, the objective and
the search order were not touched: a non-monotone bound is a finding, not a bug.

### BL-32 — A harness row does not say which engine settings produced it · closed 2026-09-06 (PR #35, D-28)

D-28 made the engine a parameter, but a `search_factory` binds its knobs inside the caller's lambda, so
`UciHarnessResult` recorded that `astar-pruned` ran and not with what `incumbent_seed`, `incumbent_slack`
or `max_frontier` — while the outside-git grid runner records all three on every line. Closed by having
the engine describe itself (`AStarSearch.configuration`), read off the instance by the selector into
`SearchResult.engine_configuration` and `UciHarnessResult.engine_configurations`.

### BL-27 — Nyström A* lower bound is O(n³) per child · closed 2026-09-04 (`docs/plans/2026-09-nystrom-downdate.md`)

`NystromCssCostFunction.lower_bound` recomputes an SVD of the selected columns and an n×n `eigvalsh` of
the deflated residual for every generated child. Correct (A* matches brute force on every bundled
instance), but it caps the search at n ≈ 40. The AAAI-15 machinery — one root eigendecomposition, then a
rank-one downdate of the parent's spectrum per child via the secular equation — makes each child O(k·r),
and the `research/repro/astar-css` reimplementation already does it. Port it before any run at n ≥ 500
(the `nystrom-certified-landmarks` first experiment). Marked at the bound itself in
`src/mllib/math/graph/nystrom_landmark_problem.py`. Since slice 6.1 the natural home is
`NystromCssCostFunction.lower_bounds`, which already decomposes the parent once for goal-depth children
(D-24); the downdate extends the same method to every depth.
**Closed.** `NystromCssCostFunction.lower_bounds` prices every child above goal depth from one
eigendecomposition of the parent and a rank-one downdate per child through `math/secular_equation.py`;
the per-child `lower_bound` is kept as the oracle. Search baseline and example output unmoved.

### BL-29 — Nyström bound on a truncated spectrum · closed 2026-09-04 (D-26; `docs/plans/2026-09-nystrom-downdate.md`)

For a full-rank kernel the per-parent eigendecomposition that BL-27 leaves behind is still n³, which
is the bottleneck at n ≥ 1,000 (EXP-09). Compute the bound on the kernel with its smallest eigenvalues
dropped, the retained rank chosen by a dropped-mass tolerance δ (`spectrum_mass_tolerance`, default 0).
The bound stays admissible because a Schur complement is monotone on the PSD cone, so no correction
term is added; goal costs and the goal-depth batch stay exact on the full kernel. Ships with D-26, the
proof in the learning log, and the admissibility tests of the plan's FR-7. Blocked on BL-27.
**Closed.** `NystromLandmarkProblem(..., spectrum_mass_tolerance=δ)`; the oracle and the fast path both
see the truncated kernel, goal costs stay exact, admissibility and optimum membership tested at four
tolerances. The threshold unification it leaves behind is BL-30.

### BL-09 — Declarative `TransformerPipeline` · closed in slice 6.3 (declarative `TransformerPipeline` from JSON config; `DataTransformer`, the `temp_*` methods and the model-name ladder deleted)
`MlLib/data/data_orchestrator.py:41,53` — the pipeline class is written but commented out ("finish above pipeline arch when
time allows"). Target: transformations declared in `ProjectSpecificDataClasses/*.json`, loaded by class name; retire the
`temp*Transformer` methods and the `if model == …` ladder in `get_transformed_data`.

### BL-23 — Classifier predicts all-ones on ad-click data · closed in slice 5.3b
**Findings (2026-08-29).** Two causes, one fixable. (1) *Model bug:* the sign hypothesis emits {-1, +1} but was
trained against {0, 1} targets, so the MSE gradient compared mismatched label spaces; on a linearly separable
synthetic set it stalled at 0.42 accuracy, and with targets encoded to ±1 it reaches 1.0. Fixed by
`GradientDescentModel.encode_targets` (identity) overridden in `MyLogisticRegression` (→ ±1); the unexplained
pre-training "descent step with the initial weights as gradient" in `grid_fit` had no measurable effect and was
removed. (2) *Data:* the ad-click features carry no linear signal — sklearn `LogisticRegression` scores 0.650
(5-fold CV) against a 0.650 majority rate; gradient boosting reaches 0.715. So ≈0.66 is the ceiling for this model
family on this dataset, and the smoke-grid baseline (10 epochs, lr 0.01) legitimately stays at the majority rate.
Guard: `tests/ml/test_logistic_regression.py::test_learns_a_linearly_separable_problem_from_zero_one_labels`.

### BL-12 — `hypothesis.py:43` call expander to reshape data · closed in slice 5.3 (base `HypothesisExpander` is the identity; the descent base applies it uniformly)
Folds into the shared gradient-descent base.

### BL-20 — Logistic/linear duplication · closed in slice 5.3 (`GradientDescentModel` base; linear/logistic are thin subclasses; split moved into `grid_fit`)
`MlLib/ml/logistic_regression.py:10,36`. Only real difference is `compute_prediction` vs `compute_classification`, already
polymorphic on `HypothesisFunction`.

### BL-10 — Regression/classification task enum · closed in slice 5.2 (`TaskKind` enum; `task_kind` on loss/cost/regularization; surfaced by `describe()`)
`cost_function.py:7`, `loss_function.py:9`, `regularization_function.py:4` all want an enum "later". One `TaskKind` enum.

### BL-16 — Auto-generated metadata markers · closed in slice 5.1 (46 dicts → class docstrings; `mllib.describe.describe()` introspects name/doc/signature)
~25 × `# TODO: review metadata (auto-generated)` plus the hand-typed `metadata = {…}` dicts they annotate. Replaced by an
introspection helper.

### BL-25 — Depth-first search is a stub · closed in slice 3.6 (iterative stack DFS with visited set and max_depth; 5 tests)
`MlLib/mathDomain/algorithmImplementations/depthFirstSearch.py` validates inputs then returns `[]`. Found in
slice 3.1 when its import root was fixed and the algorithm modules became importable. Implement iteratively
(stack, visited set, `max_depth`) mirroring the BFS shape; the BFS tests are the template. Until then no
smoke test can pass for it, so slice 3.6 implements rather than merely tests it.

### BL-18 — `graphBased/tests/nxGraphExample.py` · closed in slice 3.5 (moved to examples/graph_search_vs_networkx.py)
Not a test; a worked networkx example.

### BL-17 — Project scripts · closed in slice 3.5 (moved to examples/ad_click_model_comparison.py and boston_housing_vs_sklearn.py; sandbox.py and the import-only KNNClassifierTestScript.py deleted; MlLib/run_all_tests.py deleted — pytest config lives in pyproject)
`projectScripts/AdClickModelProjectBuildScript.py`, `testScript.py` (Boston housing vs sklearn comparison),
`KNNClassifierTestScript.py` (imports only). `sandbox.py` deleted as scratch.

### BL-15 — Duplicate DFS · closed in slice 3.5 (deleted)
`MlLib/mathDomain/graphBased/searchAlgorithms/DFS.py` (13 lines) superseded by `algorithmImplementations/depthFirstSearch.py`.

### BL-06 — AI domain · closed in slice 3.5 (deleted)
`MlLib/aiDomain/` held only an empty `__init__.py`. Intent: agent/LLM-driven model selection and explanation on top of the
evaluator records. No design existed.

### BL-05 — `algorithmImplementations/testScript.py` · closed in slice 3.5 (deleted)
Empty scratch file.

### BL-04 — Iterative deepening search · closed in slice 3.5 (deleted)
`MlLib/mathDomain/algorithmImplementations/iterativeDeepening.py` was 0 lines. Pattern to follow: `depthFirstSearch.py` on
`abstractGraphAlgorithm.AbstractGraphAlgorithm` (`run() → _search() → _notify_evaluator()` with a frozen `SearchContext`).

### BL-03 — CSV source controller · closed in slice 3.5 (deleted)
`MlLib/dataDomain/sourceControllers/CsvController.py` was 0 lines. Intent: abstract `DataOrchestrator.load_data` behind a
source-controller interface (csv / dataframe / …). Folds into BL-09.

### BL-02 — Neural network · closed in slice 3.5 (deleted)
`MlLib/mlDomain/neuralNetwork.py` was a 0-line file. Intent: a hand-built MLP as the third model in the ad-click comparison
(commented-out call in `AdClickModelProjectBuildScript.buildModels`). Candidate CS 6344 topic.

### BL-22 — Confusion-matrix FP/FN swapped · closed in slice 3.4c (one vectorised base implementation, sklearn-checked; snapshot regenerated: precision 1.0→0.6585, recall 0.6585→1.0)
`genericEvaluator.py:141,145`: "falsePositives" counts target=1/pred=0 (a false negative) and vice versa; reported precision is
really recall. Regenerate `baseline_snapshot.json` deliberately in the fixing PR.

### BL-21 — pandas ≥ 3 read-only arrays · closed in slice 3.4b (evaluator copies arrays on entry; `pandas<3` pin lifted)
`MlLib/mlDomain/modelEvaluators/genericEvaluator.py:136-137` mutates copy-on-write views in place →
`ValueError: assignment destination is read-only`. Baseline test pins `pandas<3` until fixed.

_None yet._
