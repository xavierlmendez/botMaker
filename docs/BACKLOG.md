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

### BL-27 — Nyström A* lower bound is O(n³) per child · `kept` · re-entry 1–2 days

`NystromCssCostFunction.lower_bound` recomputes an SVD of the selected columns and an n×n `eigvalsh` of
the deflated residual for every generated child. Correct (A* matches brute force on every bundled
instance), but it caps the search at n ≈ 40. The AAAI-15 machinery — one root eigendecomposition, then a
rank-one downdate of the parent's spectrum per child via the secular equation — makes each child O(k·r),
and the `research/repro/astar-css` reimplementation already does it. Port it before any run at n ≥ 500
(the `nystrom-certified-landmarks` first experiment). Marked at the bound itself in
`src/mllib/math/graph/nystrom_landmark_problem.py`. Since slice 6.1 the natural home is
`NystromCssCostFunction.lower_bounds`, which already decomposes the parent once for goal-depth children
(D-24); the downdate extends the same method to every depth.

### BL-28 — Nyström first-experiment grid · `backlog-only` · re-entry 1 day after BL-27

The harness measures the two gaps on three 40-row slices at k = 3, which is a smoke test, not
evidence. The `nystrom-certified-landmarks` first experiment needs: a Laplacian kernel beside the RBF
one, a ridge-leverage-score baseline, n = 500–3000, k up to 10, five bandwidths per kernel, and
RPCholesky averaged over ten draws. None of it is built and no interface here is pre-shaped for it.
It is blocked on BL-27 regardless, since the bound is O(n³) per child, and until then the experiment
runs on `research/repro/astar-css` with Y = K^{1/2}.

### BL-29 — Nyström bound on a truncated spectrum · `backlog-only` · re-entry slice D of `docs/plans/2026-09-nystrom-downdate.md`

For a full-rank kernel the per-parent eigendecomposition that BL-27 leaves behind is still n³, which
is the bottleneck at n ≥ 1,000 (EXP-09). Compute the bound on the kernel with its smallest eigenvalues
dropped, the retained rank chosen by a dropped-mass tolerance δ (`spectrum_mass_tolerance`, default 0).
The bound stays admissible because a Schur complement is monotone on the PSD cone, so no correction
term is added; goal costs and the goal-depth batch stay exact on the full kernel. Ships with D-26, the
proof in the learning log, and the admissibility tests of the plan's FR-7. Blocked on BL-27.

## Closed

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
