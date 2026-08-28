# Backlog — stripped initiatives and code-level TODOs

Registry of everything that was removed from the tree, or left in with a `TODO(BL-nn)` tag, under the strip rule in
`docs/REFACTOR_PLAN.md` (D-4). Phase-level milestones live in the orchestrator repo (`projects/botmaker.md`); this file is the
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

### BL-02 — Neural network · `deleted` · re-entry 0
`MlLib/mlDomain/neuralNetwork.py` was a 0-line file. Intent: a hand-built MLP as the third model in the ad-click comparison
(commented-out call in `AdClickModelProjectBuildScript.buildModels`). Candidate CS 6344 topic.

### BL-03 — CSV source controller · `deleted` · re-entry 0
`MlLib/dataDomain/sourceControllers/CsvController.py` was 0 lines. Intent: abstract `DataOrchestrator.load_data` behind a
source-controller interface (csv / dataframe / …). Folds into BL-09.

### BL-04 — Iterative deepening search · `deleted` · re-entry 30 min
`MlLib/mathDomain/algorithmImplementations/iterativeDeepening.py` was 0 lines. Pattern to follow: `depthFirstSearch.py` on
`abstractGraphAlgorithm.AbstractGraphAlgorithm` (`run() → _search() → _notify_evaluator()` with a frozen `SearchContext`).

### BL-05 — `algorithmImplementations/testScript.py` · `deleted` · re-entry 0
Empty scratch file.

### BL-06 — AI domain · `deleted` · re-entry 0
`MlLib/aiDomain/` held only an empty `__init__.py`. Intent: agent/LLM-driven model selection and explanation on top of the
evaluator records. No design existed.

### BL-07 — Sum rule / product rule · `kept` · CS 6344 pairing
`MlLib/mathDomain/probabilityBased/sumRule.py`, `productRule.py` — `pass` bodies, but imported by `test_probabilityBased.py`.
Implement alongside `bayesRule.py` / `prior.py` when probability is covered.

### BL-08 — InformationGain / ChiSquare split criteria · `kept`
`MlLib/mathDomain/graphBased/splitFunction.py` — placeholder subclasses beside a working `Gini`. Removing cleanly touches the
`SplitFunction` hierarchy and `DecisionTree` injection (> 1 h). Implement when decision trees are revisited (BL-13).

### BL-09 — Declarative `TransformerPipeline` · `kept` · Phase 6 (R1)
`MlLib/dataDomain/DataOrchestrator.py:41,53` — the pipeline class is written but commented out ("finish above pipeline arch when
time allows"). Target: transformations declared in `ProjectSpecificDataClasses/*.json`, loaded by class name; retire the
`temp*Transformer` methods and the `if model == …` ladder in `get_transformed_data`.

### BL-10 — Regression/classification task enum · `kept` · Phase 5 (R3)
`costFunction.py:7`, `lossFunction.py:9`, `regularizationFunction.py:4` all want an enum "later". One `TaskKind` enum.

### BL-11 — Trained-model exporter · `kept` · re-entry 1–2 h
`MlLib/mlDomain/projectSpecificFiles/adClickPredictionLogReg.py:29,52` — `self.exporter = None`. Intent: persist fitted weights
+ hypothesis config + evaluator record so a trained model can be reloaded without re-running the grid.

### BL-12 — `hypothesis.py:43` call expander to reshape data · `kept` · Phase 5 (R2)
Folds into the shared gradient-descent base.

### BL-13 — Decision tree debt · `kept`
`MlLib/mlDomain/decisionTree.py:23` refactor onto `graphBased` utilities · `:115` missing no-split error handling (fixed in
slice 3.4) · `:133` "abstracted later".

### BL-14 — Never-imported modules · `kept` + smoke tests (slice 3.6)
`graphBased/visualizer.py`, `probabilityBased/bayesRule.py`, `probabilityBased/gaussianPrior.py`,
`mathDomain/regularizationFunction.py`, `mathDomain/linearAlgebraHelpers.py`.
Real implementations (8–34 lines); kept by owner decision, each gets one smoke test so the keep rule holds.
*Correction 2026-08-28:* `depthFirstSearch.py` was listed here but is a stub (returns an empty traversal) — see BL-25.

### BL-15 — Duplicate DFS · `deleted` · re-entry 0
`MlLib/mathDomain/graphBased/searchAlgorithms/DFS.py` (13 lines) superseded by `algorithmImplementations/depthFirstSearch.py`.

### BL-16 — Auto-generated metadata markers · `deleted` · Phase 5 (R3)
~25 × `# TODO: review metadata (auto-generated)` plus the hand-typed `metadata = {…}` dicts they annotate. Replaced by an
introspection helper.

### BL-17 — Project scripts · `moved` → `examples/` (slice 3.5)
`projectScripts/AdClickModelProjectBuildScript.py`, `testScript.py` (Boston housing vs sklearn comparison),
`KNNClassifierTestScript.py` (imports only). `sandbox.py` deleted as scratch.

### BL-18 — `graphBased/tests/nxGraphExample.py` · `moved` → `examples/`
Not a test; a worked networkx example.

### BL-19 — tradePlatform plugin seam · `backlog-only`
North star (D-2): MlLib models plug into tradePlatform as strategy plugins with descriptors introspected at the boundary
(`docs/reviews/2026-07-18-architecture-review.md` §Cross-codebase). Prerequisites: BL-16 (introspected metadata), BL-11 (exporter).

### BL-20 — Logistic/linear duplication · `kept` · Phase 5 (R2)
`MlLib/mlDomain/logisticRegression.py:10,36`. Only real difference is `computePrediction` vs `computeClassification`, already
polymorphic on `HypothesisFunction`.

### BL-21 — pandas ≥ 3 read-only arrays · `fix` · slice 3.4b
`MlLib/mlDomain/modelEvaluators/genericEvaluator.py:136-137` mutates copy-on-write views in place →
`ValueError: assignment destination is read-only`. Baseline test pins `pandas<3` until fixed.

### BL-22 — Confusion-matrix FP/FN swapped · `fix` · slice 3.4c
`genericEvaluator.py:141,145`: "falsePositives" counts target=1/pred=0 (a false negative) and vice versa; reported precision is
really recall. Regenerate `baseline_snapshot.json` deliberately in the fixing PR.

### BL-23 — Classifier predicts all-ones on ad-click data · `fix` · slice 5.3b
Every smoke-grid permutation (and the historic example output) yields TP=1317, TN=0 → accuracy = majority rate 0.6585.
Suspects: classification threshold in `HypothesisFunction.computeClassification`, unscaled features, gradient sign.

### BL-24 — Lint debt behind temporary per-file ignores · `fix` · slices 3.5–3.7, 4.2–4.3, 5.1
`pyproject.toml [tool.ruff.lint.per-file-ignores] "MlLib/**"` lists rules the pre-refactor code violates so
the ruff gate can be on from slice 2.2. Counts at 2026-08-28 after safe auto-fixes and formatting:
E501 ×130 (long comments/strings) · TD004 ×17 (→ 3.7) · RUF012 ×12 (metadata dicts → 5.1) · F841 ×9 ·
RUF059 ×6 · B007 ×5 · E402 ×4 · E711 ×3 · RUF002 ×2 · SIM113 ×1 · B905 ×1 (→ 3.5/3.6) · N-rules (→ 4.2/4.3).
Rule: a slice that touches a file fixes that file's ignored violations; a code leaves the list when its
count reaches zero. Nothing is added to the list without a backlog entry.

### BL-25 — Depth-first search is a stub · `fix` · slice 3.6
`MlLib/mathDomain/algorithmImplementations/depthFirstSearch.py` validates inputs then returns `[]`. Found in
slice 3.1 when its import root was fixed and the algorithm modules became importable. Implement iteratively
(stack, visited set, `max_depth`) mirroring the BFS shape; the BFS tests are the template. Until then no
smoke test can pass for it, so slice 3.6 implements rather than merely tests it.

## Closed

_None yet._
