# MlLib Architecture Review — 2026-07-18

Status: review of record, no code changed. Findings F1–F6 are observations;
refactors R1–R3 are recommendations parked until explicitly scheduled. The
governing doctrine lives in the tradePlatform repo
(`~/Desktop/Develop/tradePlatform/docs/PHILOSOPHY.md`) — this library is a
planned future plugin source for that platform, and this review reads it
through that lens.

## The decomposition (the part that works)

Knowledge is sorted by *kind*, not by feature, with a strictly one-way
dependency flow — `projectScripts → mlDomain → mathDomain`, `dataDomain`
feeding the top. Math never imports ML; ML composes math.

| Layer | Contents | Role |
|---|---|---|
| `mathDomain` | HypothesisFunction, LossFunction (MSE/MAE/Perceptron/Hinge), CostFunction, HypothesisExpander, SplitFunction/Gini, graph & probability primitives | academic ideas as classes |
| `mlDomain` | linear/logistic regression, DecisionTree, SVM, perceptron, NN, KNN, evaluators | models composed from primitives |
| `dataDomain` | DataOrchestrator, DataTransformer, CsvController | data pipeline |
| `projectScripts` | AdClickModelProjectBuildScript etc. | composition roots / experiments |

The idea→class mapping is deliberately literal: the hypothesis h(x)=w·x+b is
`HypothesisFunction`; the feature map Φ is `HypothesisExpander` (polynomial
basis expansion as an injectable object — linear vs polynomial regression is
*which expander you inject*, not a different model); per-sample loss vs
dataset cost mirror the lecture distinction; each loss carries its own
gradient. `SplitFunction → GiniImpurity` with InformationGain/ChiSquare
stubbed shows taxonomy-as-hierarchy even before implementations exist.

**The spine is strategy injection, not inheritance**: models are hosts for
injected math objects, and the training loop is written once against the
`LossFunction`/`HypothesisFunction` interfaces. Two-tier specialization
(library base → thin project subclass carrying only `numWeights` + a
hyperparameter grid) is the same library/plugin split tradePlatform uses for
strategies. `AbstractGraphAlgorithm` (ABC + template method `run()` →
`_search()` → `_notify_evaluator()`, frozen-dataclass `SearchContext`) is the
most mature expression — a later, more confident pass than the ML files.

**Pipeline trace (ad-click project)**: builder constructs
`DataOrchestrator(csv, 'csv', …)` → transformer builds one dataframe per
model shape → `build_test_train_split(model)` → project subclass sets grid →
`gridFit` sweeps ParameterGrid (fresh HypothesisFunction per combo, epochs of
gradient descent) → evaluator reports best accuracy/precision/recall. Reads
top-to-bottom; the academic idea flows to a reported experiment winner.

## Findings

- **F1 — DataTransformer is a hardcoded god-class.** Per-project
  `temp*Transformer` methods; `get_transformed_data` is an
  `if model == 'logisticReg' … elif …` ladder
  (`dataDomain/DataOrchestrator.py:273`). The intended design — a declarative
  `TransformerPipeline` driven by the JSON configs in
  `ProjectSpecificDataClasses/` — is present but commented out under
  "just get it done" notes. Classic snapshot drift under deadline pressure.
- **F2 — Linear/logistic duplication**, already flagged in-code. The only
  real difference is `computePrediction` vs `computeClassification`, which is
  *already* polymorphic on `HypothesisFunction` — a shared gradient-descent
  base would collapse both.
- **F3 — Hand-typed `metadata` dicts are drifting.** `CostFunction` and
  `RegularizationFunction` both self-describe as "loss function parent
  class" (copy-paste); most classes carry `# TODO: review metadata
  (auto-generated)`; placement is inconsistently class-level (introspectable)
  vs instance-level (requires instantiation). This is a nascent
  capability-descriptor system built by hand — and the drift is the live
  argument for deriving descriptors by introspection instead.
- **F4 — Fragile subclass init.** Project `LogisticRegression.__init__`
  skips `super().__init__()`, so `learningModel`/`lossFunction`/`epochs` are
  unset until `gridFit` overwrites them; calling `.fit()` directly breaks.
- **F5 — Latent (shadowed) bug.** Base
  `ModelEvaluator.persistEvaluationRecord` builds a **set literal**, not a
  dict (`mlDomain/modelEvaluators/genericEvaluator.py:44`), and would raise
  on the unhashable metadata dict — currently harmless because both concrete
  evaluators override it, but the base template is a trap.
- **F6 — The web layer is a stub.** `fastapi_app/services.py` returns a
  hardcoded StrategyBacktester payload; `aiDomain` is empty. "Pipeline to
  app" today means: scripts run the library; the API is a separate contract
  mock. That contract's shape is the future join point to tradePlatform.

## Recommended refactors (not scheduled)

- **R1** — Finish the config-driven `TransformerPipeline`: transformations
  declared in the JSON configs, loaded by class name; retire the
  `temp*Transformer` methods and the model-name ladder.
- **R2** — Shared gradient-descent base parameterized by the hypothesis's
  predict method; linear/logistic become thin specializations.
- **R3** — Derive `metadata` by introspection (class name, docstring,
  signature) instead of hand-typing; fix F4/F5 in passing (proper
  `super().__init__()` chains; dict-literal or `NotImplementedError` base
  persist).

## Cross-codebase note

MlLib independently evolved the same architectural instincts as
tradePlatform (strategy injection, two-tier specialization, self-describing
metadata) *and* the same failure modes (hardcoded enumerations, declared-
intent-left-stubbed, descriptor drift) — with no web UI involved. That
corroborates the world-model doctrine as a statement about knowledge
distribution, not about frontends. When MlLib models plug into tradePlatform
as strategy plugins, the seam inherits the doctrine: descriptors introspected
at the boundary, so external models self-describe on arrival.
