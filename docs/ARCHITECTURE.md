# MlLib Architecture

Living document. Promoted from `docs/reviews/2026-07-18-architecture-review.md` on 2026-08-28; the
review stays as the dated record, this file tracks the current and target shape. Update it in the same
PR as any change to a layer boundary, a base-class contract, or an extension point.

## 1. Decomposition

Knowledge is sorted by *kind*, not by feature, with a strictly one-way dependency flow:

```
projectScripts ──▶ mlDomain ──▶ mathDomain
       ▲
   dataDomain
```

Math never imports ML; ML composes math; scripts (composition roots) wire data to models.

| Layer | Contents | Role |
|---|---|---|
| `mathDomain` | `HypothesisFunction`, `HypothesisExpander`, `LossFunction` (MSE, MAE, Perceptron, Hinge), `CostFunction`, `RegularizationFunction`, `graphBased/` (graph, tree, `SplitFunction`/Gini), `algorithmImplementations/` (BFS, DFS on an ABC), `probabilityBased/` | academic ideas as classes |
| `mlDomain` | `MyLinearRegression`, `MyLogisticRegression`, `MyPerceptron`, `MySVM`, `DecisionTree`, `ProbabilisticKNN`, `modelEvaluators/`, `projectSpecificFiles/` | models composed from primitives |
| `dataDomain` | `DataOrchestrator` (+ `DataTransformer`), datasets, transformer JSON configs | load → transform → split |
| `projectScripts` | `AdClickModelProjectBuildScript` etc. | composition roots / experiments (→ `examples/` in slice 3.5) |

The idea→class mapping is deliberately literal: h(x)=w·x+b is `HypothesisFunction`; the feature map Φ
is `HypothesisExpander`; per-sample loss vs dataset cost mirror the lecture distinction; each loss
carries its own gradient.

## 2. Contracts (what a new component must satisfy)

**Injected math objects.** A model is a host for injected objects and is written once against these
interfaces. Adding a loss or an expander never touches a model.

| Interface | Must provide | Used by |
|---|---|---|
| `LossFunction` | `computeLoss(actual, predicted)`, `computeGradient(actual, predicted[, dataValues])`; margin losses also `computeBias` | every gradient-descent model |
| `HypothesisExpander` | `expandHypothesis(weights)`, `fitDataToHypothesis(data)` | `HypothesisFunction` |
| `HypothesisFunction` | `computePrediction(x)`, `computeClassification(x)`; owns weights, bias, expander | every model |
| `SplitFunction` | impurity of a candidate split | `DecisionTree` |
| `ModelEvaluator` | `updateTestingPredictionData(...)`, `evaluateModel()`, `persistEvaluationRecord()` | every model's `evaluate` |
| `AbstractGraphAlgorithm` | `_search(ctx)`; the ABC owns `run()` and `_notify_evaluator()`; `SearchContext` is frozen | graph algorithms |

**Two-tier specialisation.** A library base (`MyLogisticRegression`) plus a thin project subclass that
carries only `numWeights` and a hyper-parameter grid (`projectSpecificFiles/adClickPredictionLogReg.py`).
The subclass must call `super().__init__()` (F4/slice 3.3) — a project class that skips it is a bug.

**Evaluation records.** `evaluator.evaluationRecord[iteration]` is a JSON-serialisable dict with
`modelData`, confusion-matrix counts, `accuracy`, `precision`, `recall`. The training baseline test
depends on this shape; changing it means regenerating the snapshot deliberately.

## 3. Pipeline trace (ad-click project)

`DataOrchestrator(csv, 'csv', …)` → transformer builds one dataframe per model shape →
`build_test_train_split(model)` → project subclass sets grid → `gridFit` sweeps `ParameterGrid`
(fresh `HypothesisFunction` per combination, `epoch` rounds of gradient descent) → evaluator records
each permutation → `printEvaluation` reports the best. The smoke version of this trace is
`tests/mlDomain/test_training_baseline.py`.

## 4. Extension points — how to add things

- **A loss:** subclass `LossFunction` in `mathDomain/lossFunction.py`; implement loss + gradient; add a
  unit test that checks the gradient numerically; add a learning-log entry.
- **A model:** compose from `HypothesisFunction` + a loss; expose `fit`, `predict`, `predictValues`,
  `evaluate`; give it an evaluator; add it to the smoke grid only if it is part of a project comparison.
- **A graph algorithm:** subclass `AbstractGraphAlgorithm`; implement `_search`; never override `run`.
- **A transformer (after R1):** subclass the transformer ABC; declare it by class name in the project's
  JSON config.

## 5. Known structural debt and its schedule

| Debt | Where | Resolution |
|---|---|---|
| Hardcoded model-name ladder; `TransformerPipeline` commented out (F1) | `DataOrchestrator.get_transformed_data` | Phase 6 (R1, BL-09) |
| Linear/logistic duplication (F2) | `linearRegression.py`, `logisticRegression.py` | Phase 5 (R2, BL-20) |
| Hand-typed drifting `metadata` dicts (F3) | ~25 classes | Phase 5 (R3, BL-16) |
| Subclass skips `super().__init__()` (F4) | `adClickPredictionLogReg.py` | slice 3.3 |
| Base `persistEvaluationRecord` builds a set literal (F5) | `genericEvaluator.py:44` | slice 3.2 |
| Web layer was a stub (F6) | `fastapi_app/` | removed, BL-01 |
| Copy-on-write incompatibility, swapped FP/FN, degenerate classifier | `genericEvaluator.py`, training loop | BL-21, BL-22, BL-23 |
| `camelCase` modules and methods; nested `tests/` dirs; two import roots | everywhere | Phase 4 (D-17, D-18) |

## 6. Target layout (after Phase 4)

```
pyproject.toml  uv.lock  CLAUDE.md  README.md  CONTRIBUTING.md
src/mllib/
  math/      hypothesis.py  hypothesis_expander.py  loss_function.py  cost_function.py  …
             graph/  algorithms/  probability/
  ml/        linear_regression.py  logistic_regression.py  …  evaluators/  projects/
  data/      orchestrator.py  transformers/
tests/       mirrors src/mllib; baseline snapshot beside its test
data/        datasets (≤ 1 MB each, D-19)  configs/
examples/    composition roots
notebooks/   docs/
```

## 7. Cross-codebase note

MlLib independently evolved the same instincts as tradePlatform (strategy injection, two-tier
specialisation, self-describing metadata) and the same failure modes (hardcoded enumerations,
declared-intent-left-stubbed, descriptor drift). When MlLib models plug into tradePlatform as strategy
plugins (BL-19), descriptors are introspected at the boundary so external models self-describe on
arrival — which is why R3 (introspected metadata) precedes any plugin work.
