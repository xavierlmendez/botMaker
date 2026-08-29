# MlLib Architecture

Living document. Promoted from `docs/reviews/2026-07-18-architecture-review.md` on 2026-08-28; the
review stays as the dated record, this file tracks the current and target shape. Update it in the same
PR as any change to a layer boundary, a base-class contract, or an extension point.

## 1. Decomposition

Knowledge is sorted by *kind*, not by feature, with a strictly one-way dependency flow:

```
projectScripts ──▶ ml ──▶ math
       ▲
   data
```

Math never imports ML; ML composes math; scripts (composition roots) wire data to models.

| Layer | Contents | Role |
|---|---|---|
| `math` | `HypothesisFunction`, `HypothesisExpander`, `LossFunction` (MSE, MAE, Perceptron, Hinge), `CostFunction`, `RegularizationFunction`, `graphBased/` (graph, tree, `SplitFunction`/Gini), `algorithmImplementations/` (BFS, DFS on an ABC), `probabilityBased/` | academic ideas as classes |
| `ml` | `MyLinearRegression`, `MyLogisticRegression`, `MyPerceptron`, `MySVM`, `DecisionTree`, `ProbabilisticKNN`, `modelEvaluators/`, `projectSpecificFiles/` | models composed from primitives |
| `data` | `data_orchestrator` (+ `DataTransformer`), datasets, transformer JSON configs | load → transform → split |
| `projectScripts` | `AdClickModelProjectBuildScript` etc. | composition roots / experiments (→ `examples/` in slice 3.5) |

The idea→class mapping is deliberately literal: h(x)=w·x+b is `HypothesisFunction`; the feature map Φ
is `HypothesisExpander`; per-sample loss vs dataset cost mirror the lecture distinction; each loss
carries its own gradient.

## 2. Contracts (what a new component must satisfy)

**Injected math objects.** A model is a host for injected objects and is written once against these
interfaces. Adding a loss or an expander never touches a model.

| Interface | Must provide | Used by |
|---|---|---|
| `LossFunction` | `compute_loss(actual, predicted)`, `compute_gradient(actual, predicted[, data_values])`; margin losses also `compute_bias`; class attribute `task_kind: TaskKind \| None` | every gradient-descent model |
| `HypothesisExpander` | `expand_hypothesis(weights)`, `fit_data_to_hypothesis(data)` | `HypothesisFunction` |
| `HypothesisFunction` | `compute_prediction(x)`, `compute_classification(x)`; owns weights, bias, expander | every model |
| `SplitFunction` | impurity of a candidate split | `DecisionTree` |
| `Transformer` | `fit(frame) -> self`, `transform(frame) -> new frame`, `fit_transform` | `TransformerPipeline` (6.2), `DataOrchestrator` |
| `ModelEvaluator` | `update_testing_prediction_data(...)`, `evaluate_model()`, `persist_evaluation_record()` | every model's `evaluate` |
| `AbstractGraphAlgorithm` | `_search(ctx)`; the ABC owns `run()` and `_notify_evaluator()`; `SearchContext` is frozen | graph algorithms |

**Gradient-descent models.** `ml.gradient_descent.GradientDescentModel(hypothesis, loss, learning_rate, epochs)`
owns `fit` / `predict_values` / gradient / update / cost and accepts arrays or DataFrames. A subclass sets
`predict_method` (`compute_prediction` or `compute_classification`) and, for a sign classifier,
`encode_targets` (labels → ±1) — nothing else. `MyLogisticRegression` adds
an evaluator and `grid_fit(X, y, test_size, random_state)`, which makes the train/test split itself.

**Two-tier specialisation.** A library base (`MyLogisticRegression`) plus a thin project subclass that
carries only `num_weights` and a hyper-parameter grid (`projectSpecificFiles/ad_click_logistic_regression.py`).
The subclass must call `super().__init__()` (F4/slice 3.3) — a project class that skips it is a bug.

**Descriptors.** `mllib.describe.describe(obj)` returns `{name, module, kind, doc, params, signature}` derived
from the class (name, docstring, constructor signature). Components carry no hand-typed metadata; a class
docstring is the description. This is the seam tradePlatform introspects (BL-19).

**Evaluation records.** `evaluator.evaluation_record[iteration]` is a JSON-serialisable dict with
`modelData`, confusion-matrix counts, `accuracy`, `precision`, `recall`. The training baseline test
depends on this shape; changing it means regenerating the snapshot deliberately.

## 3. Pipeline trace (ad-click project)

`DataOrchestrator(csv, 'csv', …)` → transformer builds one dataframe per model shape →
`build_test_train_split(model)` → project subclass sets grid → `grid_fit` sweeps `ParameterGrid`
(fresh `HypothesisFunction` per combination, `epoch` rounds of gradient descent) → evaluator records
each permutation → `print_evaluation` reports the best. The smoke version of this trace is
`tests/ml/test_training_baseline.py`.

## 4. Extension points — how to add things

- **A loss:** subclass `LossFunction` in `math/loss_function.py`; implement loss + gradient; add a
  unit test that checks the gradient numerically; add a learning-log entry.
- **A model:** compose from `HypothesisFunction` + a loss; expose `fit`, `predict`, `predict_values`,
  `evaluate`; give it an evaluator; add it to the smoke grid only if it is part of a project comparison.
- **A graph algorithm:** subclass `AbstractGraphAlgorithm`; implement `_search`; never override `run`.
- **A transformer:** subclass `data.transformers.Transformer` (`fit` learns state and returns `self`;
  `transform` returns a new frame, never mutating); add it to `transformers/__init__.py`; after 6.2 declare it
  by class name in the project's JSON config.

## 5. Known structural debt and its schedule

| Debt | Where | Resolution |
|---|---|---|
| Hardcoded model-name ladder; `TransformerPipeline` commented out (F1) | `data_orchestrator.get_transformed_data` | Phase 6 (R1, BL-09) |
| Linear/logistic duplication (F2) | `linear_regression.py`, `logistic_regression.py` | **done** slice 5.3 — `ml/gradient_descent.py` |
| Hand-typed drifting `metadata` dicts (F3) | ~25 classes | **done** slice 5.1 — `mllib.describe.describe(obj)` |
| Subclass skips `super().__init__()` (F4) | `ad_click_logistic_regression.py` | slice 3.3 |
| Base `persist_evaluation_record` builds a set literal (F5) | `generic_evaluator.py:44` | slice 3.2 |
| Web layer was a stub (F6) | `fastapi_app/` | removed, BL-01 |
| Copy-on-write incompatibility, swapped FP/FN, degenerate classifier | evaluator, training loop | **done** BL-21 (3.4b), BL-22 (3.4c), BL-23 (5.3b) |
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
