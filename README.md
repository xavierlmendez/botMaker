# botMaker / MlLib

A from-scratch machine-learning library in Python, built to learn the mathematics behind the models by
implementing them by hand. Models are assembled from injected math objects rather than written as
monoliths, so the same gradient-descent loop serves regression and classification depending on which
hypothesis, expander, and loss are plugged in.

## Layout

```
src/mllib/
  math/        hypothesis h(x)=w·x+b · hypothesis_expander Φ (polynomial features) · loss_function
                     (MSE, MAE, Perceptron, Hinge — each carries its own gradient) · cost_function ·
                     regularization_function · graphBased/ (graph, tree, Gini split) ·
                     algorithmImplementations/ (template-method BFS/DFS) · probabilityBased/
  ml/          MyLinearRegression · MyLogisticRegression · MyPerceptron · MySVM · DecisionTree ·
                     ProbabilisticKNN · modelEvaluators/ · projectSpecificFiles/ (ad-click grids)
  data/        data_orchestrator (load → transform → split) · transformers
tests/               mirrors src/mllib; the training baseline and its snapshot live in tests/ml/
data/                datasets (≤ 1 MB each, D-19) · configs/ (transformer JSON)
examples/            composition roots: ad-click model comparison, Boston housing vs sklearn, graph search vs networkx
docs/                plan, backlog, decisions, learning log, reviews, reports
notebooks/           coursework notebooks
```

Class, method, and variable names are still `camelCase`; slice 4.3 moves them to PEP 8.

## Run

```
uv sync          # Python 3.12, locked dependencies
uv run pytest    # full suite including the training baseline
```

Requires [`uv`](https://docs.astral.sh/uv/); it installs Python 3.12 from `.python-version` if needed.

## How a model is composed

```python
from mllib.math.hypothesis import HypothesisFunction
from mllib.math.loss_function import MSE
from mllib.ml.linear_regression import MyLinearRegression

h = HypothesisFunction(initial_weights, initial_bias)  # h(x) = w·x + b
model = MyLinearRegression(h, MSE(), learning_rate=3e-6, epochs=15_000)
model.fit(X_train, y_train)
```

Swap `MSE()` for `HingeLoss()` and the model for `MySVM` and the same shape trains a margin classifier;
inject `PolynomialRegressionExpander(degree=3)` into the hypothesis for polynomial regression.

## Status

Under an active refactor (`docs/REFACTOR_PLAN.md`): Phase 0 complete, Phase 1 in progress. Known
issues are tracked in `docs/BACKLOG.md` (notably BL-21..23, found while building the training baseline).

## Documents

- `docs/REFACTOR_PLAN.md` — phases and slices
- `docs/DECISIONS.md` — decision log
- `docs/BACKLOG.md` — stripped initiatives and tagged TODOs
- `docs/LEARNING_LOG.md` — one entry per technique implemented by hand
- `docs/ARCHITECTURE.md` — decomposition, contracts, extension points
- `docs/reviews/` — dated architecture reviews
- `docs/reports/` — the MlLib model-training report (.docx/.pdf)
