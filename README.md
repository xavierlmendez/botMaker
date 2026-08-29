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
data/                datasets (≤ 1 MB each, D-19) · configs/ (per-project transformer pipelines, JSON)
examples/            composition roots: ad-click model comparison, Boston housing vs sklearn, graph search vs networkx
docs/                plan, backlog, decisions, learning log, reviews, reports
notebooks/           coursework notebooks
```

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

The 2026-08 refactor is complete (`docs/plans/2026-08-refactor.md`, D-21). Open work is the backlog
(`docs/BACKLOG.md`); the next initiatives pair with CS 6344 (probability placeholders, PCA/SVD).

## Documents

- `docs/plans/` — executed plans, archived with a "what changed" section
- `docs/DECISIONS.md` — decision log
- `docs/BACKLOG.md` — stripped initiatives and tagged TODOs
- `docs/LEARNING_LOG.md` — one entry per technique implemented by hand
- `docs/ARCHITECTURE.md` — decomposition, contracts, extension points
- `docs/reviews/` — dated architecture reviews
- `docs/reports/` — the MlLib model-training report (.docx/.pdf)
