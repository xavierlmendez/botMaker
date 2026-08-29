# botMaker / MlLib

A from-scratch machine-learning library in Python, built to learn the mathematics behind the models by
implementing them by hand. Models are assembled from injected math objects rather than written as
monoliths, so the same gradient-descent loop serves regression and classification depending on which
hypothesis, expander, and loss are plugged in.

## Layout

```
src/mllib/
  mathDomain/        hypothesis h(x)=w·x+b · hypothesisExpander Φ (polynomial features) · lossFunction
                     (MSE, MAE, Perceptron, Hinge — each carries its own gradient) · costFunction ·
                     regularizationFunction · graphBased/ (graph, tree, Gini split) ·
                     algorithmImplementations/ (template-method BFS/DFS) · probabilityBased/
  mlDomain/          MyLinearRegression · MyLogisticRegression · MyPerceptron · MySVM · DecisionTree ·
                     ProbabilisticKNN · modelEvaluators/ · projectSpecificFiles/ (ad-click grids)
  dataDomain/        DataOrchestrator (load → transform → split) · transformers
tests/               mirrors src/mllib; the training baseline and its snapshot live in tests/mlDomain/
data/                datasets (≤ 1 MB each, D-19) · configs/ (transformer JSON)
examples/            composition roots: ad-click model comparison, Boston housing vs sklearn, graph search vs networkx
docs/                plan, backlog, decisions, learning log, reviews, reports
notebooks/           coursework notebooks
```

Subpackage and module names are still `camelCase`; slices 4.2–4.3 move them to PEP 8.

## Run

```
uv sync          # Python 3.12, locked dependencies
uv run pytest    # full suite including the training baseline
```

Requires [`uv`](https://docs.astral.sh/uv/); it installs Python 3.12 from `.python-version` if needed.

## How a model is composed

```python
from mllib.mathDomain.hypothesis import HypothesisFunction
from mllib.mathDomain.lossFunction import MSE
from mllib.mlDomain.linearRegression import MyLinearRegression

h = HypothesisFunction(initial_weights, initial_bias)  # h(x) = w·x + b
model = MyLinearRegression(h, MSE(), learningRate=3e-6, epochs=15_000)
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
