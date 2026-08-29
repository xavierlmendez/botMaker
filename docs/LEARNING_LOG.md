# Learning log

One entry per technique implemented by hand. Written for future-me. Entries below are seeded from the
code that exists on 2026-08-28; each should be expanded when the module is next touched.

## Linear regression by gradient descent · 2025-12
- **What.** h(x) = w·x + b, fit by minimising MSE (or MAE) with batch gradient descent.
- **Where.** `ml/linear_regression.py` · `math/hypothesis.py` · `math/loss_function.py` · test `ml/tests/test_linear_regression.py`
- **Design.** The loss object carries its own gradient (`compute_gradient`), so the model never knows which loss it is minimising. The Boston-housing comparison against sklearn lives in `examples/boston_housing_vs_sklearn.py`.
- **What was confusing.** Learning rate scale: unscaled features needed ~3e-6 and 15k epochs to converge.

## Polynomial regression as a hypothesis expander · 2025-12-14
- **What.** Φ(x) maps features to degree-d monomials; linear regression in Φ-space is polynomial regression in x-space.
- **Where.** `math/hypothesis_expander.py` (`PolynomialRegressionExpander`) · test `math/tests/test_hypothesis.py`
- **Design.** The expander is injected into `HypothesisFunction`, so "which regression" is a constructor argument, not a subclass.

## Logistic regression · 2025-12
- **What.** Same descent loop as linear regression; the hypothesis's `compute_classification` thresholds the output.
- **Where.** `ml/logistic_regression.py` · project configs in `ml/projects/ad_click_logistic_regression.py` · test `ml/tests/test_logistic_regression.py`
- **Open question.** BL-23: on the ad-click data every grid permutation predicts all-ones. Threshold, scaling, or gradient sign — to be resolved in slice 5.3b and written up here.

## Perceptron and svm with sub-gradient updates · 2025-12
- **What.** Perceptron loss and hinge loss are non-differentiable at the margin; both models step along a sub-gradient.
- **Where.** `ml/perceptron.py` · `ml/svm.py` · `PerceptronLoss`, `HingeLoss` in `math/loss_function.py` · test `ml/tests/test_perceptron_svm.py`
- **Design.** The two models are structurally identical; the loss is the only difference — evidence for R2 (shared descent base).

## Decision tree with Gini impurity · 2026-01
- **What.** Recursive binary splits choosing the feature/threshold that minimises weighted Gini.
- **Where.** `ml/decision_tree.py` · `math/graph/split_function.py` · `tree_structures.py` · test `ml/tests/test_decision_tree.py`
- **Next.** InformationGain / ChiSquare criteria (BL-08); no-split error handling (slice 3.4).

## Graph and tree structures; BFS/DFS as a template method · 2026-01-28 → 2026-02-15
- **What.** Adjacency-based graph and tree classes; search algorithms share `run() → _search() → _notify_evaluator()` over a frozen `SearchContext`.
- **Where.** `math/graph/graph_structures.py`, `tree_structures.py` · `math/algorithms/abstract_graph_algorithm.py`, `breadth_first_search.py`, `depth_first_search.py` · tests under both `tests/` dirs
- **Design.** Most mature code in the repo; the template-method ABC is the pattern to copy for iterative deepening (BL-04).

## Probabilistic KNN (skeleton) · 2025-12
- **What.** Prior → likelihood via nearest neighbours → posterior classification.
- **Where.** `ml/probabilistic_knn.py` · `math/probability/Prior.py`, `bayes_rule.py`, `gaussian_prior.py`
- **Next.** Sum/product rule (BL-07) alongside CS 6344 probability material.

## Model evaluation records · 2025-12
- **What.** Accuracy / precision / recall from a hand-built confusion matrix, persisted per grid iteration.
- **Where.** `ml/evaluators/generic_evaluator.py`
- **Lesson.** BL-22 (fixed 2026-08-28): FP and FN were swapped in two duplicated copies of the loop — the reported precision 1.0 was really recall. Now one vectorised base implementation checked against sklearn on an 8-row hand-built case (`tests/test_confusion_matrix.py`). Duplicated code hid the bug twice.
