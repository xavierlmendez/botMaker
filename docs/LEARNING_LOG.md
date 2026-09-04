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
- **Resolved (BL-23, 2026-08-29).** The sign output lives in {-1, +1}; training it against {0, 1} labels meant the
  gradient never pointed at the decision boundary — separable data stalled at 0.42. `encode_targets` maps labels to
  ±1 and the same model reaches 1.0. The ad-click data itself has no linear signal (sklearn logistic 0.650 CV vs
  0.650 majority), which is why the baseline was — and remains — at the majority rate. Lesson: check the label space
  of the loss before blaming the optimiser; and know the ceiling of the data (a stronger model gets 0.715).

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

## Self-describing components by introspection · 2026-08-29
- **What.** Replace 46 hand-typed `metadata = {"name", "description"}` dicts with one `describe(obj)` that reads the
  class name, docstring, and constructor signature via `inspect`.
- **Where.** `src/mllib/describe.py` · test `tests/test_describe.py`
- **Design.** Descriptors that are *derived* cannot drift; the review (F3) found three copy-pasted ones
  (`CostFunction`, `RegularizationFunction`, `Prior` all called themselves something else). Nothing ever read
  `.metadata`, so the switch was behaviour-neutral — the baseline proved it.
- **What was confusing.** Where to put the human text: the answer is the class docstring, which tooling already
  understands, not a parallel data structure.
- **Reference.** Python `inspect` module; the world-model doctrine in tradePlatform's `PHILOSOPHY.md`.

## Gradient descent as a template method · 2026-08-29
- **What.** Batch gradient descent is the same loop whether the hypothesis outputs a value or a sign: predict,
  take the loss gradient, project it through the design matrix (`Φ(X)ᵀ·∇`), step. Linear and logistic regression
  differed by one method call — so the base class holds the loop and the subclass names the method.
- **Where.** `src/mllib/ml/gradient_descent.py` · `linear_regression.py` (3 lines) · `logistic_regression.py`
  (constructor, `grid_fit`, `evaluate`) · test `tests/ml/test_gradient_descent.py` (property-based: recovers
  any line in [-3, 3]²).
- **Design.** The identity `HypothesisExpander` makes "no feature map" a valid object instead of a `None` check,
  so the gradient is always `Φ(X)ᵀ·∇` (BL-12). `grid_fit` now owns the split (BL-20).
- **What was confusing.** The grid loop applies one descent-shaped step with the *expanded initial weights* as
  the gradient before training. It is preserved verbatim because the baseline pins it; whether it is a bug is
  part of BL-23.
- **Reference.** Gang of Four, Template Method; the loss-gradient projection is the chain rule through h.

## Declarative data pipelines · 2026-08-29
- **What.** A project's feature engineering as data: a JSON file naming transformer classes and their arguments
  per frame, resolved by name from a closed registry and run in order. The orchestrator no longer knows any
  project; it loads a config.
- **Where.** `src/mllib/data/pipeline.py` · `src/mllib/data/transformers/` · `data/configs/ad_click_transformations.json`
  · tests `tests/data/` (fingerprints of the original hand-written frames are the regression oracle).
- **Design.** `fit` learns state (means, category levels, bin edges) and `transform` applies it, so the same
  pipeline encodes *new* data consistently — the hand-written version could only transform the frame it was
  given. Name resolution is restricted to `mllib.data.transformers` so config cannot execute arbitrary code.
- **What was confusing.** Two silent dependencies in the legacy code surfaced only when replaced by objects:
  the bin edges were monotonic only because NaN-filling shrank the std, and one-hot `drop_first` dropped a
  *different* level on data with fewer categories. Both are now explicit (`BinByStdRanges` clips edges;
  `OneHotEncode` records levels at fit).
- **Reference.** sklearn's `TransformerMixin` / `Pipeline` contract, which this mirrors deliberately.

## A* over an implicit graph with an admissible bound · 2026-09-04
- **What.** Best-first search where the graph is never built. The problem hands back successors on demand,
  so a search space of C(n, k) subsets is explored without materializing it. The first goal state popped is
  optimal, which is the whole point: A* does not just find a good answer, it proves no better one exists.
- **Where.** `math/graph/abstract_graph_problem.py` · `math/search_cost_function.py` ·
  `math/algorithms/a_star_search.py` · tests `tests/math/algorithms/test_a_star_search.py`.
- **Design.** The objective is terminal-only: cost belongs to a completed solution, not to the path taken to
  it, because the error of a chosen subset does not depend on the order its members were picked in. So there
  is no accumulated `g` and the priority is the bound alone. Two states that name the same position must be
  equal, which is why subsets are stored as ascending tuples — otherwise the same subset is expanded k!
  times. The algorithm subclasses `AbstractGraphAlgorithm` and implements `_search`, so the evaluator
  orchestration in the base `run` applies unchanged (D-23).
- **What was confusing.** Admissibility and tightness do different jobs, and it is easy to conflate them. A
  bound of zero is admissible whenever costs are non-negative, and A* with it still returns the optimum — it
  just enumerates the entire reachable graph to get there. The tests pin both halves on the same instance:
  the exact bound expands 4 states, the zero bound expands 11, and they return the same subset. Tightness is
  a speed property, not a correctness one.
- **Reference.** Hart, Nilsson & Raphael, *A formal basis for the heuristic determination of minimum cost
  paths*, IEEE TSSC 1968; Pearl, *Heuristics*, 1984, for the first-goal-popped optimality argument.

## Nyström landmark selection as column subset selection · 2026-09-04
- **What.** Choosing k landmarks for the Nyström approximation of a positive semi-definite kernel is column
  subset selection on K^{1/2}. The residual trace tr(K − K[:,S] K[S,S]⁺ K[S,:]) equals the squared Frobenius
  residual of selecting the same columns of K^{1/2}, so a CSS search chooses landmarks unmodified. Paired
  with the spectral bound, A* returns the landmark set no other set can beat, and says so.
- **Where.** `math/graph/nystrom_landmark_problem.py` · tests
  `tests/math/graph/test_nystrom_landmark_problem.py` · kernels in `tests/math/fixtures/`.
- **Design.** The bound asks what the best conceivable finish could remove. After projecting the chosen
  columns out of K^{1/2}, no r further *columns* can remove more energy than the top r *eigenvalues* of what
  remains, because columns are a restricted choice of direction and eigenvectors are the unrestricted best.
  Subtracting the eigenvalues therefore understates the true remaining error, which is what admissibility
  needs. At the root the bound is exactly the rank-k eigenvalue tail, i.e. the SVD residual: the second
  number the whole question needs, since it separates "this heuristic is bad" from "no subset can do better".
- **What was confusing.** Two numerical details are load-bearing rather than decorative. A kernel built from
  data carries small negative eigenvalues from rounding, and their square roots are not real, so they are
  clipped before K^{1/2} is formed. And the bound is a difference of two large near-equal quantities: on a
  badly scaled kernel it cancels to just below zero, and an unclamped negative bound sorts ahead of every
  real state and empties the fringe. Both are one line and neither is optional. CI added a corollary:
  *which side of zero* the cancellation lands on is rounding, and Apple's Accelerate and Linux's OpenBLAS
  round it differently. A test that asserted the raw value was negative passed here and failed there. The
  clamp is now pinned by injecting the overshoot through the class's own spectrum seam, and a separate test
  asserts only the platform-independent half — that the clamped value is never negative.
- **Cost.** The bound recomputes an SVD and an n×n eigendecomposition per child, which is O(n³) each and
  caps this implementation near n = 40 (BL-27). The published method downdates the parent's spectrum by rank
  one instead. Correct now, not yet fast.
- **Reference.** Arai, Maung & Schweitzer, *Optimal column subset selection by A-star search*, AAAI 2015;
  Williams & Seeger, *Using the Nyström method to speed up kernel machines*, NeurIPS 2000.

## Greedy Nyström and pivoted Cholesky · 2026-09-04
- **What.** The two deterministic landmark rules the literature actually uses, implemented so their gap to
  the certified optimum can be measured. Greedy Nyström adds the column that removes the most residual
  trace; pivoted Cholesky adds the column with the largest residual diagonal.
- **Where.** `math/algorithms/nystrom_landmark_selectors.py` · tests
  `tests/math/algorithms/test_nystrom_landmark_selectors.py`.
- **Design.** Neither rule needs to try a column to score it. On the residual kernel R left by the current
  selection, adding column j removes exactly ||R[:, j]||² / R[j, j] of the trace, so one sweep of the
  residual ranks every candidate. Pivoted Cholesky reads only R[j, j], which is one number per column
  instead of a norm, and that is the whole difference between them: the trace rule asks how much a column
  explains about everything, the diagonal rule asks only how much is left unexplained about the column
  itself. An outlier far from every other point scores high on the second and explains nothing.
- **What was confusing.** Both rules are called "greedy" in conversation, and a harness that reports one
  under that name while implementing the other changes the headline number. Worse, it is easy to invent a
  third: greedily minimizing A*'s lower bound looks like the natural greedy rule when the bound is already
  written, but no paper proposes it, because the bound credits a completion that greedy will never make.
  It is kept here under a name that cannot be mistaken for the published one (D-22).
- **Reference.** Farahat, Ghodsi & Kamel, *A novel greedy algorithm for Nyström approximation*, AISTATS
  2011; Wan & Schweitzer, IJCAI 2021, Thm 2, for the identity between the trace rule and the f = u search.

## RPCholesky, and reporting a randomized method honestly · 2026-09-04
- **What.** Randomly pivoted Cholesky draws each landmark with probability proportional to the residual
  diagonal instead of taking the largest, plus the summarization that turns any randomized selector into a
  reportable number: mean, median and spread over independent seeds.
- **Where.** `math/algorithms/nystrom_randomized_selectors.py` · tests
  `tests/math/algorithms/test_nystrom_randomized_selectors.py`.
- **Design.** The update is identical to deterministic pivoted Cholesky, so the two share one step function
  and differ only in how the pivot is chosen. That is the entire idea: weighting by the residual diagonal
  keeps the draw near what is still unexplained while leaving an outlier only a proportional chance of
  being taken, which softens the failure the deterministic rule walks straight into. Every selector takes an
  explicit seed, so a run is reproducible, and `summarize_randomized_selector` runs consecutive seeds.
- **What was confusing.** Best-of-N random looks like a baseline and is not one. Drawing N subsets and
  keeping the best spends N evaluations of the objective, so it is a crude optimizer whose quality is bought
  rather than earned, and it approaches the optimum as N grows. Quoting best-of-32 as "random" alongside a
  greedy rule that scores each candidate once compares two different budgets and flatters the wrong method.
  The class name carries N for that reason, and a test asserts more draws never do worse (D-22).
- **Reference.** Chen, Epperly, Tropp & Webber, *Randomly pivoted Cholesky*, CPAM 2025 (arXiv:2207.06503).

## Measuring an optimality gap without overstating it · 2026-09-04
- **What.** The harness that puts the landmark heuristics against the certified optimum on real data, and
  reports the two gaps that the question actually has: how far a heuristic is from the best subset, and how
  far the best subset is from the best rank-k subspace.
- **Where.** `ml/projects/nystrom_uci_data.py` · `ml/projects/nystrom_uci_harness.py` · tests
  `tests/ml/test_nystrom_uci_data.py`, `tests/ml/test_nystrom_uci_harness.py` · data `data/uci/`.
- **Design.** Reporting only the first gap makes a heuristic look bad on a kernel where nothing could have
  done well, so the SVD rank-k residual is printed beside every ratio: it is what the eigenvectors achieve
  when free to be any direction, and no subset can beat it. The A* node count is printed beside C(n, k) for
  the same reason in the other direction — a certificate that costs 363 states out of 9,880 subsets is a
  different proposition from one that costs 9,000. Bandwidth is a first-class sweep parameter because it is
  the knob that moves the spectrum between decaying and flat.
- **What was confusing.** The first version of this harness inflated its own headline. It compared A*
  against a greedy that minimized A*'s *lower bound*, a lookahead rule nobody publishes, and against the
  best of 32 random draws labelled simply "random". Both flatter the optimum. Corrected, the standard greedy
  is within 1.000–1.034× of optimal on these slices, a single random draw averages 1.22–1.42×, and
  best-of-32 sits at 1.06–1.12× purely because it evaluates 32 of 9,880 subsets. Greedy pivoted Cholesky is
  the one genuine outlier at 1.33–1.49×, which is the known failure RPCholesky was designed to soften. The
  naming rules are now a decision (D-22) rather than a habit, because the error was invisible in the output.
  Review caught the same failure once more, in this entry: the random-draw range was written from memory as
  1.21–1.39 when the run says 1.22–1.42. Every figure quoted above is now read off the committed output.
- **Scale.** At n = 40 brute force takes under a second, so nothing here is evidence about certification at
  scale. BL-27 is the bound's cost and BL-28 is the experiment that would be evidence.
- **Reference.** Dereziński, Khanna & Mahoney, *Improved guarantees and a multiple-descent curve for CSS and
  the Nyström method*, IJCAI 2021, for the best-subset-versus-SVD ratio this reports.

## Scoring siblings together: the Schur complement as shared work · 2026-09-04
- **What.** When A* expands a parent it scores every child, and for subset selection the children differ
  from the parent by one column each. One residual kernel of the parent prices all of them: adding column
  j to selection S lowers the residual trace by exactly ||R_S[:, j]||² / R_S[j, j]. That is the greedy
  selector's gain rule, reused as the search's child-scoring rule.
- **Where.** `math/search_cost_function.py` (`lower_bounds`) · `math/algorithms/a_star_search.py` ·
  `math/graph/nystrom_landmark_problem.py` · tests in `tests/math/graph/test_nystrom_landmark_problem.py`
  and `tests/math/algorithms/test_a_star_search.py` · benchmark `examples/nystrom_batched_bounds.py`.
- **Design.** The contract gains one method with a default, so the abstraction costs nothing for a problem
  that has nothing to share, and the search calls it once per expansion in the order the problem generated
  the children, so tie-breaking cannot drift. The profile that motivated it said something worth keeping:
  91% of the search was not arithmetic but the per-call overhead of 8,000 pseudo-inverses of 3×3 blocks,
  so the fix was to make fewer calls, not faster ones. The GPU question dissolved on the same evidence.
- **What was confusing.** A test broke that had nothing wrong with it. The ten-point RBF chain is a
  palindrome, so two mirror-image subsets have identical cost, and the batched and per-child arithmetic
  differ in the last bit — enough to hand back the other mirror. "The optimal subset" is only defined up
  to ties, and a brute-force oracle must return the set of optima, not the first one. The same phenomenon
  at scale is the 2^k tie collapse the reproduction saw on wdbc at k = 15.
- **Reference.** Farahat, Ghodsi & Kamel, AISTATS 2011 (the gain identity); Arai, Maung & Schweitzer,
  AAAI 2015, §4, for the parent-once, child-cheap structure this is the first step toward.
