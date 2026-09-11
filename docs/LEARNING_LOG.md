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
  real state and empties the frontier. Both are one line and neither is optional. CI added a corollary:
  *which side of zero* the cancellation lands on is rounding, and Apple's Accelerate and Linux's OpenBLAS
  round it differently. A test that asserted the raw value was negative passed here and failed there. The
  clamp is now pinned by injecting the overshoot through the class's own spectrum seam, and a separate test
  asserts only the platform-independent half — that the clamped value is never negative.
- **Cost.** The bound recomputes an SVD and an n×n eigendecomposition per child, which is O(n³) each and
  caps this implementation near n = 40 (BL-27). The published method downdates the parent's spectrum by rank
  one instead. Correct now, not yet fast.
- **Reference.** Arai, Maung & Schweitzer, *Optimal column subset selection by A-star search*, AAAI 2015;
  Williams & Seeger, *Using the Nyström method to speed up kernel machines*, NeurIPS 2000.

## The parent-once, child-cheap structure, completed · 2026-09-04
- **What.** The Nyström bound for every child above goal depth from one eigendecomposition of the
  parent, closing BL-27. Slice 6.1 had this structure for goal-depth children only (a Schur complement
  of the parent prices every complete child); this extends it to every depth, which is where the search
  actually spends its time once k > 3.
- **Where.** `math/graph/nystrom_landmark_problem.py` (`_downdated_bounds`) · tests
  `tests/math/graph/test_nystrom_landmark_problem.py` (equivalence with the oracle at every depth and
  tolerance; same expansions as the oracle; explained columns) · `examples/nystrom_batched_bounds.py`.
- **Design.** Everything happens in the kernel's eigenbasis: with K = V D Vᵀ, the reduced coordinates
  D^{1/2} Vᵀ are K^{1/2} with every column inner product intact, in r dimensions instead of n. The
  residual Gram of a parent S has the spectrum of H = D - Z Zᵀ, Z = D^{1/2} Q, Q an orthonormal basis of
  the chosen columns; one `eigh(H)` per parent. A child adds one direction, the unit residual q_j of its
  column against Q, so its Gram is H minus one outer product and its spectrum is a rank-one downdate of
  the parent's, which the secular-equation solver returns for all children at once. The child's
  remaining energy is tr(H) - ||D^{1/2} q_j||² (the same gain identity D-24 used at goal depth) and its
  best possible completion is the top eigenvalues of the downdate. The oracle `lower_bound` is kept
  untouched, and the test that matters is that both paths agree at every state of every fixture.
- **What was confusing.** Two places where the exact rank matters. First, the all-ones kernel has
  retained rank one while a child at the root still needs two more landmarks; asking the solver for
  more eigenvalues than the rank is an error, and the right answer is that everything past the rank
  is zero. Second, the paper's Theorem 4 downdates with the *projection* of the new column onto the
  parent span; the lemma needs the *residual*, and only the residual reproduces its own node counts
  (the reproduction found this; IJCAI-21 writes it correctly). The search baseline (slice 0b) is what
  turned "the numbers look right" into a byte-identical diff.
- **Reference.** Arai, Maung & Schweitzer, AAAI 2015, §4 [A1]; Wan & Schweitzer, IJCAI 2021, eq. (10)
  [A3]; Bunch, Nielsen & Sorensen 1978 [L4].

## Truncation is admissible by Schur-complement monotonicity · 2026-09-04
- **What.** Computing the bound on the kernel with its smallest eigenvalues dropped, and why that
  needs no correction term (D-26, BL-29).
- **Where.** `NystromLandmarkProblem.spectrum_mass_tolerance` · tests: admissibility against the true
  objective at every state and four tolerances; optimum membership under truncation; the rank rules.
- **Design.** Write the Nyström residual as a Schur complement, E_K(T) = tr(K / K_TT). For A ⪯ B with the
  same block partition, xᵀ(A / A₁₁)x = min_y [x; y]ᵀ A [x; y] ≤ min_y [x; y]ᵀ B [x; y] = xᵀ(B / B₁₁)x,
  so A / A₁₁ ⪯ B / B₁₁. With A = K̃ ⪯ B = K this gives E_K̃(T) ≤ E_K(T) for every complete T. The bound
  f̃(S) is admissible for the K̃ problem by the usual spectral argument, hence
  f̃(S) ≤ min_T E_K̃(T) ≤ min_T E_K(T): exactly what A* needs. Goal costs stay on K so the first goal
  popped is optimal for the true objective. The bound gets looser, never wrong; how much looser is a
  measurement (EXP-09a), not a theorem.
- **What was confusing.** The first instinct was to "add the dropped mass back" to the remaining
  energy so the bound would be computed on the right trace. That makes the bound *larger*, which is
  the wrong direction for admissibility, and it is unnecessary: both terms of the bound shrink together
  on the smaller matrix, and the inequality chain above is what carries the guarantee. The other trap
  was the oracle: if `kernel_sqrt` stayed exact while the fast path truncated, D-24's "same value both
  paths" would silently hold only at δ = 0. Both now see K̃; only the objective sees K.
- **Reference.** Schur-complement monotonicity is standard (Horn & Johnson, *Matrix Analysis*, §7.7);
  the application to the Nyström bound is the research spec's §10.

## The secular equation: a child's spectrum from its parent's · 2026-09-04
- **What.** The eigenvalues of `diag(λ) − w wᵀ`, for many `w` at once, without decomposing anything.
  This is the arithmetic that lets a search decompose a parent once and price every child by a
  rank-one downdate (BL-27); the bound that uses it lands in the next slice.
- **Where.** `math/secular_equation.py` · tests `tests/math/test_secular_equation.py`.
- **Design.** Subtracting an outer product moves each eigenvalue down into the gap below it
  (interlacing), and the new values are the roots of `s(μ) = 1 − Σ w_j²/(λ_j − μ)`. On each gap `s`
  is strictly decreasing, from +∞ at the bottom to −∞ at the top, so bisection with 64 halvings finds
  every root to rounding, and because the bracket is known in advance the search over all children is
  one array of midpoints per iteration. A gap of zero width is a repeated eigenvalue: the root is the
  eigenvalue itself and no equation is solved. The last root has no eigenvalue below it; its bracket
  is `[λ_r − ‖w‖², λ_r]`, and a test pins that it is searched rather than clamped.
- **What was confusing.** The paper describes the downdate for the projection of the new column onto
  the parent span; the lemma it rests on needs the residual against that span, and only the residual
  reproduces the published node counts (the reproduction found this; IJCAI-21 writes it correctly).
  That detail belongs to the bound, not to this solver, which only ever sees `w`, but it is why this
  module's docstring says "in the parent's eigenbasis" and nothing about columns.
- **Reference.** Bunch, Nielsen & Sorensen, *Rank-one modification of the symmetric eigenproblem*,
  Numer. Math. 1978 [L4]; Arai, Maung & Schweitzer, AAAI 2015, §4 [A1]; Wan & Schweitzer, IJCAI 2021,
  eq. (10) [A3].

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

## What A* must remember, and what it may forget · 2026-09-05
- **What.** Three ways to store less of the frontier without changing what the search does: keep only the
  cheapest goal child of each parent, since a dearer goal sibling can never be the first goal popped; drop
  any child whose bound exceeds the best goal cost seen so far, since that goal is popped first; and cap the
  frontier so a run that would exhaust memory stops with a named exception instead of an uncertified
  answer. The first two rest on one line of the cost contract, `lower_bound == goal_cost` at a goal, and
  on how A* terminates: the first goal popped is the answer, so an entry that cannot be popped before that
  goal was never needed. Expansions are unchanged; only what sits in memory between them changes.
- **Where.** `math/algorithms/pruned_a_star_search.py` (`PrunedAStarSearch`, the three mechanisms,
  `FrontierLimitExceeded`, `IncumbentBelowOptimum`, `frontier_peak`) · `math/algorithms/a_star_search.py`
  (exact A*, unchanged in behaviour; its loop split into `_price_children`, `_push_children`,
  `_no_goal_reachable` so a variant overrides only where it diverges) · tests in
  `tests/math/algorithms/test_pruned_a_star_search.py`, `tests/math/graph/test_nystrom_landmark_problem.py`,
  `tests/ml/test_nystrom_search_baseline.py` · plan `docs/plans/2026-09-nystrom-downdate.md` §10 · D-27.
- **Design.** A variant, not the algorithm. Exact `AStarSearch` compares nothing it did not compute itself
  and stays the reference; `PrunedAStarSearch` is the one that may be handed a number from elsewhere, and
  the "unchanged" tests measure it against the base class on state, cost and expansion count rather than
  against brute force alone. Pop order is preserved exactly, not approximately: every child still consumes
  an insertion index whether or not it is pushed, so the entries that survive pop in the order they always
  did. The incumbent is the goal's *bound*, which the contract makes its exact
  objective, so no extra `goal_cost` call is paid per goal child. Ties are kept on both mechanisms
  (strict inequalities), which is what keeps the optimum set the same; the frontier cap counts entries,
  not bytes, so its trigger is the same on every platform.
- **What was confusing.** The incumbent seed. The plan says a child "strictly above" the incumbent is
  pruned, and that is exactly right when the incumbent is one of the search's own goal bounds. It is not
  quite right for a seed priced somewhere else: the greedy selector prices its subset through
  `goal_cost` (one projection), the search prices the same subset through the goal-depth batch (a Schur
  complement), and on the six-point chain, where greedy is already optimal, the two differed in the
  fifteenth digit with the seed on the low side. Strict pruning dropped the optimum and the search
  raised `IncumbentBelowOptimum` against a seed that was, in exact arithmetic, an upper bound. The fix is
  the asymmetry of the problem: keeping a child is always safe and only pruning can be wrong, so the
  threshold carries a relative slack of 1e-9 (`INCUMBENT_RELATIVE_SLACK`) and the entries it keeps are the
  ones within rounding of the incumbent. Then CI showed the second half of the same lesson: on the
  all-ones kernel the optimum is zero, the greedy seed is 8.9e-16 and every goal bound is 1.8e-15, and a
  relative slack on a noise-level number is nothing. Rounding on a residual trace scales with the trace,
  not with the result, and the search cannot know the trace; so the caller states an absolute allowance
  beside the seed (`incumbent_slack`, 1e-12 of the trace for Nyström). The general lesson is D-24's "same
  value both paths, up to rounding" read from the other side: two paths to the same number are equal for
  reporting and not equal for a strict comparison, and "up to rounding" has a scale that only the
  computation knows.
- **Numbers (this machine, SPECTF, bandwidth scale 4).** n = 60, k = 5: frontier peak 3,794,117 before and
  452,830 after (442,389 with the greedy residual trace as seed), 328,579 expansions and the same landmarks
  in every run. n = 80, k = 5, the EXP-01 worst-cell shape: 1,488,433 entries after, 1,546,225 expansions;
  the plan's estimate for the same cell before was 24 million. The search baseline and the committed
  example output are byte-identical.
- **Reference.** Arai, Maung & Schweitzer, AAAI 2015, §5 (the (k̄ + 1)·f rule this slice does *not*
  build, and D-27's reason); Russell & Norvig, *AIMA*, §3.5 for the termination argument the filter and
  pruning rest on.

## Consistency of the spectral bound: measured on S1, then explained · 2026-09-06
- **What.** A* needs an *admissible* bound to be a proof; a *consistent* one (Hart, Nilsson and Raphael's
  monotone restriction: a child's bound never below its parent's) is a second property that a
  terminal-only objective does not need for correctness, since with g ≡ 0 the priority f = h is
  path-independent and duplicates are detected by state (D-23). It matters for two other things: whether
  the frontier minimum — the anytime gap of X1.1 — can move backwards mid-run, and whether a tight root
  bound predicts pruning. Whether the Nyström spectral bound has it was unverified, so it was measured:
  an opt-in counter on the engine compares every child bound with the popped parent bound, as priced by
  the batched path the search actually runs (D-24), and a grid run over the S1 cells recorded the counts.
- **Where.** `math/algorithms/a_star_search.py` (`BoundDropCounter`, `count_bound_drops`,
  `bound_drop_slack`, `bound_drops` on the instance, `BOUND_DROP_ROUNDING_TOLERANCE`), passed through by
  `PrunedAStarSearch`; tests in `tests/math/algorithms/test_bound_drop_counter.py`; the runner and result
  file live research-side (`~/Desktop/BotMaker/nystrom-grid/run_monotonicity.py`,
  `nystrom/data/monotonicity-S1-v1.jsonl`).
- **Design.** The counter lives on the engine, not the cost function, because only the engine holds the
  parent's bound as it was pushed; recomputing it through the oracle would compare two arithmetic paths
  and count their disagreement as drops. Depth is recovered without storing anything per state: the
  children of one expansion take one contiguous run of insertion indices (the `_push_children` contract),
  so two arrays with one entry per expansion and a bisection on the popped index give the depth. Rounding
  is named, not ignored: a strict drop is counted, and a drop is called the bound's only when it exceeds
  1e-9 of the parent plus an absolute slack the caller states from the objective's scale — the same shape as
  the incumbent threshold, for the same reason: a parent bound that is itself rounding noise makes any
  drop below it look total, and only the trace knows what "small" is.
- **Numbers.** Read from `nystrom/data/monotonicity-S1-v1.jsonl` (research side; grid runner `run_monotonicity.py`, botMaker at this PR, `PrunedAStarSearch` seeded from the greedy residual trace, the counter on with a slack of 1e-12 of the trace). At δ = 0, all 2,380 S1 cells: 4,471,163,589 child bounds priced, **0 strictly below their parent** — not one, not even at rounding level — so 0 of 2,380 cells with a drop, worst relative drop 0, no depth to report; every cell returned S1's certified cost. At δ = 1e-4, the 665 S1 cells where that tolerance actually drops modes (retained rank between 0.41 n and n, median 0.95 n): 389,805,416 child bounds priced, again 0 strict drops, and the same optimum cost as at δ = 0 on every cell. Drops concentrate on neither flat spectra nor ties, because there are none: every top-k trace-share bin from [0, 0.2) to [0.8, 1] reads 0, and the 20 brute-force cells with a tied optimum read 0. The counter's rounding allowance was never needed; the batched path (D-24) keeps the inequality exactly. A side observation: 33 of the 2,380 cells returned a different landmark set from S1 at a bit-identical cost and expansion count, all on movement_libras, each swapping a landmark for a duplicate row of the dataset (32 bit-identical kernel columns, one differing by an ulp) — a tie broken the other way by the downdated arithmetic, not a change in the search.
- **Why the measurement comes out this way.** In the reduced coordinates the parent's residual Gram is
  H ⪰ 0 and a child that adds column j has the residual Gram H − zzᵀ with ‖z‖² the column's residual
  energy. The parent's bound is tr H − (sum of the top r eigenvalues of H); the child's is
  (tr H − ‖z‖²) − (sum of the top r − 1 eigenvalues of H − zzᵀ). By Ky Fan's maximum principle the top-r
  sum of H is the maximum of tr(PH) over rank-r orthogonal projections P; taking P onto the span of z and
  the top r − 1 eigenvectors of H − zzᵀ gives tr(PH) = tr(P(H − zzᵀ)) + ‖z‖² ≥ (top r − 1 sum of H − zzᵀ) + ‖z‖²,
  the inequality because H − zzᵀ ⪰ 0. Rearranged, child bound ≥ parent bound. At goal depth the child's
  value is its exact cost on K, and D-26's admissibility gives the same inequality against a parent bounded
  on the truncated K̃, so the argument covers every δ. The measurement was still the deliverable: the
  argument is exact arithmetic, and the counter is what says the floating-point path keeps it.
- **Reference.** Hart, Nilsson and Raphael 1968 (the consistency condition); Pearl, *Heuristics* 1984, §3.1
  (monotone heuristics and why A* never reopens under them); Fan, *PNAS* 1949 (the maximum principle);
  Horn and Johnson, *Matrix Analysis*, Cor. 4.3.39 (Ky Fan in the form used).

## Anytime A\* with the a-posteriori gap: a stopped search still proves something · 2026-09-07
- **What.** Best-first search with an admissible bound keeps two numbers at every moment: the best
  complete solution it has seen (the incumbent, an upper bound on the optimum) and the smallest bound on
  its frontier (a lower bound: no completion of any unexpanded state can cost less). A\* proves
  optimality when the two meet, at the pop of a goal. Stopped earlier, by a memory cap or an expansion
  budget, the pair still brackets the optimum, and `incumbent − frontier_min` is an additive certificate
  on the incumbent — the branch-and-bound gap, made an output. The pass-2 entry ticket.
- **Where.** `math/algorithms/anytime_a_star_search.py` (`AnytimeAStarSearch(PrunedAStarSearch)`,
  `max_expansions`, `incumbent_seed_state`, `certified_gap`, `frontier_min`); `SearchResult.certified_gap`
  (D-28 amended); the harness's `bounded` label and reference guard in `ml/projects/nystrom_uci_harness.py`;
  tests in `tests/math/algorithms/test_anytime_a_star_search.py` and the anytime section of
  `tests/ml/test_nystrom_search_baseline.py`.
- **Design.** A variant over the pruned engine, which already holds the incumbent, the cap and the
  goal-sibling filter; it adds the incumbent's *state* (the pruned engine keeps only the cost) and the
  expansion budget, and turns both stops into a result. The base loop is not copied: the caps raise from
  `_push_children`, where the engine already learns the batch and the heap, and `_search` catches the
  stop and builds the bounded result — one private exception instead of a second copy of the loop and
  its bound-drop bookkeeping. The frontier minimum is read at the stop: the heap's top at an expansion
  cap, and at a frontier cap the minimum over the heap *and every child of the interrupted batch*, since
  those children are on the frontier whether or not they were pushed. Neither reading assumes the
  bound is monotone; monotonicity (BL-33) is what makes the gap non-increasing over a run, which the
  doubling-cap test pins on a real cell.
- **What was confusing.** Whether a stop whose gap clamps to zero should say `optimal=True`. In exact
  arithmetic the evidence is the same as a popped goal's: every remaining bound is at or above the
  incumbent. But the incumbent may be a seed from another arithmetic path whose state the engine cannot
  see, and the pruned engine never returns a seed as the answer either; it waits to pop a goal. So the
  flag means one thing only, "a goal was popped as the frontier minimum", and a zero gap at a stop is left
  for the reader to draw the conclusion from. The other confusion was the frontier cap: `max_frontier=1`
  bites on the *second* push of the root's batch, so one child is on the heap and the rest are not, and
  reading only the heap would overstate the frontier minimum by the batch's spread.
- **Numbers.** Uncapped, the variant expands exactly what exact A\* expands on all 13 snapshot cells,
  seeded and unseeded, and returns the same landmarks and residual trace. SPECTF n = 60, k = 4 (15,740
  expansions to certify): cut at 1, 1,574 and 7,870 expansions with the greedy seed, the incumbent never
  falls below the certified optimum, the frontier minimum never exceeds it, and the gap never understates
  the incumbent's true distance. SPECTF n = 40, k = 3 at caps 1, 2, 4, …, 1024: the gap is non-increasing
  and reaches 0.0 when the goal pops.
- **Reference.** Hansen & Zhou, *JAIR* 2007 (anytime heuristic search; the incumbent–bound gap as the
  quality guarantee); Land & Doig 1960 and Lawler & Wood 1966 (the branch-and-bound bracket); Pearl,
  *Heuristics* 1984, §3.1 (why the frontier minimum never falls under a monotone bound).

## Tie-breaking on a plateau: 2^k was the heap order, not the search · 2026-09-07
- **What.** With a terminal-only objective the priority is the bound alone, and on nearly rank-k data
  many states share the optimum's bound: a plateau. A heap that breaks ties first-in-first-out expands the
  plateau level by level and pops a goal only when the levels above it are exhausted — the "exact A\*
  enumerates 2^k" of the reproduction. Preferring the deeper state among equals walks one path to a goal
  and certifies it after k + 1 expansions, because on a plateau at the optimum every child of a plateau
  state is at or above the optimum and the goal's bound is its cost (Dechter & Pearl's remark that below
  the optimum no tie-break helps, and at it any depth-first one does).
- **Where.** `math/algorithms/a_star_search.py` (`tie_break`, `tie_tolerance`, `_heap_entry`,
  `_frontier_minimum`; the goal-pop certificate under a tolerance); the pruned and anytime engines push
  through `_heap_entry` and inherit both knobs; tests in `tests/math/algorithms/test_tie_break.py` and the
  tie-breaking section of `tests/ml/test_nystrom_search_baseline.py`. D-29.
- **Design.** The heap entry became `(bound key, tie-break key, insertion index, bound, state)`; with the
  defaults the key is `(bound, 0, index)`, which sorts exactly as the old `(bound, index)`, so the default
  path is byte-identical without a branch. Near-ties are made equal by quantising the bound key to the
  tolerance grid, an integer cell, rather than by pairwise comparison, which is not a total order. The
  raw bound rides along because under a tolerance the heap's top is only the smallest cell, and the
  certificate needs the smallest raw bound: one scan of the heap at the goal pop, never per expansion.
- **What was confusing.** What `optimal` means once ties have a width. The popped goal is within the
  tolerance of the frontier minimum by construction, so "certified within τ" is always true and says
  nothing. The honest statement is the one the engine can check: exact when the goal's bound is at or
  below every remaining raw bound, else `optimal=False` with the gap it did not look under, which PR 1's
  `certified_gap` already carries. So the flag kept its one meaning and the tolerance became an additive
  certificate, as C9's wording predicted. The reviewer then found the second half: with ties a cell wide,
  a goal can pop *before* a cheaper goal in the same cell that the pruned engine already holds as its
  incumbent, so the engine would return a worse answer than one on its own heap. The incumbent's state
  is now kept beside its cost, and an engine that holds one returns it when the popped goal is worse.
- **Numbers.** Toy plateau (equal weights, n = 2k): first-in-first-out expands 2^k − 1 or more, deeper-
  first exactly k + 1, for k = 3, 4, 5. On the snapshot's plateau fixtures (identity and all-ones at
  k = 3, the palindromic RBF chains) deeper-first at tolerance zero certifies the same optimum in no more
  expansions than first-in-first-out; on the SPECTF cells, which have no exact ties, the counts are
  identical (recorded in the PR). Nothing changes on any cell under the defaults.
- **Reference.** Dechter & Pearl, *JACM* 1985 (tie-breaking and the optimality of A\*); Asai & Fukunaga,
  *JAIR* 2017 (tie-breaking strategies for cost-optimal search, the depth-based rules); Xu, Yan & Chang,
  *ICPR* 1988 (best-first branch and bound for feature selection, the plateau in this problem's family).

## Conditional solves on the same tree: forced and forbidden columns · 2026-09-07
- **What.** A certified optimum says which subset is best; it does not say which of its members
  matter. The necessity margins do: for every column j, the best subset that must contain j and the best
  that must not, both against the unconstrained optimum. If forbidding j costs more, j is in every
  optimum; if forcing j costs more, it is in none; the two differences are j's signed margins. Each is
  the same search on a modified ground set — the SAT community's backbone test, LOCO in statistics,
  reduced-cost fixing in integer programming, all the same move.
- **Where.** `math/graph/nystrom_landmark_problem.py` (`forced`, `forbidden`, `constraints`,
  `initial_state`, `successors`, `_added_columns` in the cost function); the UCI harness's
  `problem_constraints`; tests in `tests/math/graph/test_nystrom_constrained_problem.py`; the driver
  `nystrom/grid/run_margins.py` research-side.
- **Design.** The bound is not touched, and that is the point of doing it on the problem rather than on
  the cost: forbidding removes candidates, which can only raise the true optimum under a bound that never
  overestimates; forcing starts the search at a state of the same tree, whose subtree the bound already
  covers. Canonicality had to be re-derived: with a forced column the state is no longer a prefix-extended
  tuple, so the successor cursor runs over the *free* columns (neither forced nor forbidden) above the
  largest free column chosen, and the child is re-sorted. Without constraints the free columns are every
  column and the rule is the old `range(state[-1] + 1, n − still_needed + 1)` to the byte. The batched
  pricing assumed a child was `parent + (column,)`; it now finds the one column a child adds wherever it
  sorts, which is what keeps the forced case on the fast path.
- **What was confusing.** Warm starts. The unconstrained optimum is a *lower* bound on a conditional
  optimum, never an incumbent; an incumbent must be feasible for the constrained problem. So the driver
  seeds the forbid-j solve with the unconstrained optimum only when j is not in it, and otherwise (and for
  every force-j solve where j is new) runs the greedy rule on the constrained ground set first.
- **Numbers.** Read from `nystrom/analysis/margins_krause_k4.json` (research side): the 52-mote Krause
  covariance at k = 4, 1 + 104 certified solves, brute force agreeing on every conditional cost.
- **Reference.** Krause, Singh & Guestrin, *JMLR* 2008 (the covariance and the sensor-placement
  framing); Kilby, Slaney, Thiébaux & Walsh, AAAI 2005 (backbones); Lei, G'Sell, Rinaldo, Tibshirani &
  Wasserman, *JASA* 2018 (LOCO); Fisher, Rudin & Dominici, *JMLR* 2019 (model class reliance, the
  "in every / some / no good model" framing the margins borrow).

## Two-hot spanning sets: the rounded cut is a RatioCut · 2026-09-10
- **What.** The graph objects behind a continuous relaxation of ratio-cut clustering. A spanning set
  is an n × r matrix V, r = n − K, whose columns are only ever asked to be zero-sum; its quality is
  the projector residual E(V) = ‖X − V V⁺ X‖²_F against the incidence matrix X, whose column for edge
  (i, j) is √w (e_i − e_j) so that X Xᵀ is the unnormalized Laplacian. Rounding a column to
  e_argmax − e_argmin turns V into a set of vertex pairs; the connected components of that pair graph
  are the clustering, and Ê = E(round(V)) is exactly Σ_B cut(B, B̄)/|B| of that partition. The floor
  under both is Σλ, the sum of the K smallest eigenvalues of the Laplacian.
- **Where.** `src/mllib/math/graph/two_hot_span_problem.py` · test
  `tests/math/graph/test_two_hot_span_problem.py` · plan `docs/plans/2026-09-two-hot-span.md` § 3
  (slice 2.1).
- **Design.** VᵀV = I is never imposed, which is the whole point of the family: orthonormality is
  what forces the classical spectral relaxation to be rounded by a *separate* algorithm (k-means,
  discretize, cluster_qr), and dropping it lets the rounding be the trivial one. Because V may go
  rank-deficient, the projector is the pseudo-inverse one everywhere in this module — a reduced QR
  would silently return an n × r Q whatever the rank and over-span, which is a different function.
  The oracle for small graphs is a restricted-growth-string enumeration of set partitions, guarded at
  n ≤ 10, not the A* engine: the search machinery buys nothing when the answer is a Bell number of
  labellings. The test checks it against a second, deliberately naive enumeration written inside the
  test file (itertools.product, keep the surjective labellings), so the same mistake would have to be
  made twice in two shapes. One asymmetry had to be designed in: a *constant* column (the zero column
  included) has equal largest and smallest entries and no pair to read off, so it rounds to the zero
  column and contributes no edge — writing +1 and −1 into the same cell would leave a 1-hot column that
  is not zero-sum, and Ê would stop being the RatioCut of the pair graph's components.
- **What was confusing.** The word "spanning forest" in "Ê = RatioCut when the rounded pairs form a
  within-block spanning forest". The forest lives on the *pair graph*, not on the input graph: its
  edges are arbitrary vertex pairs and need not be edges of G at all. Once that was clear the identity
  test wrote itself — draw a partition, draw a random tree on each block's vertex set, and the ± scaled
  two-hot columns of those n − K pairs round back to exactly that partition. The random per-column
  scales are there on purpose: rounding is scale-invariant, and a test with unit columns would not say so.
- **Reference.** Guattery & Miller 1998 for the roach graph (n = 4k, m = 5k − 2; one antenna alone is
  the K = 2 RatioCut optimum at 4/(3k) — brute force at k = 2, exhaustive 2^19 bipartition scan at
  k = 5 — while the best *balanced* cut, antennae vs. ladder at 2/k, is the one spectral bisection is
  famously wrong about); von Luxburg 2007 for the RatioCut/NCut
  relaxation this departs from. ncut is out of scope here and parked as BL-41.
