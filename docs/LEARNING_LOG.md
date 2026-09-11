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

## Gradient descent on a projector residual with a collision reward · 2026-09-10
- **What.** The first gradient-optimized module in the repo. A trainable W (n × r) is projected every
  forward pass to a zero-sum spanning set V, and Adam minimises E_ridge(V) − λ Σ_j R(v_j): the span
  quality of the previous entry, minus a reward for each column being 2-hot. λ is the only thing
  pushing a smooth relaxation towards the discrete answer; every number the run reports comes back
  through the numpy functions of slice 2.1, so a report never depends on a training knob.
- **Where.** `src/mllib/math/algorithms/two_hot_span_optimizer.py` · test
  `tests/math/algorithms/test_two_hot_span_optimizer.py` · plan `docs/plans/2026-09-two-hot-span.md`
  § 3 (slice 2.2) · decision **D-31** (the torch dependency group and the two projectors).
- **Design.** The zero-sum constraint is handled by *reparameterisation*, not by a penalty and not by
  a projection after the step: the parameter is W and the model sees V = W − 1(1ᵀW)/n, so every
  iterate is feasible by construction and `max_zero_sum_violation` is a check on the arithmetic
  rather than on the optimiser. Writing the ones vector out (instead of calling a mean) is what makes
  the ncut generalisation a parameter change later — c = √d replaces 1 and nothing else moves.
  Training uses the ridge projector V(VᵀV + εI)⁻¹Vᵀ, whose filter factors σ²/(σ² + ε) stay smooth as
  columns go dependent, while reporting uses the exact pinv projector (D-31, P-2). QR is not the
  third option it looks like: reduced QR hands back an n × r Q whatever the rank of V, so QQᵀ
  over-spans a rank-deficient V and is simply a different function, not a stabler spelling of the
  same one.
- **What was confusing.** Two things.
  (1) Adam's step size does not vanish with the gradient. Initialised at the spectral floor attainer —
  where E is exactly Σλ and the gradient is numerically zero — 20 steps still drift E\* off the floor
  on roach G_5 by 3.01e-3 at lr 0.05, 2.43e-4 at lr 0.01, 2.09e-5 at 1e-3, 1.79e-7 at 1e-4, 7.19e-10
  at 1e-5 and 4.65e-12 at 1e-6. The normalised update m̂/(√v̂ + ε) is O(1) whatever the gradient's
  size, so the drift scales like lr · steps, not like ‖∇E‖. A test asserting "the floor is held to
  1e-8" is therefore a statement about the learning rate; the test says so and pins the three
  regimes (≤ 1e-12 at step 0, ≤ 1e-8 at lr 1e-5, ≤ 1e-2 at lr 0.05) instead of pretending one bound
  covers them.
  (2) λ does its job and still does not deliver K blocks. A λ × init × lr probe on roach G_5 (K = 2)
  drives mean R(v_j) from ≈ 0.08 to ≈ 0.49, i.e. the columns really do become 2-hot — and yet the
  component count of the pair graph only falls from 9 to 3 across the sweep and never reaches 2. The
  reason is that being 2-hot is a per-column property and nothing in the objective says *which* pair
  a column should choose: across the twelve rows 14–18 of the 18 rounded pairs are distinct, so
  duplication is minor, but only 1–5 of them are edges of the roach and 5–14 join a vertex on one
  path to a vertex on the other. The pair graph's components therefore ignore the graph's own
  structure (the best row's blocks are {0, 9, 14, 19}, {10} and the other fifteen), the few
  duplicates and cycles waste columns, and the vertices no pair reaches survive as singleton
  blocks. Best Ê over the whole sweep was 2.65
  against a RatioCut optimum of 4/15 ≈ 0.267 — Ê ≥ the optimum held everywhere, which is the
  invariant the tests assert, but the relaxation is not yet competitive. A column-diversity term (or
  the parked alternating solver) is where that goes next.
- **Reference.** Kingma & Ba 2015 for Adam (§2.1 is where the normalised step comes from); Hoyer 2004
  for the sparseness-constrained projection named as the fallback in the plan's risk table.

## Three roundings of a spectral embedding, by hand · 2026-09-10
- **What.** The datum the two-hot span prototype is judged against: the spectral relaxation of the
  unnormalized Laplacian — the K eigenvectors of its K smallest eigenvalues, an n × K embedding —
  turned into a labelling three different ways. (1) k-means, as Lloyd's algorithm with k-means++
  seeding and ten restarts, lowest inertia wins. (2) Yu & Shi's *discretize*: alternate an argmax
  assignment of the row-normalised embedding against a rotation, and a re-derivation of the rotation
  from the SVD of the indicator against those rows, until the sum of singular values stops moving.
  (3) Damle–Minden–Ying's *cluster_qr*: a column-pivoted QR of the embedding transpose picks K
  representative rows, an SVD of those rows gives a rotation, and each vertex takes the argmax of
  the absolute rotated coordinate — no iteration and no seed at all.
- **Where.** `src/mllib/ml/projects/two_hot_span_harness.py` (the datum half: `spectral_embedding`,
  `round_kmeans`, `round_discretize`, `round_cluster_qr`, `datum_roundings`) · test
  `tests/ml/test_two_hot_span_datum.py` · plan `docs/plans/2026-09-two-hot-span.md` § 3 (slice 2.3).
- **Design.** Hand-written numpy, for two reasons that are not the same reason. The library imports
  no sklearn model class, and `KMeans` is a model class; and scipy is not a declared dependency and
  the library does not import it (it is installed transitively by scikit-learn), so
  `scipy.linalg.qr(pivoting=True)` is not there to reach for and the pivoted QR is written as a K-step
  modified Gram–Schmidt that picks the largest remaining column norm at each step. That is all a
  pivoted QR *is* for this purpose — only the pivot indices are wanted, never the factors. The
  sklearn implementations of all three are kept, but only as oracles inside the tests, behind
  `importorskip`, and each is compared on the quantity its own rounding optimises (labels for
  cluster_qr, the Yu–Shi objective for discretize, inertia for k-means) rather than on a number.
  The datum functions live in the harness module and use numpy only, so the harness imports the
  torch optimizer inside `run_graph`: the datum tests then run in the default suite without the
  optional torch group (D-31).
- **What was confusing.** Two things.
  (1) sklearn's `discretize` row-normalises the embedding and flips signs, and neither operation
  changes the answer — both cancel in the iteration, because the objective only ever sees
  `indicator.T @ rows` through its singular values. Chasing agreement on the *labels* was therefore
  the wrong test: sklearn seeds its rotation from a legacy `RandomState` and this from a
  `Generator`, so the two alternations settle in different local optima of the same objective. The
  test compares the best objective over a ten-seed sweep, which agrees to 1e-6.
  (2) The roach graph's K = 2 RatioCut optimum is not the balanced cut. It is one antenna alone —
  one edge, 1/k + 1/(3k) = 4/15 ≈ 0.2667 at k = 5 — and k-means and discretize both find it. The
  balanced antennae-vs-ladder cut is 2/k = 0.4, and cluster_qr returns top-path-vs-bottom-path at
  1.0, which is exactly the spectral-bisection failure Guattery & Miller documented the graph to
  exhibit. So the three roundings of one embedding disagree by a factor of 3.75 on the same graph,
  and "the spectral datum" is not a single number: which rounding was used has to be reported.
- **Reference.** Yu & Shi 2003 (*Multiclass spectral clustering*) for the discretisation; Damle,
  Minden & Ying 2019 (*Simple, direct and efficient multi-way spectral clustering*) for cluster_qr;
  Arthur & Vassilvitskii 2007 for k-means++; Guattery & Miller 1998 for the roach.

## Observing a search without changing it: the injected recorder · 2026-09-10
- **What.** A best-first search that can be watched, frame by frame, without the watching being
  visible in what it returns or in what it costs when nobody is watching. The technique is the
  observer done as a constructor argument rather than a subscription list: one collaborator,
  injected, defaulting to a null object, called behind a guard on a class attribute at the loop's
  two moments — every state it advances through (`record_expansion`) and the goal it stops at
  (`record_goal`). Two, not one: a goal is popped and never expanded, so a run recorded only at
  expansions would end one step before its own answer. The engine hands over the raw objects it already holds — the expanded state
  and its bound, the priced children, the heap, the expanded set's size — and knows nothing about
  what a reader will want from them.
- **Where.** `math/recorder.py` (`Frame`, `AbstractRecorder`, `AbstractSearchRecorder`,
  `NullRecorder`) · the two call sites and `_recorder_extras` in `math/algorithms/a_star_search.py`,
  overridden in `pruned_a_star_search.py` · `visualization/recorders/astar_landmark.py`
  (`AStarFrame`, `AStarRecorder`) · tests `tests/math/test_recorder.py` and
  `tests/visualization/recorders/test_astar_landmark.py`.
- **Design.** Three choices, each of which had a tempting alternative. (i) *The guard, not the
  no-op.* A null recorder whose `record` quietly returns would let the engine call it every
  expansion and cost a method call plus a conversion per state; `if self.recorder.enabled:` costs
  one attribute read, so everything inside — including building the pushed-or-pruned flags and
  copying the heap — is work that only happens when someone asked for it. The same guard sits at
  the goal return, where the cost is now computed once into a local and used by both the frame and
  the `SearchResult`. The null recorder then
  *raises* rather than returning, because the only way it can be called is a missing guard, and a
  silent no-op would hide that behind a search that still worked. (ii) *The child extracts.* The
  base declares `record_expansion`'s signature in `math` so the engine can be typed against it, and
  the per-problem child in `visualization` decides which fields matter and writes the caption. Had
  the engine built the frame, adding a field to a picture would have meant editing the search. One
  frame type serves both moments — a goal frame is an `AStarFrame` with `children = ()`, `goal =
  True` and its `bound` holding the goal's exact cost, which is the same number by D-23 — so the
  view reads one dict shape from the first frame to the last instead of branching on a type. The
  pruned engine's tie-tolerance branch (D-29) forced one more: when it pops one goal and returns
  the incumbent it already holds, it records a *second* goal frame naming the superseded goal, or
  the run's last frame would name a state that never came back.
  (iii) *Never on the result.* `SearchResult` carries what the search paid, how it was set up and
  what it proved (D-28); frames are none of those, so they stay on the recorder the caller
  constructed and still holds. The variants contribute their own state through `_recorder_extras`,
  which is why no variant overrides `_search`.
- **What was confusing.** Two things. First, which children were pushed: `_push_children` returns
  only an insertion index, and a pruned variant pushes fewer than it priced. The answer was already
  in the contract — every child takes one insertion index whether it is stored or not, and the run
  is contiguous — so `insertion_index - len(children) + 1` gives the first child's index and the
  indices present in the heap say which survived, with no new bookkeeping and no change to the
  default path. Second, `super()` inside a `slots=True` dataclass: the decorator rebuilds the class
  after the method's `__class__` cell is bound, so a zero-argument `super()` in `to_dict` resolves
  against the discarded class and raises. Naming the base explicitly (`Frame.to_dict(self)`) is the
  fix, and the same trap waits for every frame subclass.
- **Reference.** None needed.

## A run as a document, and a document as a page: the offline walkthrough · 2026-09-10

- **What.** The second half of the recorder seam: a **recording** — a versioned JSON document
  holding one run's frames, the problem it solved, the engine's configuration and the result — and a
  **walkthrough**, that document rendered as a single self-contained HTML page with a stepper. The
  page has no build step, no dependency and no network: it is a template, two inlined scripts and
  two inlined JSON blocks, and it works opened from `file://`.
- **Where.** `visualization/recording.py` (`Recording`, `save`, `load`, `SCHEMA_VERSION`) ·
  `visualization/html_renderer.py` with `template.html` and `walkthrough.js` ·
  `visualization/views/` (`View`, `view_for`, and `astar_landmark.py` with its sibling
  `astar_landmark.js`) · the CLI `visualization/render.py` ·
  `examples/astar_landmark_walkthrough.py` (the composition root) · the committed recording
  `tests/visualization/fixtures/astar_landmark_rbf_chain_8x8_k3.json` and the tests under
  `tests/visualization/`.
- **Design.** Four choices worth naming. (i) *A document, not an object graph.* The recording is
  plain JSON with exact floats, sorted keys and no timestamp anywhere, which is what lets one be
  committed and diffed like a baseline snapshot: a changed line means the search changed, never that
  the clock moved. The cost is that a recording cannot say when it ran; its identity is `problem`
  plus `configuration` instead, which is the pair that actually decides whether two runs are
  comparable. (ii) *The chrome is generic, the drawing is not.* The template owns everything a
  walkthrough has whatever it recorded — previous, next, slider, play with a speed, arrow keys and
  space, the caption, the raw-fields panel, the footer — and a **view** owns the picture. A view is
  a JS function registered under the problem's kind plus a `layout` dict Python precomputed for it,
  so the browser never re-derives what numpy already knew, and `view_for` dispatches on
  `problem["kind"]` and *raises* on an unknown one rather than drawing a generic picture that would
  be wrong in a way nobody notices. (iii) *The scale is the run's, not the frame's.* Frontier bars
  are drawn against the smallest and largest bound anywhere in the run. Scaling each frame to its
  own extremes would have made every frontier look identical and hidden the one thing the bars are
  for — watching the bounds climb as the cheap states are consumed. A faint full-width track behind
  each bar keeps a genuinely cheap state from reading as a bar that failed to draw.
  (iv) *One frame shape, including the goal.* The goal frame slice 1 records carries the same keys
  with `goal: true` and no children, so the view branches on a boolean (a banner instead of a
  children panel) rather than on a second document shape.
- **What was confusing.** Three things, all about the page being genuinely offline. First,
  `createElementNS` needs the SVG namespace *URL*, which would have been the only outward-looking
  string in a file that is supposed to reference nothing; assigning `svg.innerHTML` instead lets the
  HTML parser supply the namespace, so the view builds markup strings and the page contains no `http`
  at all. Second, embedding the recording: a JSON block is inert, so a caption is data however it is
  spelled, but `</` still ends a `<script>` element — and `\/` happens to be a legal JSON escape for
  `/`, so replacing the pair leaves a document that parses back to *exactly* the same values. The
  test for it renders a caption containing a real `</script><script>` and checks the parsed document
  still equals the original. Third, the substitution is `str.replace` and not `str.format` or an
  f-string, because the template is mostly CSS and CSS is mostly braces.
- **Reference.** WHATWG HTML, § "script data state" (only `</script` closes the element) and
  RFC 8259 § 7 (`\/` as an escape for `/`).

## Watching an optimizer: light and full frames · 2026-09-10

**What.** The second recorder contract in the library, and the first one for something that does not
search. `AbstractStepRecorder` (`src/mllib/math/recorder.py`) declares two moments — `record_step`
once per completed optimizer step, `record_end` once when the run is assembled — beside the
`AbstractSearchRecorder` that was already there. `fit_two_hot_span` takes an optional recorder,
guards both call sites with `if recorder.enabled:`, and defaults to the `NullRecorder`, which now
raises loudly on both of the new methods for the same reason it raises on `record_expansion`: a null
recorder that is handed a frame is a missing guard, and a silent no-op would hide it. The child that
knows what a step *means* is `mllib.visualization.recorders.two_hot_span`, and the composition root
is `examples/two_hot_span_walkthrough.py`.

**Where.** `src/mllib/math/recorder.py` (`AbstractStepRecorder`, two more raises on `NullRecorder`),
`src/mllib/math/algorithms/two_hot_span_optimizer.py` (the two call sites),
`src/mllib/visualization/recorders/two_hot_span.py` (`TwoHotSpanFrame`, `TwoHotSpanRecorder`),
`examples/two_hot_span_walkthrough.py`, `tests/visualization/recorders/test_two_hot_span.py`,
`tests/visualization/test_two_hot_span_example.py` and the committed recording
`tests/visualization/fixtures/two_hot_span_roach_g5_30steps.json`.

**Design.**

*Why the child does the pinv work.* The training loop holds W and computes V and a ridge loss; that
is all it needs to take a step. Every number a reader of a walkthrough actually wants — E\* through
the exact pseudo-inverse projector, Ê through the rounded columns, R(v_j) per column, the pair graph
and its components — costs a pseudo-inverse or a rounding that the optimizer never performs and must
never be made to perform. Putting them behind the guard in the engine would have been the same
arithmetic in the wrong place: `math` would have learned which of its intermediate quantities a
picture wants, and a run that was watched would have been a different run from one that was not, in
cost if not in numbers. So the engine hands over exactly the three things it already holds — the
step number, the training loss it just differentiated, and V as a plain float64 array — and the
recorder in `visualization` derives the rest, calling the same `two_hot_span_problem` functions the
run's own reported fields come back through. The test that matters here is not about frames at all:
a run with an explicit `NullRecorder` gives a bit-identical `spanning_set` and `loss_history` to a
run with no recorder argument.

*Light vs full frames.* A loss curve wants a point at every step; a picture of V wants a handful.
Recording V at all three hundred steps of the karate run writes a 12 MB document, which is not a
walkthrough, it is a memory dump with a stepper on it. But dropping the off-cycle steps entirely
would lose the curve, which is the one thing a reader watches continuously. So every step leaves a
frame and `frame_every` decides which kind: a light frame carries the step and its training loss, a
full frame carries those plus every derived number and V itself. The last step is always full
whatever `frame_every` divides — a walkthrough whose final picture was four steps stale would show a
V the run never reported on — and `record_end` adds one more, flagged `end`, restating the run's own
reported E\*, Ê and labels rather than recomputing them. Both kinds serialise to the same dict shape,
with the light one's derived fields `null`, so the view that will read this reads one shape from the
first frame to the last instead of branching on which kind it got.

*Training loss is never reported as E.* D-31 settles that the training loss goes through the ridge
projector `V(VᵀV + εI)⁻¹Vᵀ` and carries the collision reward `-λ Σ_j R(v_j)`, while every reported
number goes through the exact `pinv`. The two are not comparable and the training loss is not even
positive — the roach at λ = 10 trains down to about -40 while its E\* is 0.41. A frame that labelled
that "E" would be a walkthrough that teaches the wrong thing at every step, so the field, the dict
key and the caption all say `training loss`, and a test asserts no caption ever pairs the two words.
The captions keep them in one sentence but never in one quantity: "Step 29: training loss -39.9302;
E\* 0.4056 (floor 0.0713), Ê 2.9000, 3 components; mean R(v_j) 0.2259, min 0.1285."

**What was confusing.**

`NullRecorder` is declared above the new abstract class in the same module, so it cannot name it as
a base, and the alternative — a second null recorder, one per contract — would give the library two
objects both meaning "not watching" and make a caller check which engine it was about to hand one
to. `AbstractStepRecorder.register(NullRecorder)` at the end of the module is what makes the off
recorder an instance of both contracts; it took a minute to remember that `ABCMeta.register` exists
for exactly this and that a virtual subclass satisfies `isinstance` without satisfying a static type
checker.

The recorder needs `step_count` in its constructor, which felt redundant when `record_step` is told
the step number every time — until the "always full on the last step" rule, which cannot be decided
from a step number alone: the recorder does not learn a run is over until `record_end`, by which
time the frame it should have made full is already recorded.

`slots=True` on a frozen dataclass still breaks zero-argument `super()`, so `TwoHotSpanFrame.to_dict`
calls `Frame.to_dict(self)` by name, the same as `AStarFrame` — a second encounter with the same
sharp edge, and worth writing down twice.

## Replacing an animation with a recording: BFS and DFS on the recorder · 2026-09-10

*Learning-log entry for BL-43 slice 4 (`feat/visualization-graph-search`). Written to the scratchpad,
not to `docs/LEARNING_LOG.md`; Xavier appends it.*

The library's oldest picture was `math/graph/visualizer.py`: a matplotlib figure, `plt.pause(0.5)`, one
redraw per visited node. Deleting it and putting the same traversal on the recorder seam (D-32) turned
out to be four separate lessons, only one of which was about drawing.

**A snapshot's oracle can be deleted; its fingerprint cannot.** CONTRIBUTING's baseline rule has a
clause for exactly this case — "when deleting the code the snapshot was taken from, capture its
fingerprints first and test against those" — and the interesting part was that the old example had no
output to capture. It printed nothing; its result was pixels, at 0.6 s a frame. So the fingerprint had
to be taken by *replicating the builder* and running the same search directly: BFS from node 1 for node
7 over the fifteen-node graph is `[1, 2, 5, 3, 4, 6, 7]`, DFS is `[1, 2, 3, 4, 5, 6, 7]`. Those two
lists are now `tests/math/algorithms/test_graph_search_example_fingerprint.py`, and they are the reason
the rewrite is a refactor rather than a rewrite-and-hope. The test replicates the graph rather than
importing the new example, deliberately: a fingerprint that follows the code it fingerprints is not one.

**Two moments, not one — again.** Slice 1 learned that a search has an expansion moment *and* a goal
moment, because a walkthrough that ended on the last expansion stops one step before the answer. A
traversal has the same shape and it is not obvious until you try to caption the last frame: the run
that found node 7 on its seventh visit and the run that exhausted the graph on its fifteenth end on an
identical *visit*. Only a frame recorded after the last visit can say which happened, so
`AbstractTraversalRecorder` has `record_traversal_end(traversal_order, found)` beside
`record_visit(...)`, and the
end frame observes no node at all (`node_id` and `depth` are `None`). One frame type, two moments, as
`AStarFrame` already does with `goal`.

**The pending container is the whole difference, so record it raw.** BFS and DFS in this repo are the
same loop with `deque.popleft()` in one and `list.pop()` in the other. Everything else — the visited
guard, the depth limit, the neighbour sort — is shared. That made the recorder's one real design
decision the pending snapshot: pass it as a single `queue_or_stack` argument, in the container's own
order, and *do not normalise it*. A queue's first pair is the one that comes out next; a stack's last
pair is. Normalising to "next first" would have made the two recordings indistinguishable and thrown
away the only thing the pages are meant to teach. One recorder, one view, two documents that differ
exactly where the algorithms differ.

A smaller thing fell out of the same argument: the snapshot is taken *after* the node comes off the
container and *before* its neighbours go on, because that is the one call site the instruction budget
allows. So "pending" means "what was already waiting", not "what will be waiting". Naming that in the
frame's docstring was cheaper than moving the call.

**Deleting a module is mostly deleting its claims.** `visualizer.py` was 30 lines; removing it touched
its smoke test, the headless `matplotlib.use("Agg")` that test alone needed, the BL-14 "named kept set"
entry that promised to keep it, and the dependency comment in `pyproject.toml` that named it as
matplotlib's reason for being there. The comment turned out to be the interesting one: after the
deletion *and* the example's un-plotting, nothing in `src/` or `examples/` imports matplotlib at all —
only `notebooks/MLFall25HW1.ipynb` does. The honest comment says so, which is a better outcome than the
instructed text would have been, and it hands the "do we still need this dependency?" question to
whoever wants to ask it, with the evidence attached rather than hidden behind a stale name.

**`register` binds names, not signatures — so the traversal's end got its own name.** The traversal
recorder ends a run with `record_traversal_end(traversal_order, found)`, which reads clumsier than
`record_end` and is deliberate. Three facts collided when the slices were stacked. D-33 gives each
algorithm shape its own contract, so the traversal ABC and slice 3's `AbstractStepRecorder` are
separate classes — but `NullRecorder` implements *every* recording method in the module, because a
missing guard has to raise wherever it happens, and it is a virtual subclass of the traversal ABC
(declared above it, so it cannot inherit from it). One class, two contracts, and
`AbstractStepRecorder.record_end(run)` already owned the name. Two methods with one name and
different signatures on that class means one silently shadows the other, and `abc.register` checks
nothing at all — it records a claim, it does not verify a single method — so the collision would have
surfaced as a `TypeError` inside somebody's traversal, months later, and not at the definition. The
rename is what D-33 costs when contracts share a null implementation: the rule reaches method names,
not just classes. Worth remembering the general shape — a mechanism that makes a claim true to
`isinstance` without checking it is a mechanism that cannot catch a name clash for you.

**The composition root got smaller when the renderer arrived.** Before slice 2 landed, the example
built its recording as a literal dict, wrote it with `json.dumps(indent=2, sort_keys=True)` and kept
a `write_pages` that reported it could not render yet. Wiring it was mostly *deletion*:
`Recording.from_recorder(recorder, problem=…, configuration=…, result=order)` replaced the dict,
`recording.save(path)` replaced the dumps (and with it my guess at the file's formatting — the
committed fixture now matches the other two by construction rather than by coincidence), and
`write_walkthrough(recording, view_for(recording), path)` replaced the apology. The example's own
`schema_version` constant went too: two copies of a version number is one copy too many. The lesson
is about stacked slices rather than about drawing — the placeholder shaped like the eventual call
made the wiring a deletion instead of a rewrite, and the one thing worth keeping from the interim
was the problem *kind* string, which was the only real contract between the halves all along.

## Making a run explain itself: narration derived from frames · 2026-09-10

BL-43 phase 2 (plan § 6, D-34) put an explanation layer on the walkthrough pages: an opening panel,
a sentence per frame, key moments on the slider, a legend with hovers, a quantity strip, an ending
panel and glossary tooltips — every one of them computed in the view's Python from the recording and
embedded as data, so a re-recorded run re-narrates itself and a wrong sentence is a failing string
test. The technique is small; what it taught was about the runs, because a sentence per frame is a
reading of the recording nobody had done.

**Two optima tie on the RBF chain, and the tie-break never ran.** The fixture's goal frame now
reads "22 states remain on the frontier, none with a bound below 0.7231 … 1 of them ties with it at
0.7231 to four decimals": {1, 3, 6} and {1, 4, 6} are the chain's reflection of each other and have
the same residual trace. The first draft of the sentence said "so the tie-break (fifo) decided which
was expanded first", which the review caught as false: the two were pushed at 0.7231008921940996 and
0.7231008921941, the heap key is `(bound, insertion_index)` with `tie_tolerance = 0`, and fifo would
have picked the *other* one, inserted two expansions earlier. Which optimum the search "proves" on
this cell is decided by the fifteenth digit of a floating-point bound. The sentence now claims the
tie-break only when the pushed bounds are exactly equal or the tolerance is positive — the honest
version is derivable from the frames, the confident version was not, and the capped anytime cell
added to the example finds the other optimum first as its incumbent. The first complete selection
priced (frame 4) is neither. CI then proved the point the hard way: on the ubuntu runner OpenBLAS
puts the fifteenth digit the other way and the exact cell pops {1, 4, 6}, so a byte-for-byte fixture
guard was a guard on the Mac's LAPACK. The guard now reads the tie off the fixture's own goal frame
and accepts either reflection as the goal, everything else exactly; the cell itself keeps
`tie_tolerance = 0`, because any positive tolerance turns a 4e-16 difference into `optimal=False`
and the page into an exact run that certifies nothing.

**The two-hot optimizer lowers a loss that is not the objective.** On roach at λ = 10 the narration
says, frame after frame, "E\* rose by … since step N" while the training loss falls: the collision
term owns the descent. Ê settles at step 85 of 300 and holds for the remaining 214 steps; at the end
all 18 columns are 2-hot on 17 distinct pairs, of which 5 are edges of the graph — the rounded pairs
are chords, and no full frame ever reaches K = 2 components (the `K reached` marker is absent by
construction). The drift the seed asked to mark cannot be marked: only the run-wide maximum zero-sum
violation (6.7e-16 here) is in the result, so the ending panel names the absence instead.

**A frame is taken before the visit's own pushes.** The traversal engines record at the pop, before
the neighbour loop, so the ids that appear in frame i's container were pushed by visit i − 1. The
first narration said "1 neighbour joined the queue (7)" at the visit that did not push it; the
sentence now credits nothing to the current visit ("since the previous visit node 7 joined it").
The same seam property shows on the A\* side: the expansion that hits the anytime cap is never a
frame, so a goal priced on that expansion is visible only in the result, and the capped example cell
needed cap 7, not 5, to show an incumbent at all. The DFS frames add one more: node 5 is still on the
stack on the frame that visits node 5, because the engine pushes duplicates and `visited` there means
*seen*. None of this was changed — the layer's rule is that a value the frame lacks becomes a stated
absence in the plan's risks, never a new field.

**The glossary test did domain modelling.** The seed asked that tooltip terms be `CONTEXT.md` terms
and that a test check it. Neither glossary had the two-hot words (E\*, Ê, Σλ, rounded pair, rounded
cut, component count), and "incumbent" lived only on the BotMaker side. The test made the choice
explicit: the page's dictionary is keyed to BotMaker's `CONTEXT.md`, which gained a "Two-hot spanning
sets" area, and `used_terms` claims match spans longest-first so "relaxed objective" never carries
the search sense of "objective". A vocabulary rule with a test behind it is a vocabulary rule that
grows the glossary, which is the point.

## A learning-rate schedule cannot choose the pair: the adjacency probe · 2026-09-10

**What.** Two experimental training knobs on `fit_two_hot_span`, both off by default, and the probe
that ran them: a learning-rate schedule (`constant`, `linear`, `cosine`, `warmup_cosine`, driven by
one `LambdaLR` over one Adam so the moment estimates are never reset) and an optional per-column
graph term `- μ Σ_j term(v_j)/‖v_j‖₂²` in two forms — the Laplacian quadratic form `vᵀ L v` and the
edge product `|v|ᵀ A |v|`. The grid was λ ∈ {3, 10} × init ∈ {spectral, random} × schedule ∈
{constant, cosine, warmup 30} × adjacency ∈ {none, laplacian μ ∈ {0.3, 1, 3}, edge_product
μ ∈ {0.3, 1, 3}}, 84 runs at 300 steps on each of the roach G₅ and karate. The answer is negative
and worth having: on both graphs the best Ê is `μ = 0`, `constant` — roach 2.6500 (against an
optimum of 4/15 = 0.2667), karate 2.5793 (just under the datum's best rounding, 2.5972). No
schedule and no value of μ beat the unscheduled default on either graph.

**Where.** `src/mllib/math/algorithms/two_hot_span_optimizer.py` (`TwoHotSpanConfig`'s five new
fields and its `__post_init__`, `learning_rate_lambda`, `adjacency_term`, `graph_matrices`,
`TwoHotSpanRun.learning_rate_history`), `src/mllib/ml/projects/two_hot_span_harness.py` (the
pass-through flags), `examples/two_hot_span_adjacency_probe.py` (the probe),
`tests/math/algorithms/test_two_hot_span_schedule_adjacency.py`, and the bit-identity test in
`tests/math/algorithms/test_two_hot_span_optimizer.py`.

**Design.**

*Why both knobs are training knobs and never reported.* D-31 fixed that `epsilon` — the ridge in the
training projector — shapes the trajectory and never reaches a report, because every reported number
is recomputed in numpy through the exact `pinv` projector. The schedule and the graph term are the
same kind of object by the same argument: μ is not part of the objective the brief states in §20,
and a table that reported `E_ridge(V) - λ Σ R - μ Σ term` beside a brute-force RatioCut would be
comparing two different quantities. So the term is guarded off entirely at `μ = 0` (not one tensor
of it is built), the defaults are bit-for-bit the run slice 2.2 shipped, and `E*`, `Ê`, `R(v_j)` and
the clustering all still come back through `two_hot_span_problem`.

*The sign, which I got backwards first.* The instinct is that `vᵀ L v` is a cut and should be
*minimised*, so a pair that is an edge should score low. It is the other way round. For a unit
2-hot `v = (e_i - e_j)/√2`, `vᵀ L v = (L_ii + L_jj - 2 L_ij)/2 = (d_i + d_j + 2 w_ij)/2`, because
`L_ij = -w_ij` off the diagonal. The form is **larger** on an edge than off it, by exactly `w_ij`.
So if the goal is "prefer pairs that are edges", the Laplacian form has to be *rewarded* —
subtracted from the loss — not penalised. And it comes with a passenger: `(d_i + d_j)/2` is there
whether or not `(i, j)` is an edge, so the term also pays for putting the two hot entries on
high-degree vertices, which has nothing to do with the question. The edge-product form
`|v|ᵀ A |v| = w_ij` on an edge and exactly `0` off one is the same preference with the bias removed,
which is why both are implemented and both are tested against those hand values rather than against
each other.

*Why neither helped, which is the actual finding.* The probe's extra columns say it plainly. Turning
either form on does raise the fraction of rounded pairs that are real edges — on karate at λ = 3 the
`none` rows round 32 distinct pairs of which 12 are edges, and the `edge_product` rows at μ = 1
round 8 pairs of which 6 are edges. The trouble is the *8*. A per-column reward for landing on an
edge is, like `R`, invariant to which column does the landing, so the cheapest way to collect it is
for many columns to converge on the same few good edges. The pair graph then has almost no edges,
the components multiply (27 to 33 on karate, against K = 2), and Ê — which is the RatioCut of that
component partition — explodes from 2.58 to 230–414. The graph term buys edge-ness per column and
pays for it in coverage across columns, and Ê only reads the coverage. The schedule is a milder
version of the same lesson: annealing changes how far the columns travel, never which pair each one
picks, and the roach rows show it changing E\* substantially (1.06 → 0.07 at μ = 3) while leaving Ê
alone or worse. Nothing here couples the columns, and coupling the columns is what the rounding
needs — which is the same wall slice 2.3 hit from the other side.

**What was confusing.** Three things.

The first was the sign above, and it stayed confusing until the hand values were written as tests
rather than as a comment: `(d_i + d_j + 2 w_ij)/2` for an edge and `(d_i + d_j)/2` for a non-edge,
on a five-vertex weighted graph small enough to check with a pencil, is the thing that settles it.

The second was `LambdaLR`'s indexing. The multiplier is applied at construction (`_initial_step`),
so `lambda(0)` is in force for the *first* optimizer step and `lambda(step_count - 1)` for the last.
That makes "the fraction reached at the last step" an interpolation over `step_count - 1`, not
`step_count` — off by one in the direction that makes the final learning rate slightly wrong and no
test fail unless the test asserts the endpoint exactly. `learning_rate_history` exists for that: it
records what the param group actually held before each step, so the assertion is on the schedule the
run had, not on a formula re-derived beside it.

The third was that "off by default" and "bit for bit" are not the same claim, and only the second is
worth anything. Five new dataclass fields with neutral defaults *look* inert. Proving it took a test
that runs the same seed twice — once naming nothing, once naming all five at their defaults — and
compares `spanning_set` and `loss_history` with `np.array_equal`, not `approx`. It also cost the
committed walkthrough fixture a deliberate regeneration, because the recorder serialises the whole
config with `dataclasses.asdict` and five more keys is five more lines of JSON; the regenerated file
differs from the old one in exactly those ten lines (five in `configuration`, five in the result's
`config`) and in no number the run produced.

- **Reference.** Prof. Schweitzer's handoff brief §10 (the objective, `E(V) - λ Σ_j R(Cv_j)`, and
  the instruction not to carry the constant ½), §11 (the span term is rotation-invariant inside the
  span, the collision term deliberately breaks that invariance), §12–§13 (exact 2-hotness is not
  required, and Ê is the number that counts, not E\*), and §20 (the rcut objective this slice must
  leave standing as the default).

## Coupling the columns: the vertex-load diversity term · 2026-09-10

**What.** A third experimental training knob on `fit_two_hot_span`, off by default: the
column-diversity term `+ ν D(V)` with `D(V) = Σ_i ℓ_i²`. Each column carries the brief's §9
distribution `p_j = |v_j| / ‖v_j‖₁`; the **vertex load** `ℓ_i = Σ_j p_ij` is how much of the r units
of mass the whole spanning set puts on vertex i. `Σ_i ℓ_i = r` whatever V is, so by Cauchy-Schwarz
`D(V) ≥ r²/n`, with equality exactly when the load is flat. On exact 2-hot columns it is a statement
about the pair graph and nothing else: a column on (i, j) has `p = (½, ½)`, so `ℓ_i = deg_i/2` and
`D(V) = ¼ Σ_i deg_i²`.

This is the **first term in the module that couples the columns**. Everything before it scored a
column on its own — the span term is invariant to any rotation inside the span, `R(v_j)` asks only
whether a column is nearly 2-hot, and slice 2.4's graph term asks only whether its two hot entries
are joined by an edge — so nothing in the objective could tell two columns apart from one column
used twice. That is exactly the failure 2.4 diagnosed and could not fix.

The probe (λ ∈ {3, 10} × init ∈ {spectral, random} × adjacency ∈ {none, `edge_product` μ ∈ {0.3, 1}}
× ν ∈ {0, 0.3, 1, 3, 10}, 60 runs at 300 steps on each of the roach G₅ and karate) says the term
does what it claims and, on one graph, buys the answer. `ℓ_max` falls monotonically in ν almost
everywhere — on the roach from 1.93-3.46 at ν = 0 to 0.945-1.019 at ν = 10, against the flat load
r/n = 0.9 — and the collapse onto a handful of pairs that 2.4 measured is undone with it: karate's
`edge_product` rows round 7-9 distinct pairs at ν = 0 and 8-16 at ν = 10, and the roach's
`edge_product` μ = 1 rows rise from 7-9 to 13-18 out of r = 18.
  And on the roach the best cell is **Ê = 0.2667 at λ = 10, random init,
`edge_product` μ = 0.3, ν = 10 — exactly the 4/15 antenna optimum, on exactly K = 2 components**,
against 2.6500 for the best 2.4 cell and 1.0 for spectral bisection. It is converged: re-run at 1000
and 3000 steps Ê stays 0.2667 and `ℓ_max` creeps further toward 0.9 (1.0040 → 1.0018 → 1.0008).

On karate it does not win. The best Ê there is still 2.5793 at λ = 3, spectral, ν = 0, and that cell
too is converged (2.5793 at 300, 1000 and 3000 steps). So the term stays off by default and the
plan's slice 2.5 row owes no decision.

**Where.** `src/mllib/math/algorithms/two_hot_span_optimizer.py` (`TwoHotSpanConfig.diversity_weight`
and its `__post_init__` check, `diversity_term`, the guarded branch in `training_loss`, the call site
in `fit_two_hot_span`), `src/mllib/ml/projects/two_hot_span_harness.py` (the `run_graph` kwarg, the
`--diversity-weight` flag, the recorded `config` key), `examples/two_hot_span_diversity_probe.py`
(the probe, importing 2.4's pair-counting helpers rather than copying them),
`tests/math/algorithms/test_two_hot_span_diversity.py`, and one line each in the bit-identity test
and the harness's recorded-config test.

**Design.**

*The identity, which is the whole argument for this shape.* `‖p_j‖₂² = R(v_j)` by definition, so
expanding `D(V) = ‖Σ_j p_j‖₂²` gives

    Σ_{j<k} p_jᵀ p_k = ½ (D(V) - Σ_j R(v_j)).

D is therefore the **pairwise column overlap plus the collision sum**. The λ reward already pushes
`Σ_j R(v_j)` up; what ν adds on top of it is precisely a penalty on the overlap — on two columns
putting their mass in the same place. Writing it as `Σ_i ℓ_i²` rather than as the explicit double sum
is the same quantity in `O(nr)` instead of `O(nr²)`, and it is the form in which the ¼ Σ deg_i²
reading is obvious.

*Why the constant is dropped.* `D(V) - r²/n` is the nonnegative "load deficit" and it is tempting,
because it reads as zero-at-perfect. It is constant in V, so it changes no gradient and no comparison
between two runs on the same graph; carrying it would only invite reading the loss term as a deficit
score. This is the brief's §10 argument for not carrying the ½ in `½ - R(Cv)`, applied a second time,
and the tests pin the floor separately (`test_the_diversity_term_is_never_below_the_uniform_floor`)
rather than folding it into the term.

*Why it is added and not subtracted.* Both of the earlier terms are *rewards* and are subtracted; D
is a *penalty* and is added. Piling the columns onto the same vertices raises D, so minimising the
loss spreads them. Getting that sign wrong is silent — the run still finishes, holds zero-sum and
reports finite numbers — which is why `test_a_heavier_diversity_weight_spreads_the_load_further`
asserts the direction on a real run rather than trusting the reading.

*Still a training knob.* ν is not in the brief's §20 objective, so by the D-31 argument it never
reaches a report: it is guarded off entirely at ν = 0 (not one tensor of it is built), the defaults
are bit for bit the run slice 2.2 shipped, and E\*, Ê, R(v_j) and the clustering all still come back
through `two_hot_span_problem`'s exact `pinv` path.

**What was confusing.** Two things.

The first was that "the term spreads the load" and "the term improves Ê" are separate claims, and the
probe had to be built to separate them. `ℓ_max` falls in ν on nearly every row of both graphs — the
term is doing its job everywhere — while Ê improves dramatically on the roach and not at all on
karate. That is only legible because `ℓ_max` and `distinct pairs` are columns of the table beside Ê;
with Ê alone the roach row would look like luck. The mechanism the two columns together show is:
ν raises the number of distinct pairs, more distinct pairs make the pair graph connected, a connected
pair graph rounds to K components, and only then can Ê be the RatioCut of a partition anyone wanted.

The second was that the roach's winning cell needs **both** μ and ν. μ = 0.3 alone at λ = 10 gives
Ê = 4.6786 and ν = 10 alone gives 7.1250; together they give 0.2667. The graph term says *which*
pairs are worth taking and the diversity term says *don't all take the same one* — neither question
is the other, and 2.4 could only ask the first. That is the finding this entry extends: a
learning-rate schedule cannot choose the pair, a per-column graph term can choose a pair but not r
different ones, and it takes a term that reads the whole matrix to ask for r different ones.

- **Reference.** Prof. Schweitzer's handoff brief §9 (the distribution `p_i = |u_i|/‖u‖₁` and the
  collision probability `R(u) = Σ p_i²` built on it — D is the same construction read across the
  columns instead of down one), §10 (do not carry the constant), §11 (the span term is rotationally
  invariant within the span and the collision term deliberately breaks that invariance — D breaks the
  remaining permutation-of-columns symmetry that neither of them touches), §13 (Ê is the criterion,
  not R), and §20 (the objective this slice leaves standing as the default).

## The eighty-node cockroach: three parts, not two · 2026-09-10

**What.** A second cockroach instance for the two-hot span prototype, taken from He, Gu & Zhang,
*Nodal domain partition and the number of communities in networks* (arXiv:1201.5767, 2012), Fig. 1-2
— `docs/REFERENCES.md` **L5**. It is the same Guattery-Miller family the prototype already carried,
at k = 20 instead of k = 5: two paths of 40, rungs along 20 columns, antennae of 20 on the far side,
so n = 4k = 80 and m = 5k - 2 = 98. What is new is not the graph, it is **K**.

The paper's point is why K = 2 is the wrong question on this graph. The Fiedler vector cuts
horizontally through all twenty rungs — the classic Guattery-Miller failure — and against it the
paper sets what its Fig. 2a calls the ideal cut, a single dashed line just past the last rung. That
one line makes **three** blocks, not two, because once the ladder is taken out the two antennae are
not joined to each other by anything: `roach_graph`'s only antenna-to-ladder edges are 19-20 and
59-60, and deleting exactly those two leaves the top antenna, the bottom antenna, and a ladder that
stays in one piece through its twenty rungs. So the instance is planted at K = 3 (label 0 the
ladder, 1 the top antenna, 2 the bottom one), and the paper's own explanation of why the Fiedler
vector cannot express it is the nodal-domain bound: a Fiedler vector's weak nodal domains number at
most two, so a three-part answer is not in it at any threshold, while the third eigenvector (their
Fig. 2b) is the one that carries it. Their λ₂ = 0.0057, λ₃ = 0.0062, λ₄ = 0.0246 for the
unnormalized Laplacian come back from numpy to the digit — 0.005664, 0.006165, 0.024623 — which is
the check that the repository's graph really is their graph.

**Where.** `src/mllib/math/graph/two_hot_span_problem.py` (`roach_g20_instance`, and the fifth entry
of `default_test_graphs`), `tests/math/graph/test_two_hot_span_problem.py` (six new tests — the shape,
the planted labels, the 0.15 cut, the 0.10 cut, the components of the severed graph, and the
eigenvalues — plus one modified: `test_default_test_graphs_are_the_four_named_connected_instances`
became `..._five_...`),
`examples/two_hot_span_walkthrough.py` (`roach_g20` at λ = 10 spectral, and a `layout_positions`
that reads k off n = 4k), `examples/two_hot_span_adjacency_probe.py` (`CLUSTER_COUNTS`, `is_roach`,
`references_for`, and `cross_path_pair_count` taking the boundary as an argument),
`examples/two_hot_span_diversity_probe.py` (the same cluster count threaded through `probe_row`),
`src/mllib/ml/projects/two_hot_span_harness.py` (docstring only — the suite picked the graph up on
its own), `tests/math/algorithms/test_two_hot_span_probe_graphs.py`,
`tests/ml/test_two_hot_span_harness.py` and `tests/visualization/test_two_hot_span_example.py`.

**Design.**

*Reuse of `roach_graph`, not a second generator.* The graph the paper draws is `roach_graph(20)` up
to a mirror image; writing a `paper_cockroach()` beside it would have duplicated the one function
whose n = 4k, m = 5k - 2 arithmetic is already pinned by a parametrized test. The instance is
therefore a *labelling*: a `GraphInstance` with `cluster_count = 3` and planted labels, and nothing
in the graph module changed except the addition.

*The planted labels are the reference, and there is no optimum here.* n = 80 is far past
`BRUTE_FORCE_NODE_LIMIT = 10`, so the harness reports `brute force = n/a` and the only datum with a
name is the planted RatioCut, 2/40 + 1/20 + 1/20 = 0.15. Beside it the K = 2 antennae-vs-ladder cut
is 0.10 — *lower*. That is not a contradiction and it is worth stating in the docstring rather than
being discovered later: RatioCut pays a cut term per block, so merging the two antennae into one
block halves the number of blocks paying for the same two edges. The paper's ideal cut is the better
*community structure*; it is not the smaller RatioCut. The instance is reported with both numbers
side by side and neither is called the optimum.

*Appended last.* `default_test_graphs()` is indexed positionally by existing tests, so the fifth
instance goes after `two_moons_knn` even though the two roaches would read better adjacent.

*Constants became properties of the instance.* Two literals in the examples were the k = 5 roach in
disguise: the ladder layout's `2 * ROACH_RUNG_COUNT` and the probes' `ROACH_PATH_SPLIT = 10`. Both
now come from the graph (`n // 2`), and the probes' hard-coded `cluster_count = 2` became
`CLUSTER_COUNTS[name]`, since `roach_g20` is the first non-bisection they see. The k = 5 values are
unchanged by construction, which the byte-identical visualization fixture confirms.

**What was confusing.** Two things, both about reading the figure.

The paper's cockroach is drawn **mirrored** relative to this repository's convention: their rungs are
on the left and their antennae on the right, `roach_graph`'s rungs are on the *last* k columns. It
took a node-by-node reading of Fig. 2a to be sure that the horizontal solid line (Fiedler) and the
near-vertical dashed line (ideal) were the same two cuts the repository already had names for, and
that the mirror changes no RatioCut, no eigenvalue and no component count.

The second was the caption's arithmetic: "has 80 nodes ... with each suspension points representing
a line of 16 nodes". The figure draws 2 + 16 + 2 nodes along each of its four straight runs, so the
"16" is the elided middle of a run of 20, not a run of 16, and reading it as 16 gives n = 64 and a
graph that reproduces none of the paper's eigenvalues. The eigenvalue check is what settled it: at
k = 20 numpy returns 0.0057 / 0.0062 / 0.0246 exactly as printed, and at k = 16 it does not.

**Reference.** He, Gu, Zhang, *Nodal domain partition and the number of communities in networks*,
arXiv:1201.5767 (2012), §"Partition by weak nodal domain" and Fig. 1-2 (**L5**); Guattery, Miller,
*On the quality of spectral separators*, SIAM J. Matrix Anal. Appl. 1998, for the cockroach itself
and for why spectral bisection misses its antenna cut.

## λ is not scale-free: the collision reward against the spectral floor · 2026-09-10

**What.** A six-node instance — two weight-1 triangles, 0-1-2 and 3-4-5, joined by the weight-0.1
bridge (2, 3) — entered the prototype as the sixth `default_test_graphs()` entry because Xavier's own
minimal implementation of the same objective ships it as its `main()` example, and the two engines
were run side by side on it today. It is the smallest instance on which *nothing but the optimizer*
can be blamed: n = 6 is inside `BRUTE_FORCE_NODE_LIMIT`, so the RatioCut optimum is enumerated rather
than referenced and comes back as 1/15 = 0.066667 at exactly the planted labels [0,0,0,1,1,1], and
all three datum roundings — `kmeans`, `discretize`, `cluster_qr`, seed 0 — land the same partition.
Every other instance in the suite has either no optimum (n past the guard) or a datum that itself
misses it; here the objective, the rounding rule and the datum are all exactly right, so a run that
does not reach 1/15 has missed it for reasons of its own.

The first thing it settled was **λ**. Σλ = 0.063771 on this graph. The collision reward is bounded by
λ·r/2 — R(v) ≤ ½ per column, r = n − K = 4 — so at the prototype's λ = 10 its ceiling is **20**,
three hundred times the floor it is competing with. And the run does exactly what that arithmetic
says it should: the span term stops mattering, columns go 2-hot on whatever pairs are nearest and
ignore the graph entirely, E\* rises to 1.5–2.5 — *above* the optimum, not near the floor — and Ê
lands at 1.5–5.5 from every start tried: three random seeds at 300 and at 3000 steps, the spectral
initialisation, and all ten restarts of the joint-factorization engine run beside it. **Not one hit.**
At λ = 0.1 the ceiling is 0.2, the same order as Σλ, and this engine reaches Ê = 1/15 from the
spectral initialisation and from
two of those ten seeds (3 and 7).

The reading: **λ is not a number, it is a ratio**, and the band the earlier slices quote — λ ∈ {0.1,
0.3, 1, 3, 10} across every graph — is quoting the numerator alone. The quantity that transfers is
**λ·r/Σλ**: the reward measured in units of the floor it has to beat, its ceiling being half of that.
On the roach G₅ at K = 2, Σλ = 0.0713 and r = 18, so λ = 10 is λ·r/Σλ = 2525; on the triangles
λ = 10 is 627 and λ = 0.1 is 6.3. Two graphs at "the same λ" are nowhere near the same place in the
objective — the roach's λ = 10 sits four times deeper into the reward than the triangles' same λ —
and a λ quoted without Σλ and r beside it says nothing about either.

**Where.** `src/mllib/math/graph/two_hot_span_problem.py` (`two_triangles_graph`,
`two_triangles_instance` and its docstring, which carries the λ·r/2 arithmetic, and the sixth entry
of `default_test_graphs`), `tests/math/graph/test_two_hot_span_problem.py` (three new tests and
`..._five_...` → `..._six_...`), `tests/ml/test_two_hot_span_harness.py` (the suite test, which on
this instance finally has a `brute_force_optimum` to assert Ê against instead of `None`),
`tests/ml/test_two_hot_span_datum.py` (the three roundings), `examples/two_hot_span_walkthrough.py`
(the graph at λ = 0.1 with a schedule of its own, the explicit layout, `--learning-rate`),
`tests/visualization/test_two_hot_span_example.py`, and three pages under
`src/mllib/visualization/walkthroughs/`.

**Design.**

*The instance is a labelling, and its numbers are facts.* Like `roach_g20` it adds no mathematics —
seven `add_edge` calls — and like it, it is appended **last** so every index-based reference to
`default_test_graphs()` still names what it named. What is new is the epistemic status of its
reference numbers. `roach_g20`'s planted RatioCut of 0.15 is a *reference*: it is not the optimum,
and the instance is reported with two cut values side by side precisely because neither can be
called one. The triangles' 1/15 **is** the optimum, enumerated, and it is also what the datum
reaches. That is the difference that makes the instance useful: it converts "the run scored worse
than the reference" into "the run is wrong", which no other instance in the suite can do.

*The graph carries its own schedule, and only this one does.* λ = 0.1 needs a longer, finer run than
λ = 10 does — 5000 steps at lr 0.01, against the shared 300 at 0.05 — because Σλ here is three orders
below the roach's and the shared step size walks straight past the optimum. So a `GRAPHS` entry may
now carry `learning_rate` and `step_count`, but they apply *only* when the CLI left its own flags at
their defaults, and the three older entries carry neither. The roach G₅ walkthrough fixture is the
guard: it is byte-identical, which is the statement that nothing about the older runs moved.

**The side-by-side, which is the observation this entry exists for.** Xavier's engine reached 1/15 on
**5 of 10 restarts** at λ = 0.1, against this repository's 2 of 10 seeds and its spectral hit. Two
differences, and they are not separated yet. His formulation is a joint (W, H) factorization rather
than a single moved V; and he takes ten restarts and selects the best by **training loss**. On this
instance that selection rule coincides with selecting by Ê, which is a coincidence worth naming
rather than trusting: the losing restarts are legible failures, one column stuck at R ≈ 0.43–0.47 —
visibly short of ½, so visibly not 2-hot — and the training loss sees that through its collision term.
Where a failure is *not* that legible the two selection rules must come apart, because the training
loss is the ridge loss and carries the reward (D-31), and Ê is the criterion (§13). So this is an
observation with a follow-up and not a result: **BL-46** pairs the two formulations on roach G₅ over
ten seeds at λ scaled by Σλ/r, reported by Ê, and asks whether the restarts alone explain the gap.

**What was confusing.** That E\* *rises* under a large λ was, at first reading, a bug. It is not: E\*
is reported at the unrounded V through the exact `pinv` projector, and nothing constrains it to fall —
the thing being minimised is the training loss, of which the span term is one part and the collision
reward another, and at λ·r/2 = 20 against Σλ = 0.064 the optimizer is trading nearly two full units of
span residual for a fraction of a unit of reward and coming out ahead on the loss. The objective is
doing exactly what it was asked. The number that says so is not E\* on its own but E\* read against
Σλ, which is why the floor is on every page and in every report row.

- **Reference.** Prof. Schweitzer's handoff brief §13 (Ê is the criterion, R is the diagnostic — the
  λ = 10 run maximises R on every column and is still wrong, which is that sentence with numbers
  attached) and §20 (the objective, and the λ it leaves unspecified).

## A penalty as an injected object · 2026-09-11

**What.** The three terms the two-hot training loss adds to its ridge cost — the collision reward
`-λ Σ_j R(v_j)`, the graph reward `-μ Σ_j term(v_j)/‖v_j‖₂²` in its two forms, and the diversity
penalty `+ν D(V)` — as objects the loss is handed rather than branches the loss owns. The loss is
now one line: the ridge cost, then `loss + penalty.compute_penalty(V)` for each injected penalty in
turn. Each object carries its own weight as a knob and its own sign, so a reward comes back negative
and a penalty positive, and the loss never asks which is which. `term(V)` beside it is the
unweighted number a hand check reads off a five-vertex graph; `compute_penalty` is the signed
contribution the loss sums. The two adjacency forms became two classes, `LaplacianAdjacencyPenalty`
and `EdgeProductAdjacencyPenalty`, rather than one class with a form knob: they are two
implementations of one concept, which is what earns a class (D-35 (1)), and a string that selects a
branch inside a method is the shape the decision retired.

**Where.** `src/mllib/math/regularization_function.py` (`AbstractRegularizationFunction`, an
`abc.ABC` per D-35 (2), array-agnostic per D-35 (6)) ·
`src/mllib/math/algorithms/two_hot_span/penalties.py` (the four penalties; `graph_matrices` now
sits beside `laplacian_matrix` in `math/graph/two_hot_span_problem.py`, slice 2's dedupe target) ·
`training_loss` and the transitional `_penalties_from_config` in
`src/mllib/math/algorithms/two_hot_span_optimizer.py` · tests
`tests/math/algorithms/test_two_hot_span_penalties.py` (old arithmetic as the oracle, exact
equality; `gradcheck` per penalty; signs; the adapter) and
`tests/math/algorithms/test_two_hot_span_refactor_baseline.py` (three seeded runs pinned to the last
bit across slices 1-4).

**Design.** Three choices worth carrying.
- *Zero weight means absent, not zero.* The old loss guarded each term behind `if weight != 0.0` so
  that a default run built not one tensor of any term. The objects keep that property by not
  existing: the adapter builds a penalty only for a non-zero weight, and a composition root that
  does not want a term leaves it out. A `weight=0.0` penalty is legal but is not a switch — it still
  builds its tensors, and on a column of zeros the diversity term divides by zero and `0.0 * NaN`
  poisons the loss. So the default run is bit for bit the run slice 2.2 shipped because the
  arithmetic is literally the same expression tree, not because a zero happened to cancel.
- *Adding `-(w·t)` is the subtraction it replaced, to the last bit.* IEEE 754 defines `a - b` as `a
  + (-b)`, and negation is exact, so `loss + (-(w * t))` and `loss - w * t` round identically.
  Autograd agrees: `sub` gives the second operand `-grad`, and `add` of `neg` gives `neg(grad)`, the
  same `-w` on `t` either way. The refactor snapshot is the proof: three 30-step runs with every
  term on, byte-identical before and after.
- *Two methods, not one.* `term` exists because the hand values in the adjacency tests — `(d_i + d_j
  + 2 w_ij)/2` on an edge, `w_ij` for the edge-product form, `0` off an edge — are statements about
  the unweighted term, and a test that had to divide a signed weighted number back out would be
  checking the sign convention and the hand value at once.

**What the wider CI run found.** Running every torch test on the ubuntu runner for the first time
failed the five spectral-start fixtures and nothing else. The spectral start is the optimum of the
span term, so the first gradient there is rounding noise on 20 of 360 entries, and Adam's first
step, `lr · g / (|g| + 1e-8)`, is a sign function of it: a 1e-14 perturbation of the start moves V by
0.094 after one step, where a random start moves by 8e-15. That is a property of the start, not of
this slice — the refactor snapshot passed on the runner — and it is recorded as BL-50 with the
five tests skipped off the platform that wrote their fixtures.

**What was confusing.** Whether `compute_penalty` should return the unweighted term and let the loss
apply the weight and the sign. That would put the sign — the one fact that distinguishes a reward
from a penalty — back in the loss as a per-class branch, which is the branch the slice removes. The
training loss is a plain sum precisely because each term knows its own sign; the training knobs stay
training knobs and never reach a reported number (D-31).

- **Reference.** D-35 (1)-(3), (6); D-31 for the reporting boundary; the brief's §9 (the
  distribution `p_j`), §10 (why the ½ is dropped from the collision term) and §20 (the objective).

## The ridge projector as a filter-factor smoother · 2026-09-11

**What.** The projector onto the span of a spanning set V, as one concept with two arithmetics.
The exact projector P_V = V V⁺ through the pseudo-inverse is the definition of "the span": it is
the same map whether V's columns are independent or three copies of one vector, because the
pseudo-inverse projects onto the column space and nothing else. The ridge projector
P_ε = V (VᵀV + εI)⁻¹ Vᵀ is not that map; in the singular basis of V, with singular values s_i,
it keeps direction i with the filter factor s_i²/(s_i²+ε), which is one where s_i² ≫ ε, zero
where a column has collapsed, and smooth in between. So ‖X - P_ε X‖² = ‖X‖² - Σ_i (2f_i - f_i²)
‖u_iᵀ X‖²: each direction gives X back what it keeps, twice, minus what keeping it twice
double-counts. The ridge residual is never below the exact one and falls to it as ε falls. That
smoothness is the whole point: `torch.linalg.solve` and the gradient stay defined as columns go
dependent, which the pseudo-inverse does not promise, and VᵀV = I is never imposed (D-31).

**Where.** `src/mllib/math/projector.py` (`AbstractProjector`, `ExactProjector`) ·
`src/mllib/math/algorithms/two_hot_span/projectors.py` (`RidgeProjector`) ·
`src/mllib/math/cost_function.py` (`AbstractCostFunction`) · `SpanCost` in
`src/mllib/math/graph/two_hot_span_problem.py` · tests `tests/math/test_projector.py` and
`tests/math/algorithms/test_two_hot_span_projectors.py` (the old bodies pasted as oracles, exact
equality; the closed form; ε → 0; a dependent column; gradcheck) and the refactor snapshot
`tests/math/algorithms/test_two_hot_span_refactor_baseline.py`, byte-identical.

**Design.** Four choices worth carrying.
- *One concept, two arithmetics, one name.* The exact and the ridge projector are two members of
  `AbstractProjector`, not a function and an unrelated inline solve, which is D-35 (6) applied:
  the array library is the implementation's business and the concept has one home.
- *One cost class, the projector injected.* The plan named a `RidgeSpanCost`. `SpanCost(projector,
  X)` serves both the report and the training loss by taking its projector, so there is no
  `ExactSpanCost` beside it and the cost is written once (D-35 (1): a class is earned by injection,
  and the projector is the thing injected).
- *The collision measure is not a duplicate.* The plan's evidence called the numpy
  `collision_measure` and the torch collision term "the same arithmetic twice". It is not: the
  numpy one returns 0 for the zero vector and divides by the largest entry before squaring, so a
  vector of subnormals does not square to 0/0, and neither guard belongs on the gradient path. Two
  arithmetics of one diagnostic, kept apart on purpose, and the plan's §2 says so now.
- *The Laplacian dedupe waits for slice 4.* D − A from the graph and X Xᵀ from the incidence
  matrix are the same matrix up to ulps, and ulps are exactly what the spectral start amplifies
  (BL-50). Collapsing them moves the spectral-start fixtures, and slice 4 is the slice that
  regenerates fixtures, so the collapse rides with it rather than forcing a regeneration here.

**What was confusing.** Whether `ExactProjector` needed to be a class at all, with no knob and one
implementation. It does, because the concept has two implementations and a caller injects one of
them into `SpanCost`; the empty frozen dataclass is the price of `describe()` reading an empty
parameter list rather than `(*args, **kwargs)` off a plain class. And whether the D-31 guard could
survive the removal of `projector_residual`: it does, moved onto `ExactProjector.residual`'s
signature and `describe(ExactProjector)["params"]`, and it still says the same thing.

- **Reference.** D-31 (the reporting boundary; ε is a training knob); D-35 (1) (a class is earned)
  and (6) (one class per concept per arithmetic); Hansen, *Rank-Deficient and Discrete Ill-Posed
  Problems* (SIAM 1998) for filter factors, from memory.
