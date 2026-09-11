# The optimizer object model: the two-hot span optimizer onto injected math objects

Status: **accepted** (2026-09-11, grilled with Xavier) · Written 2026-09-11 · Owner: Xavier
Target: six stacked slices into `main` (`main` @ `4c76986`): slice 0 from
`docs/object-model-philosophy`, slices 1–5 from `refactor/optimizer-object-model-<n>` branches,
each off the previous.
Backlog entry: **BL-48** (BL-49 opened beside it). Decision: **D-35**.
Philosophy: `docs/philosophy/mllib-object-model.md`.

---

## 0. Why

The next two things the two-hot span optimizer has to do are already in the backlog. BL-46 wants
restarts and a joint (W, H) factorization as an optimizer variant; BL-41 wants the ncut objective in
place of rcut. Neither slots in: the training loss is a free function, the penalties are branches
inside it, Adam and its schedule are built inside the function, and a variant is an edit to
`fit_two_hot_span`. Next to it the search stack takes a problem, a cost function, an engine and a
recorder as injected objects, and every variant so far — pruned, anytime, conditional solves, tie
tolerance — arrived without editing the base loop (D-23, D-24, D-28, D-30). The library's two
families of run are meant to have one shape; today they have two.

The refactor gives the optimizer the search stack's shape and, in doing so, gives the library a rule
for when a concept is a class at all (D-35). The acceptance test is concrete: after slice 4, BL-46
and BL-41 each open as an addition and touch no existing class.

## 1. Decisions and assumptions

The rules are D-35's ten; this table names which each slice leans on.

| # | Rule, one clause | Source |
|---|---|---|
| P-1 | A class is earned by injection, plurality or state; otherwise a function. | D-35 (1) |
| P-2 | Interfaces are `abc.ABC` named `Abstract*`; Protocols only at BL-19. | D-35 (2) |
| P-3 | Behaviour is injected; a knob lives on the object it parameterises, default on the class. | D-35 (3) |
| P-4 | `configuration` is the algorithm's dataclass plus each injected object's `describe()` params. | D-35 (4) |
| P-5 | A grid is over factories; fresh objects per cell, no mutation between cells. | D-35 (5) |
| P-6 | One class per concept per arithmetic; torch only in concrete implementations, together. | D-35 (6) |
| P-7 | The comment ladder: one home per kind of sentence, a decision cited by id plus one clause. | D-35 (7) |
| P-8 | Harnesses are composition roots that consume the run contract and record `configuration`. | D-35 (8) |
| P-9 | A result is paid, set-up, delivered; histories on the recorder; a stop is a reason, not a raise. | D-35 (9) |
| P-10 | Intent first: the cited frameworks yield when they would hide what a thing is for. | D-35 (10) |

| # | Assumption | Why |
|---|---|---|
| A-1 | `tests/ml/test_training_baseline.py`, `tests/ml/test_nystrom_search_baseline.py` and, from slice 1, the optimizer's own refactor snapshot `tests/math/algorithms/two_hot_span_refactor_snapshot.json` (three seeded 30-step runs, exact on the platform that wrote it) stay byte-identical in every slice, the docs slice included as the check that they pass on the branch. | They are the only guarantee that a restructuring changed no number (CONTRIBUTING, refactor rule). |
| A-2 | The two-hot walkthrough fixture (`tests/visualization/fixtures/two_hot_span_roach_g5_30steps.json`, compared up to float noise) and the stress fixtures regenerate only in slice 4, and that PR says so. | A fixture regenerated in a slice that also changes arithmetic hides the change; slice 4 proves the loop first (risk 1). |
| A-3 | torch stays a PEP 735 group (D-31): every torch test uses `pytest.importorskip`, the default suite is green without it, and CI's torch job is the one that runs them. | A contributor without the group still gets a green suite; the abstract bases in `math` never import torch (P-6). |

## 2. Current state (evidence)

Captured 2026-09-11 at `4c76986`. Paths under `src/mllib/` unless stated.

- **One module, no hierarchy.** `math/algorithms/two_hot_span_optimizer.py` is 422 lines:
  `TwoHotSpanConfig` (`:57`, a frozen dataclass of 13 knobs), `ZeroSumViolation` (`:117`),
  `NonFiniteLoss` (`:121`), `TwoHotSpanRun` (`:125`, carrying `loss_history` and
  `learning_rate_history`), `learning_rate_lambda` (`:138`), `adjacency_term` (`:173`),
  `diversity_term` (`:200`), `graph_matrices` (`:230`), `training_loss` (`:247`),
  `_project_zero_sum` (`:289`) and `fit_two_hot_span` (`:299`). Adam and `LambdaLR` are built inside
  the function, before the step loop (`:343-344`); the recorder is the one injected object
  (`:399-402`, `:420-421`).
- **The projector exists twice by design.** Exact `pinv` in `projector_residual`
  (`math/graph/two_hot_span_problem.py:113`) for every reported number; the ridge
  `torch.linalg.solve` inside `training_loss` (`:267-273`) for the gradient (D-31).
- **Two measures exist twice by accident.** The collision measure is `collision_measure`
  (`two_hot_span_problem.py:121`) and again inline in `training_loss` (`:275-277`); the Laplacian is
  `laplacian_matrix` (`two_hot_span_problem.py:107`) and again as `X @ X.T` in `graph_matrices`
  (`:237-239`). Same arithmetic, two array libraries.
- **Two concepts have no implementation.** `CostFunction` (`math/cost_function.py`) and
  `RegularizationFunction` (`math/regularization_function.py`) have no subclass in the tree; their
  only callers are `tests/math/test_task_kind.py` and `tests/math/test_smoke_modules.py`.
- **Histories on the result have four readers.** `TwoHotSpanRun.loss_history` is read by
  `visualization/recorders/two_hot_span.py` (`:209-212`, `:311-317`),
  `visualization/views/two_hot_span.py` (`:341`, `:561`), `ml/projects/two_hot_span_stress.py`
  (`:768`) and `ml/projects/two_hot_span_harness.py` (`:340-341`).
- **The harness bypasses the contract.** `ml/projects/two_hot_span_harness.py:290-342` imports
  `fit_two_hot_span` lazily, builds one config and calls it positionally per cell with no recorder,
  reading fields off `TwoHotSpanRun` directly. `fit_two_hot_span` has ten callers: the harness, the
  stress harness, three examples and five test modules.
- **The search stack, for contrast.** `AStarSearch.__init__`
  (`math/algorithms/a_star_search.py:183-194`) takes `problem`, `cost_function`, `evaluator` and
  `recorder` as objects and four knobs by keyword; `AbstractGraphProblem`
  (`math/graph/abstract_graph_problem.py:20-33`) is an `ABC` with three abstract methods. Nothing in
  the optimizer module answers to either.

## 3. Phases and slices

Each slice: branch off the previous slice's branch;
`uv run ruff check . && uv run ruff format --check .`; `uv run pytest` (with and without the torch
group); reviewer-agent; Xavier commits and opens the PR. Tests and records ship in the same PR as
the code they cover. Both baselines (A-1) re-run in every slice. "Done when" is in addition to that.

**Acceptance, verbatim, on slices 1–4:**

- "a class is earned by injection, plurality or state" (D-35 (1))
- "abstract bases in `math` never import torch" (D-35 (6))
- "a run with the null recorder is byte-identical to today" (D-32) — for slice 4, up to the
  result's shape, with the loss history proved element for element before it moves

| Slice | Branch | Change | Tests | Records | Done when |
|---|---|---|---|---|---|
| 0 | `docs/object-model-philosophy` | Docs only. `CONTEXT.md` gains math object, knob, configuration, composition root, optimizer, step rule, training loss; `docs/philosophy/mllib-object-model.md`; D-35; `docs/ARCHITECTURE.md` §1, §2, §4, §5, §6; one `CLAUDE.md` read-order line; BL-48 and BL-49; this plan. | The glossary test (`tests/visualization/test_glossary.py`) still parses `CONTEXT.md`; full suite green as a check. | All of the above. | Every new term is a `**Term**:` heading; no file names in the philosophy doc; ARCHITECTURE §1 no longer says "deliberately literal". |
| 1 | `refactor/optimizer-object-model-1` | `RegularizationFunction` becomes `AbstractRegularizationFunction`, an `ABC` with `compute_penalty(parameters)` (P-2); `CollisionPenalty`, the adjacency term (two classes, `LaplacianAdjacencyPenalty` and `EdgeProductAdjacencyPenalty`: two implementations of one concept, P-1) and `DiversityPenalty` implement it in torch under `math/algorithms/two_hot_span/`, each weight a knob on its class with today's default; `training_loss` sums injected penalties instead of branching. | The refactor snapshot (`test_two_hot_span_refactor_baseline.py`, generated from `main` before the change) byte-identical; each penalty equals today's inline term on a fixed V to the last bit, and `training_loss` equals `main`'s old `training_loss` pasted as the oracle; a numeric gradient check against autograd per penalty; `test_task_kind.py` and `test_smoke_modules.py` updated for the abstract base; torch tests `importorskip`. | BL-48 progress line; LEARNING_LOG entry "a penalty as an injected object". | Both baselines byte-identical; every existing two-hot test passes unchanged in what it asserts. |
| 2 | `refactor/optimizer-object-model-2` | `CostFunction` redefined as `compute_cost(parameters)` with an optional `task_kind`; the projector as one concept with two arithmetics — an abstract projector, `ExactProjector` (numpy, `pinv`, reporting) and `RidgeProjector` (torch, `epsilon` a knob, training); `RidgeSpanCost` as the first `CostFunction`; `collision_measure` and the Laplacian each reduced to one implementation, the torch side calling the numpy one outside the gradient path or keeping a torch member under P-6 if it is inside it. | `ExactProjector` reproduces `projector_residual` bit for bit; `RidgeSpanCost` reproduces today's `training_loss` with penalties off; the D-31 test that the reporting path never sees `epsilon` stays and passes; `test_task_kind.py` rewritten for the new contract. | LEARNING_LOG entry "the ridge projector as a filter-factor smoother" (σ²/(σ²+ε)); D-31 unchanged. | Both baselines byte-identical; one definition of each measure remains under `src/`. |
| 3 | `refactor/optimizer-object-model-3` | An `AbstractStepRule` (`step(parameters, gradient)` plus a per-step multiplier) with `AdamStepRule` holding the learning rate as its knob and one schedule object per form — `constant`, `cosine`, `warmup_cosine`, `linear`, the four `LEARNING_RATE_SCHEDULES` names — as injected schedules; `describe()` reads the knobs off each. | Each schedule's multiplier sequence equals today's `learning_rate_lambda` exactly over 1 000 steps for the default and one non-default knob setting; an Adam step equals `torch.optim.Adam` plus `LambdaLR` on one seed, bit for bit; `describe(AdamStepRule(...))["params"]` names the learning rate. | LEARNING_LOG entry "step rules with schedules"; BL-48 progress line. | Both baselines byte-identical; `learning_rate_lambda` deleted or a one-line adapter, stated. |
| 4 | `refactor/optimizer-object-model-4` | `AbstractOptimizer` in `math/algorithms` owning `run()` (recorder guard, stop check, result assembly) with a `_step` seam, mirroring `AbstractGraphAlgorithm`; `TwoHotSpanOptimizer` on it in `math/algorithms/two_hot_span/`, taking a `TwoHotSpanProblem` (X, cluster count, the exact reporting arithmetic), a cost, penalties, a step rule and a recorder, with a frozen dataclass of the loop's own knobs (step count, seed, zero-sum tolerance, normalisation); `configuration` assembled per P-4; the result per P-9 with a `stop_reason` in place of the two exceptions, which are deleted; histories move to the recorder; the harness and the stress harness become composition roots with grids over factories (P-5, P-8) and put `configuration` on every row; the recorder and view read histories from the recorder. `fit_two_hot_span` is **removed**: after this slice every caller outside `examples/` is a composition root by P-8, so a bare function would be a second entry point with a second configuration story; the three examples construct the optimizer themselves. | Before the result moves: the new loop's per-step losses equal the old loop's `loss_history` element for element on the roach G₅ fixture seed, asserted in a test that is then kept against the recorder's frames. After: a stopped run returns a result with its reason and the frames up to the stop; `configuration` names every knob including `epsilon` and the step rule's learning rate; the harness row carries it; the stress harness's frozen rows compare on the same fields as today plus `configuration`; the walkthrough fixture and the stress fixtures regenerated deliberately and named in the PR; the five test modules that call `fit_two_hot_span` are rewritten to construct the optimizer, asserting what they assert today. | BL-48 progress line; ARCHITECTURE §2 contract for `AbstractOptimizer` confirmed against the code; `docs/reports` untouched. | Both baselines byte-identical; the acceptance sentences for BL-46 and BL-41 written in §5 and checked by the reviewer-agent; `two_hot_span_optimizer.py` gone, its module docstring's D-31 and BL-41 citations carried to the package. |
| 5 | `refactor/optimizer-object-model-5` | Fix slice. `math/hypothesis.py:51` calls `hypothesis_expander.expand(self.hypothesis, self.degree)`, a signature no expander defines; fix to the expander contract the descent base already uses. | A test that calls `HypothesisFunction.expand_hypothesis` through each expander and asserts the shape; the training baseline byte-identical. | BL-49 amended to say the bug is closed here. | Both baselines byte-identical; the test fails on `main` @ `4c76986` and passes on the branch. |

The prose calls this six slices: slice 0 is the docs slice, slices 1–4 the refactor, slice 5 the
fix.

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Running the whole suite in `ci-torch` (slice 1) exposes fixtures that reproduce only on the platform that wrote them. | It did: the five spectral-start fixture tests fail on the ubuntu runner and skip there under `only_on_fixture_platform` with the id BL-50; the random-start refactor snapshot passes at 1e-9 on both platforms, so the guard this plan rests on is platform-stable. |
| A fixture regenerated in slice 4 hides a numeric change made in the same slice. | Slice 4 lands in two steps inside one branch: first the new loop with the old result shape, and a test that the per-step losses equal the old loop's `loss_history` element for element on one seed; only once that test passes is the result reshaped and the fixtures regenerated, with the PR naming each regenerated file (A-2). |
| The torch-free suite goes red because an abstract base or a test imports torch at module level. | Every abstract base lives in `math/` proper and is array-agnostic (P-6); every concrete torch class lives under `math/algorithms/two_hot_span/`; every torch test opens with `pytest.importorskip("torch")`. CI's default job, which has no torch, is the check. |
| The step-rule abstraction leaks torch into `math` through its annotations. | `AbstractStepRule` annotates parameters and gradients as `Any`; the torch types appear only on `AdamStepRule`. A test imports `mllib.math.algorithms.abstract_optimizer` with torch absent (`sys.modules` guard) and asserts it loads. |
| The `configuration` record changes the stress harness's frozen rows. | The stress fixtures compare rows on named fields (`collision_measures`, `labels`, `max_zero_sum_violation`, `final_training_loss`, the cut numbers); `configuration` is added as one more compared field, the existing fields keep their values, and the fixture regeneration in slice 4 is the one place the rows change. |
| BL-46's restarts turn out to be an outer loop over runs, not a variant of the loop. | Then restarts are a composition-root concern — a harness runs the optimizer N times and selects — and `AbstractOptimizer` stays single-implementation until the joint factorization arrives, which is a `_step` override. Either way no existing class is edited, which is the acceptance test. |
| D-31 says `epsilon` is "never reported"; P-4 puts it in `configuration`. | The two say different things. D-31's test guards the reported *numbers* — E\*, Ê and their differences never pass through the ridge projector — and stays. `configuration` is the configuration record, and a knob that shaped the run belongs there (D-28). The slice-2 PR states this reading in one sentence beside D-31's id. |
| The earned-class rule (P-1) is applied unevenly, and slice 1 grows a class per penalty *form*. | Each slice's PR lists every new class with the one of the three tests it passes; the reviewer-agent checks the list. A class that passes none becomes a function in review, not in a later slice. |

## 5. Immediate next actions

1. **Slice 0 PR checklist.** `uv run pytest` green on `docs/object-model-philosophy`; the glossary
   test passes with the seven new headings; `docs/ARCHITECTURE.md` §1 no longer says "deliberately
   literal" and §2 names the optimizer contract; the `CLAUDE.md` read-order line points at the
   philosophy doc; BL-48 is `in-progress`, BL-49 `backlog-only`; D-35 cites this plan. Xavier
   commits and opens the PR with the reviewer-agent's report.
2. **Slice 1 opens** from the slice-0 branch on `refactor/optimizer-object-model-1`: read D-35 (1),
   (3) and (6) and this plan's §3 row; decide the adjacency form's shape (one class with one knob,
   or two classes) by P-1 and write the reason in the class docstring as one clause.
3. **Acceptance sentences**, to be checked by the reviewer-agent when slice 4 opens and again when
   BL-46 and BL-41 open: "the BL-46 joint factorization is a `_step` override or a new
   `AbstractOptimizer` subclass and edits no existing class"; "the BL-41 ncut objective is a new
   `CostFunction` implementation and edits no existing class".
