# MlLib Architecture

Living document. Promoted from `docs/reviews/2026-07-18-architecture-review.md` on 2026-08-28; the
review stays as the dated record, this file tracks the current and target shape. Update it in the same
PR as any change to a layer boundary, a base-class contract, or an extension point.

## 1. Decomposition

Knowledge is sorted by *kind*, not by feature, with a strictly one-way dependency flow:

```
examples ──▶ visualization ──▶ ml ──▶ math
    ▲
  data
```

Math never imports ML; ML composes math; `visualization` sits above both and nothing below it
imports it; scripts (composition roots, in `examples/`) wire data to models.

| Layer | Contents | Role |
|---|---|---|
| `math` | `HypothesisFunction`, `HypothesisExpander`, `LossFunction` (MSE, MAE, Perceptron, Hinge), `AbstractCostFunction` (a scalar of the parameters being optimized, D-35 (2); `SpanCost` in `graph/two_hot_span_problem.py` is its first implementation), `projector.py` (`AbstractProjector`, `ExactProjector`: the projection onto a spanning set's span, read through its residual; the exact pseudo-inverse is the reporting arithmetic, D-31), `AbstractRegularizationFunction` (a penalty added to a training loss, D-35 (2); its two-hot implementations live in `algorithms/two_hot_span/penalties.py`), `SearchCostFunction`, `secular_equation` (eigenvalues of a rank-one downdate), `graph/` (graph, tree, `SplitFunction`/Gini, `AbstractGraphProblem`, `NystromLandmarkProblem`, `two_hot_span_problem`), `step_rule.py` (`AbstractStepRule`: one run's update rule, bound once) and `learning_rate_schedule.py` (`AbstractLearningRateSchedule` with its four members), `algorithms/` (BFS, DFS, A* on an ABC; Nyström landmark selectors; the `two_hot_span/` package (today `penalties.py`, `projectors.py` and `step_rules.py`, the `AdamStepRule`; `AbstractOptimizer` arrives with BL-48 slice 4)), `probability/` | academic ideas as classes, or as functions when a class is not earned (D-35) |
| `ml` | `MyLinearRegression`, `MyLogisticRegression`, `MyPerceptron`, `MySVM`, `DecisionTree`, `ProbabilisticKNN`, `evaluators/`, `projects/` (ad-click grids, Nyström UCI harness) | models composed from primitives |
| `data` | `data_orchestrator` (+ `DataTransformer`), datasets, transformer JSON configs | load → transform → split |
| `visualization` | `recorders/` (per-problem `Frame` + recorder children, e.g. `astar_landmark`), `recording.py` (the versioned JSON document), `html_renderer.py` + `template.html` + `walkthrough.js` (one offline page with a stepper), `views/` (a problem's drawing plus its layout), `render.py` (the CLI) | a run turned into frames a reader can step through, and a page to step through them in |
| `examples/` | `ad_click_model_comparison.py`, `nystrom_batched_bounds.py`, `graph_search_vs_networkx.py`, `boston_housing_vs_sklearn.py` | composition roots / experiments |

A class is earned, not granted by naming (D-35): a concept is a class when it is injected, when it
has two or more implementations, or when it carries state across calls, and is otherwise a function
named after the concept; the abstract statement is `docs/philosophy/mllib-object-model.md`. The
familiar examples hold under the rule — h(x)=w·x+b is `HypothesisFunction` and the feature map Φ is
`HypothesisExpander` because both are injected; per-sample loss and dataset cost stay distinct
because each is injected somewhere; each loss carries its own gradient because losses are plural.

## 2. Contracts (what a new component must satisfy)

**Injected math objects.** A model is a host for injected objects and is written once against these
interfaces. Adding a loss or an expander never touches a model.

**Knobs and configuration.** A number or a name is a knob owned by the object it parameterises, with
its default on that object's concrete class. An algorithm's `configuration` (D-28) is its own
frozen dataclass plus each injected object's parameters as `describe()` reads them, keyed by role.
A grid is over factories: a cell is a configuration plus a constructor per injected role, built
fresh per cell, never a mutated instance (D-35 rules 3–5).

| Interface | Must provide | Used by |
|---|---|---|
| `LossFunction` | `compute_loss(actual, predicted)`, `compute_gradient(actual, predicted[, data_values])`; margin losses also `compute_bias`; class attribute `task_kind: TaskKind \| None` | every gradient-descent model |
| `HypothesisExpander` | `expand_hypothesis(weights)`, `fit_data_to_hypothesis(data)` | `HypothesisFunction` |
| `HypothesisFunction` | `compute_prediction(x)`, `compute_classification(x)`; owns weights, bias, expander | every model |
| `SplitFunction` | impurity of a candidate split | `DecisionTree` |
| `Transformer` | `fit(frame) -> self`, `transform(frame) -> new frame`, `fit_transform` | `TransformerPipeline`, `DataOrchestrator` |
| `TransformerPipeline` | a `Transformer` of `Transformer`s; `from_config([{transformer, args}])` resolves names in `mllib.data.transformers` only | `ProjectTransformations` |
| `ModelEvaluator` | `update_testing_prediction_data(...)`, `evaluate_model()`, `persist_evaluation_record()` | every model's `evaluate` |
| `AbstractGraphAlgorithm` | `_search(ctx)`; the ABC owns `run()` and `_notify_evaluator()`; `SearchContext` is frozen | graph algorithms |
| `AbstractRecorder` | `enabled` (class attribute), `frames`, `metadata`, `record(frame)` (dense, ordered), `frame_dicts()`, `describe_result(result)`; `AbstractSearchRecorder` adds `record_expansion(...)` and `record_goal(...)` | any algorithm that offers to be watched |
| `AbstractOptimizer` | `_begin()` (prepare the parameters, may `Stop`), `_step(step)` (one step; the training loss or a `Stop`), `_iterate()` (the current parameters for the recorder), `_assemble(...)` (the result), `_recorder_extras()`; the ABC owns `run()` — one per optimizer, refused twice — the recorder guard, the stop check and the end record; `configuration` is a property; `OptimizerResult` carries the parameters moved, the final training loss, the steps taken, a `StopReason`, its detail, `configuration` and the delivered numbers, never a per-step frame (D-35 rule 9) | `TwoHotSpanOptimizer` |
| `TwoHotSpanProblem` | the instance an optimizer moves a spanning set over: `X`, `laplacian`, `adjacency`, `constraint_vector` (ones for rcut; sqrt(d) is BL-41's override), `report(spanning_set)` through the exact arithmetic, `configuration` | `TwoHotSpanOptimizer`, the harnesses |
| `AbstractCostFunction` | `compute_cost(parameters)`: a scalar of the parameters being optimized, differentiable in the training arithmetic; optional `task_kind` | `training_loss` of the two-hot optimizer (`SpanCost`); `TwoHotSpanOptimizer` |
| `AbstractProjector` | `residual(X, spanning_set)`: ‖X − P_V X‖²_F, the projection onto V's span read through what it leaves of X; `ExactProjector` (numpy `pinv`, no knob) for every reported E\* and Ê, `RidgeProjector` (torch, `epsilon` a knob) for training, and never the other way round (D-31) | `SpanCost`; `rounded_cut`; the optimizer's reported numbers; the two-hot recorder and stress harness |
| `AbstractRegularizationFunction` | `compute_penalty(parameters)`: the signed, weighted scalar a training loss adds, its weight a knob on the concrete class; `term(parameters)`: the unweighted quantity the term measures, the number a hand check reads | `training_loss` of the two-hot optimizer; `TwoHotSpanOptimizer` |
| `AbstractStepRule` | `bind(parameters, step_count)` once per run, refused twice; `learning_rate_in_force()`; `zero_gradient()`; `step()`: move the parameters by the gradient they hold and advance the schedule. The learning rate and the schedule are its knobs; its state (Adam's moments) is never part of a configuration | the two-hot loop (`TwoHotSpanOptimizer`); `TwoHotSpanOptimizer` |
| `AbstractLearningRateSchedule` | `multiplier(step, step_count)`: the dimensionless factor on a step rule's learning rate; its knobs are its shape, the horizon is passed per call | injected into a step rule |

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

`DataOrchestrator(csv, 'csv', config.json)` → `ProjectTransformations` builds one `TransformerPipeline` per
frame declared in `data/configs/ad_click_transformations.json` (transformers resolved by class name) →
`get_transformed_data(frame_name)` drops the configured target →
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
- **A search over an implicit problem:** implement `AbstractGraphProblem` (`initial_state`, `is_goal`,
  `successors`) and inject a `SearchCostFunction` (`lower_bound`, `goal_cost`) into `AStarSearch`. The
  bound must be admissible and must equal `goal_cost` at a goal state, or the search stops being a proof
  (D-23). `AStarSearch` is exact and stores every child; `PrunedAStarSearch` is the variant that stores
  less (goal-sibling filter, incumbent pruning, frontier cap) and rests on the same equality; a seeded
  incumbent must be an exact objective, never a truncated bound (D-27). `AnytimeAStarSearch` is the
  pruned variant that returns at a cap instead of raising: the incumbent, `optimal=False`, and the
  gap it certified, on `SearchResult.certified_gap` (what the search *proved*, D-28 amended). States must be hashable and
  canonical, so one position is one node. When a parent's successors share work, override
  `lower_bounds(parent, successors)`; the default scores them one at a time (D-24). A conditional
  solve (a subset that must, or must not, contain given columns) is the problem's knob, `forced` and
  `forbidden`, never the cost function's: the bound is unchanged and the harness records the
  constraints beside the engine's settings (`problem_constraints`, BL-39).
- **A search variant:** subclass `AStarSearch` and override only the step where it diverges —
  `_price_children`, `_push_children` (return the insertion index as if every child were pushed, so pop
  order is unchanged) or `_no_goal_reachable`; never `_search`'s loop. Test it against the base class.
  To measure it on the harness's cells, register it rather than replacing anything:
  `AStarLandmarkSelector(name="astar-<variant>", search_factory=...)` appended to
  `default_selectors(sample_seed)` and passed as `run_nystrom_on_uci_dataset(..., selectors=...)`.
  The name `"astar"` must stay in the list; it is the reference every ratio is taken against (D-28).
  A variant that adds a knob overrides `configuration` to name it, or its rows go on the record
  without the settings that produced them (D-28, amended). The base engine has one opt-in
  measurement of its own, `count_bound_drops` (`BoundDropCounter`, with `bound_drop_slack` as the
  caller-stated rounding allowance): whether a child's bound ever fell below its parent's as priced.
  Off by default, it changes no expansion or result and leaves its counts on the instance as
  `bound_drops`, not on `SearchResult`, which carries only what the search paid and how it was set up.
- **A recorder:** subclass `Frame` in `visualization/recorders/<problem>.py` with the fields that
  problem's reader needs and extend `to_dict` explicitly (name the base — `Frame.to_dict(self)` —
  since `slots=True` breaks a zero-argument `super()`); subclass `AbstractSearchRecorder` (or
  `AbstractRecorder` for a non-search algorithm), convert every value to plain Python and write a
  caption sentence. The engine injects it (`AStarSearch(..., recorder=...)`), defaults to
  `NullRecorder`, and calls it from one guarded call site per recorded moment — for a search two of
  them, `record_expansion` at each state advance and `record_goal` at the goal it returns on, so a
  walkthrough ends at the answer rather than one step before it; a variant contributes its own
  state through `_recorder_extras()` and never overrides `_search`. Observation is never
  part of a result (D-28, D-32): the frames stay on the recorder the caller constructed. `math`
  declares the recorder's shape and never imports `visualization`.
- **A transformer:** subclass `data.transformers.Transformer` (`fit` learns state and returns `self`;
  `transform` returns a new frame, never mutating); add it to `transformers/__init__.py`; after 6.2 declare it
  by class name in the project's JSON config.
- **A penalty:** subclass `AbstractRegularizationFunction` as a frozen dataclass in
  `math/algorithms/two_hot_span/penalties.py`; implement `term(parameters)` (the unweighted
  quantity) and `compute_penalty(parameters)` (`±weight * term`, negative for a reward), with the
  weight as a knob on the class; never read the cost or another penalty. Add two tests: a hand
  value on a column small enough to read, and `torch.autograd.gradcheck` on a V with no zero
  column. It enters a run by injection only (D-35 rule 3); a weight of zero is not a switch, the
  composition root leaves the penalty out; the optimizer is not edited.
- **A projector arithmetic:** subclass `AbstractProjector` as a frozen dataclass, knobs on the
  class; implement `residual(X, spanning_set)`. Add one test that it approaches `ExactProjector`'s
  residual as its knob goes to zero on a full-rank V, and one against a pasted closed form. It is
  injected into `SpanCost` for training only; a reported number never comes through it (D-31).
- **A step rule:** subclass `AbstractStepRule` in `math/algorithms/two_hot_span/step_rules.py`;
  implement `bind` (once per run, refuse a second), `learning_rate_in_force`, `zero_gradient` and
  `step`; own the learning rate and the schedule as knobs with defaults on the class; keep the
  state on the instance and out of equality. Test it `torch.equal` after every step against the
  hand-built pair it wraps, or against a closed-form step on a quadratic. A grid over learning
  rates is a grid over step-rule constructors (D-35 rule 5).
- **A schedule:** subclass `AbstractLearningRateSchedule` in `math/learning_rate_schedule.py` as a
  frozen dataclass whose fields are its shape knobs only; implement `multiplier(step, step_count)`.
  Add one exact-sequence test against a hand formula over the whole horizon, and never let it
  into a report (D-31).
- **An optimizer variant:** subclass `AbstractOptimizer` and fill `_begin`, `_step`, `_iterate`
  and `_assemble`, or change nothing and inject a different step rule, cost or penalty into
  `TwoHotSpanOptimizer` — never `run()`. A variant that adds a knob puts it on its own settings
  dataclass so `configuration` names it (D-35 rule 4). Test it against the refactor snapshot on
  the same cell and record both under the harness's `configuration`. A composition root builds it
  through `mllib.ml.projects.two_hot_span_composition.compose_two_hot_span` or by naming the
  classes itself; a run stops with a `StopReason` and never raises for a numeric stop (D-35 rule
  9). BL-46 and BL-41 are the two variants the shape is judged by: restarts are a composition root
  over fresh optimizers, the joint factorization a subclass, and ncut a `TwoHotSpanProblem` whose
  `constraint_vector` is sqrt(d) plus its cost — each without editing an existing class (D-35).
- **Where a sentence lives:** one home per kind of sentence, D-35 rule 7 and
  `docs/philosophy/mllib-object-model.md` §7; a module docstring cites the decision that binds it as
  one clause plus the id, and a class docstring is the description `describe()` reads.

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
| Descent stack breaks injection: `MyLogisticRegression` builds its own hypothesis and loss and mutates them per grid cell; `LossFunction.compute_gradient` has a 2-arg and a 3-arg form | `ml/logistic_regression.py`, `math/loss_function.py` | BL-49 (D-35) |
| `HypothesisFunction.expand_hypothesis` calls an expander `expand(hypothesis, degree)` no expander defines | `math/hypothesis.py:51` | fix slice 5 of `docs/plans/2026-09-optimizer-object-model.md` |
| The Laplacian has two derivations, D − A from the graph (`laplacian_matrix`) and X Xᵀ from the incidence matrix (`graph_matrices`), each pinned by its own fixtures; they differ by ulps the spectral start amplifies | `math/graph/two_hot_span_problem.py` | with BL-50: whichever fixture set is regenerated deliberately takes the other's derivation |
| `CostFunction` and `RegularizationFunction` had no implementation; the two-hot objective and penalties were free functions in one module | `math/cost_function.py`, `math/regularization_function.py`, `math/algorithms/two_hot_span_optimizer.py` | **done** BL-48 slices 1–4: `AbstractRegularizationFunction`, `AbstractCostFunction`, `AbstractOptimizer` and `TwoHotSpanOptimizer` |

## 6. Target layout (after Phase 4)

```
pyproject.toml  uv.lock  CLAUDE.md  README.md  CONTRIBUTING.md
src/mllib/
  math/      hypothesis.py  hypothesis_expander.py  loss_function.py  cost_function.py  …
             recorder.py  graph/  algorithms/  probability/
             algorithms/abstract_optimizer.py  projector.py  step_rule.py  learning_rate_schedule.py
             algorithms/two_hot_span/   optimizer, step rules, projectors, penalties (torch; BL-48)
  ml/        linear_regression.py  logistic_regression.py  …  evaluators/  projects/
             projects/two_hot_span_composition.py   names and knobs → objects (composition roots)
  visualization/  recorders/  views/  recording.py  html_renderer.py  render.py
                  template.html  walkthrough.js
  data/      orchestrator.py  transformers/
tests/       mirrors src/mllib; baseline snapshot beside its test; committed recordings in
             tests/visualization/fixtures/
data/        datasets (≤ 1 MB each, D-19)  configs/
examples/    composition roots
notebooks/   docs/       philosophy/ (the abstract object model, D-35)  plans/  reviews/
```

## 7. Cross-codebase note

MlLib independently evolved the same instincts as tradePlatform (strategy injection, two-tier
specialisation, self-describing metadata) and the same failure modes (hardcoded enumerations,
declared-intent-left-stubbed, descriptor drift). When MlLib models plug into tradePlatform as strategy
plugins (BL-19), descriptors are introspected at the boundary so external models self-describe on
arrival — which is why R3 (introspected metadata) precedes any plugin work.
