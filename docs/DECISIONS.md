# Decisions

Append-only log of decisions with lasting consequences (ADR style). Supersede rather than edit.
D-1…D-13 were made 2026-08-27; D-14…D-20 were the plan's assumptions A-1…A-7, confirmed 2026-08-28.

## D-1 — The framework is both engineering and code framework · 2026-08-27 · accepted
**Context.** "Framework" was ambiguous. **Decision.** Process/gates/docs *and* base-class contracts; the
engineering framework ships first. **Consequences.** Phases 1–2 precede any code refactor.

## D-2 — Audience and north star · 2026-08-27 · accepted
**Decision.** Xavier, with a professional lens; tradePlatform plugin-source is the north star, not a current
dependency. **Consequences.** API stability matters at the plugin seam (BL-19), not yet elsewhere.

## D-3 — Delete `fastapi_app` · 2026-08-27 · accepted
**Decision.** Removed in slice 0.3; contract preserved as BL-01. **Consequences.** No web layer in this repo;
a future seam is built in tradePlatform.

## D-4 — Strip rule · 2026-08-27 · accepted
**Decision.** Cleanly removable stubs are deleted; anything whose clean removal exceeds ~1 hour or needs an
intensive session to re-add is kept and tagged `TODO(BL-nn)`. Never-imported modules stay if tested; a named
set of untested ones also stays (BL-14). **Consequences.** `docs/BACKLOG.md` §3 of the plan is the record.

## D-5 — Deleted initiatives are recorded · 2026-08-27 · accepted
**Decision.** In `docs/BACKLOG.md`, with re-entry cost.

## D-6 — Branching · 2026-08-27 · accepted
**Decision.** Feature branches + PR into protected `main`; CI required; `main` always green.

## D-7 — Project-level tracking lives in the orchestrator · 2026-08-27 · accepted
**Decision.** Phase milestones and `next` in `orchestrator/projects/botmaker.md`; code-adjacent detail here.

## D-8 — Xavier commits · 2026-08-27 · accepted
**Decision.** Claude prepares reviewable slices grouped in phases; Xavier reviews and commits every change.

## D-9 — GitHub Actions replaces AWS CodeBuild · 2026-08-27 · accepted
**Consequences.** `buildspec.yaml` and `Dockerfile` removed (slice 0.3). *Update 2026-08-29:* the CodeBuild GitHub
webhook (id 574383043) is still registered and fails on every push; it is not a required check. Deletion pending (owner).

## D-10 — Fix F4/F5 in the migration · 2026-08-27 · accepted
## D-11 — R1–R3 are in scope as scheduled phases · 2026-08-27 · accepted
## D-12 — Delete stale checkouts · 2026-08-27 · accepted · done 2026-08-28
## D-13 — Plan lives in-repo plus a shareable page · 2026-08-27 · accepted

## D-14 — Cross-repo standards live in `engineering-standards` · 2026-08-28 · accepted
**Decision.** github.com/xavierlmendez/engineering-standards; repos copy fragments with a source header.
The orchestrator stays the state system. **Consequences.** Improvements flow back by PR.

## D-15 — Tracking split · 2026-08-28 · accepted
**Decision.** Phase-level items in the orchestrator; per-file registry in `docs/BACKLOG.md`.

## D-16 — Python 3.12 · 2026-08-28 · accepted
## D-17 — PEP 8 `snake_case` modules, methods, functions · 2026-08-28 · accepted
**Consequences.** One mechanical rename slice per domain (Phase 4); `.git-blame-ignore-revs` lists them.

## D-18 — `src/` layout with a single `tests/` tree · 2026-08-28 · accepted
**Context.** Mixed import roots caused the collection error in the baseline.

## D-19 — Dataset size ceiling 1 MB per file · 2026-08-28 · accepted
**Decision.** The two current CSVs (465 KB, 417 KB) stay in git; larger datasets are fetched by script.

## D-20 — Reports and notebooks are kept, not deleted · 2026-08-28 · accepted
**Decision.** `docs/reports/`, `notebooks/`.

## D-21 — Refactor plan complete; the framework is the steady state · 2026-08-29 · accepted
**Context.** `docs/plans/2026-08-refactor.md` executed as PRs #1–#21 (Phases 0–7). **Decision.** From here, the
rules in `CONTRIBUTING.md` are not migration rules but the way the repository works: protected `main`, PR + CI,
one-concern slices, `TODO(BL-nn)` only, tests with every change, decision and learning entries in the same PR.
New work (CS 6344 pairings) starts from the open backlog. **Consequences.** `docs/REFACTOR_PLAN.md` no longer
exists; plans live under `docs/plans/` and are archived, not deleted, when done.

## D-22 — Optimality-gap harnesses report named baselines and both gaps · 2026-09-04 · accepted
**Context.** The first Nyström harness compared A* against a greedy that minimised A*'s own lower bound
(a lookahead heuristic nobody publishes) and against the best of 32 random draws labelled "random". Both
inflated the apparent gap between heuristics and the optimum.
**Decision.** Any harness that measures how far a heuristic is from a certified optimum must: name published
baselines and instrumented heuristics separately; quote randomised selectors as a mean and median over
independent seeds, and show a single draw only when it is labelled as one; carry the sample count in the
name of a best-of-N selector; report the SVD rank-k residual and the subset-to-SVD ratio next to every
algorithm-to-optimum ratio; and print the exact solver's node count next to the size of the search space.
**Consequences.** Two ratios, not one: an algorithm can only be called bad once the best subset is known to
be good. The naming rule is enforced by the selector's `name`, so a report cannot relabel it.

## D-23 — Graph algorithms accept implicit problems as well as materialized graphs · 2026-09-04 · accepted
**Context.** `AbstractGraphAlgorithm` was written against `Graph`, whose nodes and edges all exist before
search starts. Subset-selection search has C(n, k) states and cannot materialize them.
**Decision.** `AbstractGraphProblem` is a second input type for the same base: it supplies `initial_state`,
`is_goal` and `successors` on demand. The template method stays as it is — subclasses implement `_search`
and never override `run` — and `run` takes an optional context, since a problem carries its own start state.
**Consequences.** `ARCHITECTURE.md` §4 gains the rule. `run` without a context raises on a materialized
`Graph`, which has no start state of its own. `SearchCostFunction` is the matching injected contract for
algorithms that need costs and admissible bounds, and it is *terminal-objective only*: cost belongs to a
completed solution, not to the path, so there is no edge cost and the priority is the bound alone. A
problem whose cost accumulates along edges needs a contract this repository does not yet have.

## D-24 — Batch scoring of a parent's successors is the cost contract's extension point · 2026-09-04 · accepted
**Context.** Profiling the Nyström search at n = 40 showed 91% of A*'s time in scoring goal-depth children
one at a time, each with its own pseudo-inverse, when all children of one parent share the parent's
residual and differ by a single column. This is not a Nyström accident: every subset-selection problem has
it, and three of the four research candidates are subset selection. A GPU is the wrong lever — the cost is
per-call overhead on 3×3 matrices, Apple Silicon has no CUDA, and no available library runs symmetric
eigendecomposition on Metal. NumPy is already linked against Accelerate.
**Decision.** `SearchCostFunction.lower_bounds(parent, successors)` scores all of a parent's successors
together. Its default calls `lower_bound` per child, so a cost function with nothing to share implements
only `lower_bound`. A search must call `lower_bounds` once per expansion, preserving successor order so
tie-breaking is unchanged. An override must return what `lower_bound` would, up to rounding.
**Consequences.** `AStarSearch` and the greedy-on-bound selector use it. The Nyström override prices every
goal-depth child from one Schur complement; non-goal children fall back, because what they share is the
eigendecomposition itself, and sharing that is BL-27's rank-one downdate — which now has a method to live
in. A tensor library is not adopted; the reasons above are recorded so the question is not reopened
without new evidence.

## D-25 — A slice is a vertical slice of functionality, not a line count · 2026-09-04 · accepted
**Context.** `CONTRIBUTING.md` defined a slice as one concern under ~300 changed lines. Measured against
that, six of the eight Nyström slices were "over", and the only way to comply would have been to ship a
cost function in one PR and the tests that pin its numerics in another — which is exactly the split the
slicing discipline exists to prevent.
**Decision.** A slice is one end-to-end piece of functionality that can be run, tested and judged on its
own: contract, implementation, tests and records together. Its size follows from the smallest such piece.
Splitting means finding a smaller end-to-end piece, never delivering a layer without the rest. The line
ceiling is retired.
**Consequences.** `CONTRIBUTING.md` here and in `engineering-standards` say so (back-port by PR). The
Nyström plan's deviation 4 is void: those slices were vertical and the right size. Reviewers judge a
slice by whether it is complete end to end, not by its diffstat.

## D-26 — The Nyström bound may be computed on a truncated spectrum; the objective never is · 2026-09-04 · accepted
**Context.** After BL-27 the search decomposes each parent once, but for a full-rank kernel that is still
n³ per parent, and the field-scale experiment (BL-28, EXP-09) needs n ≥ 1,000. Dropping the smallest
eigenvalues at the root makes every parent cheap, but a bound computed on the wrong kernel is only useful if
it is still a lower bound on the right one.
**Decision.** `NystromLandmarkProblem` takes `spectrum_mass_tolerance` δ (default 0). The retained rank is the
smallest whose dropped mass is within δ·tr(K), never above the numeric rank and never below one; the reduced
coordinates, `kernel_sqrt` and therefore both bound paths (oracle and downdate) see the truncated kernel K̃,
so D-24's "same value both paths" holds at every δ. `kernel_matrix`, `goal_cost` and the goal-depth batch
stay exact on K. No correction term is added to the bound.
**Why it is admissible.** The Schur complement is monotone on the PSD cone: K̃ ⪯ K implies every residual
kernel of K̃ is ⪯ the corresponding residual of K, so the best completion under K̃ is never worse than under
K, and a bound that is admissible for K̃ is admissible for K (learning-log entry of the same date). Tightness
is not preserved and is measured, not proven (EXP-09a).
**Consequences.** Every result record must carry its δ; a number without one is not comparable. δ = 0 keeps
the search baseline byte-identical. The threshold rule for an explained column is left split (BL-30).

## D-27 — Pruning bounds must be computed on the untruncated objective · 2026-09-05 · accepted
**Context.** D-26 lets the admissible bound be computed on a truncated spectrum. A pruning rule
discards states whose bound exceeds an *upper* bound on the optimum. The two bounds move in opposite
directions under truncation: the lower bound gets smaller and stays admissible; an upper bound derived
from it, such as the (k̄ + 1)·f rule of AAAI-15 §5, also gets smaller and stops being an upper bound.
**Decision.** A pruning threshold is either an exact goal cost seen during the search (always on the
full kernel, D-26) or a quantity derived from the untruncated root bound, (k + 1)·E*_svd. The
(k̄ + 1)·f̃ rule on truncated per-state bounds is not admissible for pruning and is not used.
**Consequences.** Incumbent pruning (P-14) is valid at every δ. The per-state Deshpande–Rademacher
rule, if built, applies only at δ = 0 or through the root. Every pruning rule ships with the test
that expansion counts and returned subsets are unchanged on the search baseline and the reference cells.

## D-28 — The engine is a parameter; the reference is a name · 2026-09-06 · accepted
**Context.** Pass-2 wants variant engines (anytime, weighted, focal, bounded) measured on the harness's
own cells against the certified optimum. Before this the engine was hard-coded in `AStarLandmarkSelector`
and the UCI runner's selector list, so a variant could only be measured by a bespoke script, on cells
nobody else runs, against numbers computed elsewhere.
**Decision.** Three parts, all of them, not a choice among them. (i) `AStarLandmarkSelector` takes
`search_factory: (problem, cost_function) -> AStarSearch`, defaulting to exact `AStarSearch`, and `run_nystrom_on_uci_dataset` takes `selectors=None`,
which builds the default suite when omitted — so a variant is *added* under its own name, never
substituted for the reference. (ii) The certified reference is identified by the name `"astar"`
(`CERTIFIED_SELECTOR`), and a selector list without it is refused with a `ValueError` rather than
ratioed against whatever else is present: every ratio, and `subset_to_svd_ratio`, is defined relative to
the certified optimum, so a run without one produces numbers that look like results and are not.
(iii) The memory an engine paid rides out on the result: `SearchResult` gains
`frontier_peak: int | None = None`, `None` meaning "no engine counted", and `UciHarnessResult` gains
`frontier_peaks` keyed like `selector_results`. One optional field on the shared type was chosen over a
`SelectorRun` wrapper: every existing selector keeps working through the default, and no second result
type has to be kept in step with the first.
**Amended 2026-09-06 (PR #35), part (i).** A factory is opaque: `PrunedAStarSearch(p, c,
incumbent_slack=..., max_frontier=...)` binds its knobs inside the caller's lambda, so the harness
recorded *that* a variant ran and never *how*. D-26 already rules that a number without its δ is not
comparable; a cost is likewise not comparable across engine settings, and pruning knobs change the
measurement while leaving the answer alone — which is precisely the pair that produces two rows that
look alike and are not. So an engine describes itself: `AStarSearch.configuration` returns its
JSON-serialisable settings (base: the class name), `PrunedAStarSearch` adds `incumbent_seed`,
`incumbent_slack`, `max_frontier`, and the selector reads it off the instance it just ran, landing in
`SearchResult.engine_configuration` and `UciHarnessResult.engine_configurations`. Read off the object,
never accepted from the caller: a declared config could disagree with the lambda, and a record that
misdescribes its own run is worse than one that says nothing. It is the complement of
`mllib.describe.describe`, which names the knobs a class has rather than the values one instance got.
A variant that forgets to override shows only its class name — incomplete on its face, not silently
wrong. The printed block states a configuration only for engines differing from the reference's, so a
default run's output is byte-identical.
**Amended 2026-09-07 (BL-33).** The base configuration is no longer the class name alone: `AStarSearch` has
one measurement of its own, the bound-drop counter, and its two knobs (`count_bound_drops`,
`bound_drop_slack`) are stated there under the same rule, so a row that ran with the counter on says so.
The counter's result stays on the instance (`bound_drops`), not on `SearchResult`: it is a property of
the bound on the data, neither paid nor set up.
**Amended 2026-09-07 (anytime A\*, `feat/anytime-astar-gap`).** The first variant that stops before it
pops a goal produced a number the two kinds cannot hold: the a-posteriori gap `incumbent − frontier_min`
is neither what the search paid nor how it was set up. It is what the search *proved*, and that kind
already exists on the type — `optimal` is a statement about `cost`, not a cost or a setting. So the rule
gains its third kind: `SearchResult.certified_gap: float | None = None`, an additive certificate on
`cost`. Exact `AStarSearch` sets `0.0` on the goal it pops (a popped goal proves a zero gap), a stopped
`AnytimeAStarSearch` sets the finite gap it certified, and every selector that proves nothing leaves
`None` — "no certificate", never a fabricated zero, on the same reading as `frontier_peak`. `optimal`
implies `certified_gap == 0.0`; the converse is not promised, because a stop whose gap clamps to zero
did not pop its incumbent as a goal and the flag means only that. The alternative — the gap on the
instance, as `bound_drops` is, read off the engine through a new selector return path — was rejected: a
gap is about the answer, not about the bound, and every future certificate-bearing variant would repeat
the plumbing. The selector's `replace(...)` pass-through carries the field into
`UciHarnessResult.certified_gaps` beside `frontier_peaks`. Two guards land with it, deferred from PR #34:
the harness refuses a run whose selector named `"astar"` returned `optimal=False` (the name finds the
reference; the certificate makes it one), and `selector_kind` reads the result, not only the name —
`optimum` and `baseline` stay name-keyed (D-22), a variant that popped its optimum prints `certified`, one
a cap stopped with a finite gap prints `bounded` and its row states the gap, and `instrumented` is what
remains. The test that pinned the name-only labelling was changed deliberately in the same PR.
**Consequences.** `None` is "not measured", never zero; a reader who sees a peak knows an engine counted
it. `SearchResult` carries what the search *paid* (`nodes_expanded`, `frontier_peak`), how it was
*set up* (`engine_configuration`) and what it *proved* (`optimal`, `certified_gap`), and that is the whole
rule: a further measure of cost joins the first, a further knob is named inside the second, a further
certificate joins the third, and nothing else is added to the shared type. That
bound is what makes option (a) affordable — the `SelectorRun` wrapper was rejected for being larger
today, and this is the line that stops it being larger tomorrow. Defaults are a value (`default_selectors`), not a behaviour, so the
suite's order and seeds are pinned by test and remain part of every committed number.

## D-29 — Ties on the frontier are defined by a caller-stated absolute tolerance; under it the certificate is additive · 2026-09-07 · accepted
**Context.** The spectral bound is computed by cancellation, so two bounds equal in exact arithmetic
agree only to rounding, and a tie-break that fires on exact equality never fires on real data (research
candidate C9, cell C2). Making near-ties equal needs a tolerance, and a tolerance changes what the first
goal popped proves: a smaller raw bound may remain in the goal's cell.
**Decision.** `AStarSearch.tie_tolerance` is absolute, in the objective's units, stated by the caller
(a Nyström caller passes a small multiple of the kernel trace, the same shape and reason as
`incumbent_slack` and `bound_drop_slack`); the engine never infers a scale. The heap key is the bound
quantised to that grid (floor to a multiple), so the default tolerance of zero is the raw bound and the
reference order exactly. When a goal pops under a tolerance the engine scans the frontier for the
smallest raw bound once and certifies exactly (`optimal=True`, gap 0) only if the goal's bound is at or
below it; otherwise it returns `optimal=False` with `certified_gap = goal bound − frontier minimum`, which
is below the tolerance by construction. `optimal` keeps one meaning under every knob: the popped goal was
the frontier minimum. The pruned and anytime engines hold an incumbent, and under a tolerance a goal in
its cell can pop before it; they then return the incumbent (its state is now kept beside its cost) with
the gap recomputed against the same frontier minimum, certified only if it came from a priced goal.
`tie_break="deepest"` orders equal keys by `-len(state)`, then insertion; it needs sized states and is the
only alternative to first-in-first-out.
**Consequences.** Both knobs are on `configuration`, so a row that ran with a tolerance says so (D-28). A
tolerance turns the exact certificate into an honest additive one and is therefore a different
measurement, never a default. Every engine variant inherits the key through `_heap_entry` and reads the
frontier minimum through `_frontier_minimum`; a variant that builds heap entries by hand loses both knobs.

## D-30 — A conditional solve is the problem's knob; the cost, the bound and the selectors do not see it · 2026-09-07 · accepted
**Context.** The necessity margins need, for every column, the best subset that must contain it and the
best that must not: the same certified search on a modified ground set (BL-39, research candidate E3).
Three places could hold the constraint — the problem, the cost function, or the engine — and the harness
had to decide what a ratio means when the problem is not the one the baselines solved.
**Decision.** `NystromLandmarkProblem` takes `forced` and `forbidden` and shapes its tree accordingly:
`initial_state` is the forced columns, `successors` add free columns only, every admissible subset is
one canonical node. The bound is untouched, because removing candidates only raises the optimum and
starting deeper is a subtree of the tree the bound already covers; the objective is untouched and stays
defined on every subset, so `_validate_state` does not enforce the constraints and a selector that never
reads the tree can still price an unconstrained choice. A seeded incumbent's feasibility is the caller's
claim, as its cost already is (D-27); the research driver seeds a conditional solve from the
unconstrained optimum only when that optimum is feasible for it. The UCI harness threads the two knobs to
the problem and records `problem.constraints` as `UciHarnessResult.problem_constraints`, empty by
default; when set, the printed block states them and says that the selectors other than the searches
are unconstrained.
**Consequences.** Under constraints every search row solves the constrained problem and every heuristic
row the unconstrained one, so `cost_ratios_to_optimal` can fall below one and compares two problems; the
row says so, and such a run is a record of a conditional solve, not a baseline measurement (D-22). The
default run is byte-identical: no constraint, no line, the same tree to the byte. A future selector that
wants the constraints reads them off the problem it is given, never off the harness.

## D-31 — PyTorch enters as a local dependency group with the CPU index; the objective is never computed through the regularized projector · 2026-09-10 · accepted
**Context.** The two-hot span prototype (`docs/plans/2026-09-two-hot-span.md`, research candidate
`constrained-eigenvectors`) is the repo's first gradient-optimized module. `main` @ `85f2205` has no
torch: absent from `pyproject.toml` dependencies and groups, zero matches in `uv.lock`, no import
under `src/`. Three shapes were possible — a core dependency, a published extra, or a PEP 735 group —
and the projector `P_V = V(VᵀV)⁺Vᵀ` needed an implementation that stays differentiable when columns
go dependent, without imposing VᵀV = I, which the source forbids.
**Decision.** torch is a PEP 735 `[dependency-groups] torch` group, with
`[[tool.uv.index]] name = "pytorch-cpu", url = "https://download.pytorch.org/whl/cpu", explicit = true`
and `[tool.uv.sources] torch = [{ index = "pytorch-cpu" }]`. CI adds one job running
`uv sync --locked --dev --group torch`; torch-dependent tests use `pytest.importorskip`, so the
default suite runs without it. Not an extra: uv documents extras as published package metadata that
is not synced by default, and dependency groups as local-only and never published — which is what a
research module's solver dependency is. Separately: the **training** loss uses the ridge projector
`V(VᵀV + εI)⁻¹Vᵀ` (filter factors σᵢ²/(σᵢ²+ε), smooth, imposing no orthogonality), while **every
reported number** — E\*, Ê and the differences — is computed with the exact `pinv` projector in
numpy. The objective is never reported through the regularized projector.
**Consequences.** D-21 is not reopened: `CONTRIBUTING.md` stays the steady state and this is an
ordinary decision entry. A contributor without the group installed still gets a green suite. Because
PyPI's default Linux wheel is the GPU build, the explicit index is what keeps a Linux checkout on the
CPU wheel. `epsilon` is a training knob only, and a test asserts the reporting path never sees it —
so a run's numbers mean the same thing whatever ε was. Whether the CPU wheels run on the R620 is
undocumented upstream and is recorded as BL-42, not assumed here.
**Verified 2026-09-10 (slice 2.2).** torch 2.14.0 resolves from the CPU index on macOS arm64
(wheel `macosx_14_0_arm64`) and as `2.14.0+cpu` for Linux; uv writes the platform fork into the lock
itself, so no `sys_platform` marker is needed.

## D-32 — Observation is an injected recorder: abstract in math, children in visualization, off by default, never on a result · 2026-09-10 · accepted
**Context.** Nothing in the tree can watch a run. The evaluators record once per run, not per step
(`evaluator.evaluation_record[iteration]` is one row for a whole fit), and D-28 closed `SearchResult`
to a stated rule — cost, set-up, certificate, and nothing else — so per-step observation has no seat on
a result and must not take one. The only per-run instrumentation that exists, the bound-drop counter
(BL-33), already made that choice: opt-in, left on the instance, never on the result. Meanwhile
`math/graph/visualizer.py` is a matplotlib animation nothing imports. The remaining constraint is the
one-way dependency flow of `docs/ARCHITECTURE.md` §1 — "Math never imports ML; ML composes math;
scripts (composition roots) wire data to models" — which forbids the obvious shortcut of letting an
algorithm import the thing that draws it.
**Decision.** Observation is a collaborator injected into the algorithm, under four rules. (i) An
algorithm takes `recorder: AbstractRecorder` defaulting to a fresh `NullRecorder` (never a shared
mutable default), and every call site is guarded by `recorder.enabled`, so an unobserved run pays one
boolean read per step. (ii) The abstract contract — `AbstractRecorder`, the frozen `Frame`, and
`NullRecorder` — lives in `math/recorder.py`; the per-problem children live in
`mllib.visualization.recorders`, and **the child extracts and captions**: the engine hands it the raw
objects it already holds, and variants add their own through a `_recorder_extras()` hook the base loop
calls, so no variant overrides `_search` and no engine grows a second call site. (iii) A recording is a
versioned JSON document written when the run ends, with exact floats and no timestamps; a walkthrough is
that recording rendered as one offline HTML page. (iv) Nothing observational is added to a result type
(D-28); a recorder's frames are read off the recorder the caller injected.
The new package `mllib.visualization` sits **above** both `ml` and `math`: it may import them, nothing
below it imports it, and the composition roots in `examples/` wire the two together — an algorithm and
the recorder that watches it meet in a script, never inside `math`.
**Consequences.** The two behavioural baselines are the guarantee this rests on: every slice that
touches an engine re-runs `tests/ml/test_training_baseline.py` and
`tests/ml/test_nystrom_search_baseline.py` and leaves both byte-identical, which is what makes "off by
default" a checked claim rather than an intention. Instrumenting a new problem adds a child recorder and
a view; the engine does not change. `CLAUDE.md` gains the commit scope `viz`. Because the pages carry
JavaScript, the TODO-id gate in CI and pre-commit now greps `*.html` and `*.js` beside `*.py` *and*
matches `//` beside `#` as the comment marker — extending the file list alone left the gate hollow,
since no JavaScript comment starts with `#` (a change to back-port to `engineering-standards`), and the templates ship as package data
(`[tool.setuptools.package-data]`), so an installed `mllib` can render a walkthrough. The plan is
`docs/plans/2026-09-visualization.md`, the initiative BL-43.

## D-33 — One recorder contract per problem shape; the null recorder satisfies them all; frames are light or full · 2026-09-10 · accepted
**Context.** D-32 settled that observation is an injected recorder, but it fixed the seam in the
terms a *search* engine speaks: `record_expansion` with the expanded state, its bound, the priced
children and the frontier, and `record_goal` when the goal is popped. An optimizer has none of
those. It advances by steps and holds a step number, a training loss and the current V; a traversal
advances by visits and holds a node, its neighbours and the queue. Three engines, three sets of raw
objects, and one call vocabulary that fits only the first. The choice was between widening the base
contract until it means nothing in particular (`record(**anything)`), giving each engine its own
recorder hierarchy including its own off switch, or one base with a contract per shape.
**Decision.** `AbstractRecorder` stays the shared base — `enabled`, `frames`, `record`, `metadata`,
`frame_dicts`, `describe_result` — and each problem *shape* gets its own abstract contract beside it
in `math/recorder.py`: `AbstractSearchRecorder` for the engines that expand states,
`AbstractStepRecorder` for the ones that take gradient steps, and a traversal contract in slice 4. A
contract is exactly the abstract methods the engine calls, so an engine's call site is checkable
against a type rather than against a convention. `NullRecorder` is declared once, against the base,
and satisfies each later contract through `ABC.register`: there is one off switch in the library,
not one per shape, and `recorder: AbstractStepRecorder = NullRecorder()` keeps meaning what it says.
A step recorder records two kinds of frame: a **light** frame every step, carrying the step number
and the training loss, and a **full** frame every `frame_every` steps and at the run's end, carrying
V and every number derived from it through the exact `pinv` projector (D-31). Both kinds are the
same dict shape, the light one's derived fields `null`, so a view reads one shape from first frame
to last. The recorder's own `frame_every` default is 1 — a recorder records what it is given — and a
composition root chooses a coarser default when the document it writes would otherwise be large.
**Consequences.** A third engine adds an abstract contract and a child recorder; it never adds a
second null recorder, and nothing in `math` learns what a walkthrough is. The `register` call is the
one place this is weaker under static typing than an inheritance chain would be — a checker sees
`NullRecorder` as satisfying `AbstractStepRecorder` only through the registration — which is
acceptable because the repo has no type-check gate and the runtime relationship is exact. Recording
size becomes a composition-root decision and is documented where it is made: at the example's
`--frame-every` default of 5, where a full frame every step would write a twelve-megabyte document
for karate and the page that embeds it.

## D-34 — Narration is derived from frames, never authored per run; glossary tooltips are CONTEXT.md terms · 2026-09-10 · accepted
**Context.** Phase 1 of BL-43 (D-32, D-33) made a run watchable: a recording of frames and one
offline page with a stepper. It did not make a run legible. A page shows the frontier moving and Ê
falling; it never says what a bound is, which frames decided the run, or why the run ended where it
did. The bar this layer is held to is a reader with the page and nothing else. Three shapes were on
the table. Hand-written prose per page — a paragraph typed once beside each example — goes stale the
moment the run is re-recorded, and nothing catches it, because prose is not compared against
anything. Sentences composed in the browser from the layout keep the prose in step with the numbers,
but they are untestable without a JS runtime, which is exactly what P-5 refuses ("tests assert on
data, never pixels"; no headless browser). Sentences derived in the view's Python at render time,
from a frame and its predecessor, and embedded as data are string values a test can compare for
equality.
**Decision.** The third. A view exposes `explain(recording) -> Explanation`, pure and deterministic:
same recording in, same strings out, no clock, no randomness, no I/O. Every element of the layer —
opening panel, per-frame narration, key moments, legend, quantity strip, ending panel, glossary — is
data in the layout JSON under `explain`, and the stepper only displays it: the JS selects a string
by frame index and marks glossary terms, and **it never composes a sentence**. Tooltips come from
one dictionary, `visualization/glossary.py`, whose keys are `CONTEXT.md` `**Term**:` headings
spelled identically; a term that gets a tooltip must be a heading, and a test parses `CONTEXT.md` to
enforce it, so the page's vocabulary is the repository's vocabulary and cannot fork from it. The
two-hot 2-hot rule is a named constant in the view (`TWO_HOT_SHARE = 0.95`), not a number inside a
sentence. Nothing in this layer adds a recorder call site or a frame field: the one-way flow of
`docs/ARCHITECTURE.md` §1 — "Math never imports ML; ML composes math; scripts (composition roots)
wire data to models" — puts the narration above the engines, and a value a narration wants but no
frame holds is written into the plan's risks as a stated absence, not fetched by widening a frame.
**Consequences.** Re-recording a run re-narrates it, which is the property the hand-written option
could not have. Narration tests are string equality on committed fixtures, so a wording change is a
visible diff and a derivation change that moves a number fails loudly; the fix for a wrong sentence
is always the derivation, never the sentence. The template gains fixed slots — opening, moments,
quantities, narration, legend, ending — that every view fills or leaves empty, so a view is a list
of things to supply rather than a page to design, and a layout without `explain` still renders with
those slots hidden. The glossary grows only through `CONTEXT.md`: a page cannot introduce a word the
domain language has not accepted, and adding a tooltip is a domain-modelling act with a test behind
it. The plan is `docs/plans/2026-09-visualization.md` § 6 (P-7, P-8), the initiative BL-43.

## D-35 — A class is earned by injection, plurality or state; knobs live on the objects they parameterise; the optimizer stack is built like the search stack · 2026-09-11 · accepted
**Context.** The two-hot span optimizer (`math/algorithms/two_hot_span_optimizer.py`, D-31) is one
422-line module with no class hierarchy: a frozen config of thirteen scalars, free functions for the
training loss and its penalties, torch's Adam and its schedule constructed inside the function, a
result that carries per-step arrays, and two stop conditions that raise. Next to it the search stack
is four injected objects — problem, cost function, evaluator, recorder — plus a result contract, and
adding a variant touches none of them (D-23, D-24, D-28, D-32). The backlog already holds the two
extensions that will test the optimizer's shape, BL-46 (restarts, joint factorization) and BL-41
(ncut), and neither slots in without editing the loop. The repository's stated design stance was one
sentence in `CLAUDE.md` ("models are composed from injected math objects") and one in
`docs/ARCHITECTURE.md` §1 ("the idea→class mapping is deliberately literal"), and the code the
second sentence describes does not follow it: the older math layer declares abstractness by `raise
NotImplementedError` on plain classes, `CostFunction` and `RegularizationFunction` have no
implementation, and the first model subclass constructs its own hypothesis and loss instead of
receiving them. `engineering-standards` has no design guidance at all, only process and tooling, and
its rules are meant to hold for a front-end or a CRUD service, which this philosophy would not.
Grilled with Xavier 2026-09-11; the abstract statement is `docs/philosophy/mllib-object-model.md`.
**Decision.** Ten rules, for `src/mllib` only. (1) **A class is earned**, not granted by naming: a
concept becomes a class when it is injected, when it has two or more implementations, or when it
carries state across calls. Otherwise it is a function named after the concept. This supersedes
"deliberately literal" in `docs/ARCHITECTURE.md` §1. (2) **An interface is an `abc.ABC`** with
`@abstractmethod`, named `Abstract*`; `typing.Protocol` is reserved for the tradePlatform seam
(BL-19), where the other side does not subclass us. (3) **Behaviour is injected, numbers are
knobs.** Anything with behaviour — a cost, a projector, a penalty, a step rule, a recorder, a
problem — is a math object passed to the constructor. A number or a name is a knob owned by the
object it parameterises, with its default on the concrete class, so injecting a version is
instantiating it and passing it, the way `MSE()` is passed today. Adam and its schedule are step
rules, injected, never built inside a loop. (4) **Configuration is assembled, not declared once.**
An algorithm's `configuration` (D-28) is its own frozen dataclass plus each injected object's
parameters as `describe()` reads them, keyed by role. The ridge epsilon appears in that record; the
reported numbers still never depend on it (D-31). (5) **A grid is over factories.** A cell is a
configuration plus a constructor per injected role; the composition root builds fresh objects per
cell and never mutates an instance between cells. (6) **One class per concept per arithmetic, never
per array library.** The ridge and the exact projector are two implementations of one concept
(D-31); a measure written identically in numpy and torch is a duplicate to remove. Abstract bases in
`math` never import torch; concrete torch implementations live together in one package. (7) **The
comment ladder.** `CONTEXT.md` says what a term is; `docs/DECISIONS.md` says why a rule holds; a
module docstring says what the module holds and which decisions bind it, as one clause plus the id;
a class docstring is the concept and its contract, and is the description `describe()` reads; a
method docstring states pre- and postconditions; an inline comment carries a local why that no level
above explains. `# TODO(BL-nn)` is unchanged. (8) **Harnesses are composition roots.** They may name
concrete classes; they consume an algorithm through its run and result contract like any caller;
every row they record carries `configuration`. (9) **An optimizer's result follows D-28 and D-32**:
the parameters it moved, the final training loss, the steps taken, a stop reason, its configuration
and the numbers it delivered. Per-step frames stay on the recorder. A stop condition returns a
reason on the result; it does not raise. (10) **Intent first.** The cognitive-load handbook and
*Clean Code* are cited as frameworks, not obeyed as authorities; where either conflicts with showing
what a function or class is for, intent wins, and the philosophy document names the claims it signs
and the ones it rejects. **Consequences.** The optimizer stack takes the search stack's shape: an
`AbstractOptimizer` in `math/algorithms` owning `run()` with a `_step` seam, mirroring
`AbstractGraphAlgorithm`; a problem object holding the data and the exact reporting arithmetic;
`CostFunction` redefined as a scalar of the parameters being optimized, with the ridge cost as its
first implementation; `RegularizationFunction` made abstract with the collision, adjacency and
diversity terms as its first implementations; step rules and schedules as objects; the two-hot
recorder, view and stress harness reading per-step frames from the recorder, with their fixtures
regenerated deliberately in that slice. The acceptance test is BL-46 and BL-41: each must slot in
without editing an existing class. The descent stack is out of scope and its breaks are recorded
(BL-49, BL-26); the latent `hypothesis.py` signature bug is a fix slice with a test, not part of the
refactor. `CONTEXT.md` gains the terms math object, knob, configuration, composition root,
optimizer, step rule and training loss. `fit_two_hot_span` is removed in slice 4 — every caller
becomes a composition root, and a bare function would be a second entry point with a second
configuration story — so its five test modules, the two harnesses and the three examples construct
the optimizer themselves. `docs/ARCHITECTURE.md` §1, §2, §4, §5 and §6 are rewritten to this
decision in the same PR; `CLAUDE.md` gains one read-order line pointing at the philosophy document
for use when designing a component. The plan is `docs/plans/2026-09-optimizer-object-model.md`, the
initiative BL-48.
