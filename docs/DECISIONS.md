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
