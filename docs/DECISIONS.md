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
