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
