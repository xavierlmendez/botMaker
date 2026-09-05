<!-- structure from engineering-standards @ 869af91 -->
# botMaker / MlLib — Agent Guide

From-scratch ML library (single owner: Xavier). Three purposes: learn masters-program techniques by
implementing them by hand, showcase professional practice, and serve as a reference brain. Models are
composed from injected math objects (hypothesis, expander, loss); nothing is imported from sklearn's
model classes — sklearn is used only for `train_test_split` and `ParameterGrid`.

## Read before working (in order, only what the task needs)

0. `CONTEXT.md` — the glossary; use its canonical terms in names, docs and PRs
1. `docs/DECISIONS.md` — settled decisions; don't re-litigate, append new ones
2. `docs/BACKLOG.md` — open initiatives; new work starts from an entry here (or adds one)
3. `docs/ARCHITECTURE.md` — decomposition, contracts, extension points (dated reviews in `docs/reviews/`)

## Hard rules

- `main` is always green. Changes arrive by PR from a `<type>/<slug>` branch (`CONTRIBUTING.md`).
  Xavier commits; Claude prepares slices and never commits.
- No unstarted code on `main`: intent goes to `docs/BACKLOG.md`, not empty modules or `pass` bodies.
- Every feature or fix ships with a deterministic test (`.claude/agents/testing-agent.md`).
- The training baseline (`tests/ml/test_training_baseline.py`) must pass before and after
  every code slice. Regenerate its snapshot only deliberately (`BASELINE_UPDATE=1`) and say so in the PR.
- `# TODO(BL-nn): …` is the only accepted TODO form.
- Datasets over 1 MB are fetched, not committed (D-19).

## Commands

```
uv sync                      # Python 3.12 (.python-version), locked deps, dev group
uv run pytest                # unit suite incl. the training baseline (~40 s)
uv run pytest tests/ml/test_training_baseline.py   # baseline only (~10 s)
```

Lint/format: `uv run ruff check . && uv run ruff format --check .` — also run by pre-commit (`uv run pre-commit install` once per clone).

## Conventions

- PEP 8 names throughout (`snake_case` modules/functions/attributes, `CapWords` classes); ruff's `N` rules
  enforce it. `X`, `X_*`, `Q` are the only exemptions (design matrix, quadratic form).
- Commits: `<type>(<scope>): <imperative summary>`, scope ∈ `math ml data docs ci chore`.
- Dates ISO 8601. Line length 100.
- A decision with lasting consequences → `docs/DECISIONS.md`; a technique implemented by hand →
  `docs/LEARNING_LOG.md`; both in the same PR as the change.

## Agents (`.claude/agents/`)

- `testing-agent` — the testing standard; spawn for test work.
- `reviewer-agent` — reviews a slice's diff against the plan and backlog before Xavier commits.
