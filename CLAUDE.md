<!-- structure from engineering-standards @ 869af91 -->
# botMaker / MlLib — Agent Guide

From-scratch ML library (single owner: Xavier). Three purposes: learn masters-program techniques by
implementing them by hand, showcase professional practice, and serve as a reference brain. Models are
composed from injected math objects (hypothesis, expander, loss); nothing is imported from sklearn's
model classes — sklearn is used only for `train_test_split` and `ParameterGrid`.

## Read before working (in order, only what the task needs)

1. `docs/DECISIONS.md` — settled decisions; don't re-litigate, append new ones
2. `docs/REFACTOR_PLAN.md` — the active migration plan; work happens in its numbered slices
3. `docs/BACKLOG.md` — every `TODO(BL-nn)` in code resolves here
4. `docs/ARCHITECTURE.md` — decomposition, contracts, extension points (dated reviews in `docs/reviews/`)

## Hard rules

- `main` is always green. Changes arrive by PR from a `<type>/<slug>` branch (`CONTRIBUTING.md`).
  Xavier commits; Claude prepares slices and never commits.
- No unstarted code on `main`: intent goes to `docs/BACKLOG.md`, not empty modules or `pass` bodies.
- Every feature or fix ships with a deterministic test (`.claude/agents/testing-agent.md`).
- The training baseline (`MlLib/mlDomain/tests/test_training_baseline.py`) must pass before and after
  every code slice. Regenerate its snapshot only deliberately (`BASELINE_UPDATE=1`) and say so in the PR.
- `# TODO(BL-nn): …` is the only accepted TODO form.
- Datasets over 1 MB are fetched, not committed (D-19).

## Commands

```
uv sync                      # Python 3.12 (.python-version), locked deps, dev group
uv run pytest                # unit suite incl. the training baseline (~40 s)
uv run pytest MlLib/mlDomain/tests/test_training_baseline.py   # baseline only (~10 s)
```

`pandas<3` is pinned in `pyproject.toml` until BL-21 is fixed. One test file is excluded until slice 3.1
fixes its import root: add `--ignore=MlLib/mathDomain/algorithmImplementations/tests/test_breadthFirstSearch.py`.
Lint/format: `uv run ruff check . && uv run ruff format --check .` — also run by pre-commit (`uv run pre-commit install` once per clone).

## Conventions

- Current code is `camelCase`; slice 4.2/4.3 moves to PEP 8 `snake_case`. Don't mix styles within a file;
  new files follow PEP 8 now.
- Commits: `<type>(<scope>): <imperative summary>`, scope ∈ `math ml data docs ci chore`.
- Dates ISO 8601. Line length 100.
- A decision with lasting consequences → `docs/DECISIONS.md`; a technique implemented by hand →
  `docs/LEARNING_LOG.md`; both in the same PR as the change.

## Agents (`.claude/agents/`)

- `testing-agent` — the testing standard; spawn for test work.
- `reviewer-agent` — reviews a slice's diff against the plan and backlog before Xavier commits.
