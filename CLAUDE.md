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

## Commands (until slice 2.1 replaces them with `uv sync`)

```
# unit suite (one file excluded until slice 3.1 fixes its import root)
PYTHONPATH=. uv run --no-project --python 3.12 --with "numpy<2.1" --with "pandas<3" \
  --with networkx --with scikit-learn --with hypothesis --with pytest \
  python -m pytest MlLib -q --ignore=MlLib/mathDomain/algorithmImplementations/tests/test_breadthFirstSearch.py
# training baseline
PYTHONPATH=. uv run --no-project --python 3.12 --with "numpy<2.1" --with "pandas<3" \
  --with networkx --with scikit-learn --with pytest python -m pytest MlLib/mlDomain/tests/test_training_baseline.py -q
```

`pandas<3` is required until BL-21 is fixed. `uv run` without `--no-project` picks up `MlLib/pyproject.toml`
and creates a stray `MlLib/.venv` — don't.

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
