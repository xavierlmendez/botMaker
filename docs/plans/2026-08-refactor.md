# BotMaker / MlLib — Refactor & Framework Plan

Status: **complete** · Written 2026-08-28 · Executed 2026-08-28 → 2026-08-29 (PRs #1–#21) · Owner: Xavier
Archived from `docs/REFACTOR_PLAN.md` in slice 7.1. See §7 for what changed against the plan.
Supersedes nothing; builds on `docs/reviews/2026-07-18-architecture-review.md` (2026-07-18).

---

## 0. Why this document exists

MlLib is a from-scratch ML library that serves three purposes at once:

1. **A learning instrument** — the masters-program techniques (CS 6344 first) get implemented by hand here so they are understood, not just used.
2. **A professional showcase** — the repository is a portfolio artifact and must read like one on `main` at all times.
3. **A reference brain for ML work** — future-Xavier should be able to open this repo and re-derive any topic already worked through.

The code's decomposition (hypothesis / loss / cost injection, template-method graph algorithms, composable models) is sound. What is missing is a **framework around the code**: rules that keep `main` clean, a way to hold future intent without leaving it half-built in the tree, and gates that enforce both without willpower.

This plan does three things, in order:

- **Standardize** what exists (docs, tooling, layout, naming).
- **Capture** the future initiatives currently sprinkled through the code as stubs, empty files, and `TODO`s — and move them out of the tree into a tracked backlog.
- **Sequence** the real refactors (R1–R3 from the architecture review) as reviewable slices grouped into phases.

### Decisions already made (owner, 2026-08-27)

| # | Decision |
|---|---|
| D-1 | The "framework" is both the *engineering* framework (process, gates, docs) and the *code* framework (base-class contracts). Engineering framework ships first because it gates everything else. |
| D-2 | Audience: Xavier, with a professional lens. tradePlatform plugin-source remains the north star, not a current dependency. |
| D-3 | `fastapi_app/` is **deleted** now and recorded as a future initiative. |
| D-4 | Strip rule: a cleanly removable stub is deleted. Anything whose clean removal would take **> 1 hour** or an intensive session to re-add is **kept and marked TODO**. `TransformerPipeline` is a keep. Never-imported modules are kept if tested; some untested ones are also kept (listed explicitly in §3). |
| D-5 | Deleted initiatives are recorded (this repo: `docs/BACKLOG.md`). |
| D-6 | Branching: feature branches + PR into `main`, CI required, `main` always green. |
| D-7 | Project-level TODO tracking lives in the **orchestrator** repo under `projects/botmaker.md`. |
| D-8 | Xavier commits every change, in reviewable slices grouped into phases. |
| D-9 | AWS CodeBuild deploy is dormant; GitHub Actions replaces it as the CI of record. |
| D-10 | Known bugs F4 and F5 are fixed in this migration. |
| D-11 | R1–R3 are in scope as scheduled phases. |
| D-12 | Delete the accidental outer repo at `~/Desktop/BotMaker/.git` and the stale `~/RiderProjects/botMaker` checkout. |
| D-13 | Deliverable is this file, committed in-repo, plus a shareable page. |

### Assumptions this plan makes (confirm or overrule)

| # | Assumption | Why |
|---|---|---|
| A-1 | **Where cross-repo standards live:** a new dedicated repo, `xavierlmendez/engineering-standards`, holding the reusable `CLAUDE.md` fragments, `CONTRIBUTING` template, tool configs, and skills. Repos consume it by copying the relevant fragments (with a header noting the source) — not by submodule. The orchestrator stays the *state* system (projects, captures, decisions); it is not a standards library, and its own `CLAUDE.md` forbids adding to it without `/reweave`. | Keeps orchestrator's invariants intact; standards are read by Claude at repo-open time and need to physically exist in each repo. |
| A-2 | **Split of TODO tracking:** phase-level milestones and the `next` pointer live in orchestrator's `projects/botmaker.md` (`## Schedule` line-items are sanctioned hand-edits per orchestrator D-002). The per-file registry of deleted initiatives and code-level TODOs lives in this repo's `docs/BACKLOG.md`, because it references file paths and line numbers that only make sense next to the code. | Orchestrator's schema has no backlog table; forcing 25 items into it would fight the tool. |
| A-3 | **Python 3.12** as the target (pyproject says `>=3.10`; the checked-in `.venv` is 3.9 and cannot run the tests; machine has 3.14 and `uv`). | Modern enough for `X \| Y` typing and `match`, stable across NumPy/networkx. |
| A-4 | **Module naming moves to PEP 8 `snake_case`** (currently `camelCase` files like `hypothesisExpander.py`, and `PascalCase` like `DataOrchestrator.py`). Done in one mechanical slice with `git mv`. | Ruff's `N` rules will flag it forever otherwise; portfolio readers expect PEP 8. |
| A-5 | **Layout moves to a `src/` layout** with one top-level `tests/` tree mirroring the package. | The current mixed import roots (`MlLib.…` vs `botMaker.MlLib.…`) are the direct cause of the collection error; a `src/` layout with an installed package removes the ambiguity permanently. |
| A-6 | Large CSVs (`ad_click_dataset.csv` 465 KB, `shopping_behavior_updated.csv` 417 KB) stay in git for now under `data/`; a `DECISIONS.md` entry records the size ceiling (1 MB per file) above which datasets must be fetched, not committed. | Small enough to keep; the rule prevents the next one from being 50 MB. |
| A-7 | The `.docx` / `.pdf` report and the `.ipynb` homework are moved to `docs/reports/` and `notebooks/` rather than deleted. | They are part of the "reference brain" purpose. |

---

## 1. Current state (evidence)

Captured 2026-08-28 from `~/Desktop/BotMaker/botMaker` (`main` @ `e29cd03`).

**Repository shape**

- Two nested git repos: the real one at `~/Desktop/BotMaker/botMaker` (remote `xavierlmendez/botMaker`) and an accidental, remote-less one at `~/Desktop/BotMaker/` with a single commit that treats the project as untracked. `~/RiderProjects/botMaker` is the .NET-era ancestor (last commit 2025-10-03).
- All Python. C# removed wholesale in `f6a50c3`.
- `__pycache__/`, `.DS_Store`, `.venv/` are not ignored at the repo root (`.idea/` is ignored and untracked — corrected 2026-08-28).
- No `README.md` (pyproject references one that does not exist), no `CLAUDE.md`, no `CONTRIBUTING.md`, no lint/format config, no pre-commit, no GitHub CI. `buildspec.yaml` + `Dockerfile` target a dormant AWS CodeBuild → ECR pipeline.

**Test baseline**

```
uv run --python 3.12 --with pytest,numpy,networkx,pandas,hypothesis \
  python -m pytest MlLib -q
→ 1 collection error (test_breadthFirstSearch.py: `No module named 'botMaker'`)
→ ignoring that file: 27 passed, 1 skipped
```

The error is an import-root inconsistency: three tests under `algorithmImplementations/tests` import `botMaker.MlLib.…`; all others import `MlLib.…`. `pyproject.toml` sets `pythonpath = [".."]`, which only satisfies the second form.

**Behavioural baseline (added 2026-08-28)**

`MlLib/mlDomain/tests/test_training_baseline.py` runs the real ad-click pipeline (CSV → `DataOrchestrator` → project `LogisticRegression` classes → `gridFit` → evaluator) with an 8-permutation smoke grid and a seeded split, in ≈ 7 s. Snapshot in `baseline_snapshot.json`; deterministic across runs. Regenerate only on purpose with `BASELINE_UPDATE=1`. Requires `pandas<3` until BL-21 is fixed. The full `AdClickModelProjectBuildScript.py` grid (15 840 permutations) is ≈ 5 h and is **not** the baseline.

```
PYTHONPATH=. uv run --no-project --python 3.12 --with "numpy<2.1" --with "pandas<3" \
  --with networkx --with scikit-learn --with pytest \
  python -m pytest MlLib/mlDomain/tests/test_training_baseline.py -q
→ 7 passed
```

**Known bugs (from the architecture review, verified still present)**

- **F4** — `projectSpecificFiles/adClickPredictionLogReg.py` `LogisticRegression.__init__` skips `super().__init__()`; `.fit()` breaks unless `gridFit` ran first.
- **F5** — `modelEvaluators/genericEvaluator.py:44` builds a **set literal** where a dict is intended in the base `persistEvaluationRecord`; masked because both subclasses override it.

**Core abstractions worth protecting** (the part that works)

`HypothesisFunction` · `HypothesisExpander` (Φ) · `LossFunction` / `CostFunction` · `abstractGraphAlgorithm` template method with frozen `SearchContext` · `graphStructures` / `treeStructures` / `splitFunction` · models composed from injected math objects · `ModelEvaluator` · `DataOrchestrator`.

---

## 2. The framework

### 2.1 Files that will exist after Phase 1

| File | Purpose | Reusable across repos? |
|---|---|---|
| `CLAUDE.md` | Agent guide: read-order, hard rules, run/test commands, conventions. Short; procedures move to skills/agents. Modeled on the tradePlatform `CLAUDE.md` house style. | Structure yes (template in standards repo); content no |
| `README.md` | What MlLib is, layout, install, run tests, how models compose. Portfolio front door. | No |
| `CONTRIBUTING.md` | The slice rules, branch/PR workflow, definition of done, commit format. | **Yes** — verbatim from standards repo |
| `docs/DECISIONS.md` | Append-only ADR log. First entries: D-1…D-13 above plus A-1…A-7 once confirmed. | Template yes |
| `docs/BACKLOG.md` | Registry of stripped initiatives and code-level TODOs, each with the *why*, what existed, and the re-entry cost. | Template yes |
| `docs/ARCHITECTURE.md` | The architecture review promoted to a living document: the decomposition, the contracts, the extension points. | No |
| `docs/REFACTOR_PLAN.md` | This file. Archived to `docs/plans/` when complete. | No |
| `docs/LEARNING_LOG.md` | The "reference brain": one entry per topic implemented by hand — the math, the design choice, what was confusing, links to the module and its test. | Template yes |
| `pyproject.toml` (root) | Single project file: metadata, deps, `[tool.ruff]`, `[tool.pytest]`, `[tool.mypy]` (deferred), `uv` lock. | Config blocks yes |
| `.pre-commit-config.yaml` | ruff (lint + format), end-of-file, trailing whitespace, large-file guard, `uv lock --check`. | **Yes** |
| `.github/workflows/ci.yml` | `uv sync` → `ruff check` → `ruff format --check` → `pytest` on 3.12. Required status check on `main`. | **Yes** |
| `.editorconfig` | Indent/EOL/charset. | **Yes** |
| `.claude/agents/testing-agent.md` | The testing standard (deterministic, no wall clock, no unseeded randomness; every fix ships with a test). Ported from tradePlatform and generalized. | **Yes** |

### 2.2 The `engineering-standards` repo (A-1)

```
engineering-standards/
  README.md                       what this is, how a repo adopts it
  CLAUDE.template.md              skeleton with the sections every repo CLAUDE.md has
  CONTRIBUTING.md                 the slice/PR/DoD rules, copied verbatim into repos
  templates/
    DECISIONS.md  BACKLOG.md  LEARNING_LOG.md  PLAN.md
  configs/
    python/  pyproject.ruff.toml  .pre-commit-config.yaml  ci.yml  .editorconfig
    (later) typescript/ …
  agents/
    testing-agent.md  reviewer-agent.md
  sources/
    SOURCES.md                    the published vendor docs each rule is derived from,
                                  with the date consulted (ruff, uv, pytest, GitHub
                                  Actions, pre-commit, Anthropic's CLAUDE.md guidance)
```

Rule for the standards repo: **every rule cites its source.** Where a convention is Xavier's own choice rather than vendor guidance, `SOURCES.md` says so. This is what makes the standards defensible in an interview and keeps them honest when tools change.

### 2.3 Contribution rules (the part that keeps `main` clean after the migration)

These go into `CONTRIBUTING.md` and are enforced by CI + pre-commit, not memory.

1. **`main` is always green and always shippable.** No direct commits. Every change arrives by PR from a short-lived branch (`<type>/<slug>`: `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`), squash-merged.
2. **One concern per slice.** A PR is reviewable if it is ≤ ~300 changed lines of non-generated code, has one purpose stated in its title, and CI passes. Larger changes are split *before* opening.
3. **No unstarted code on `main`.** Empty modules, `pass`-bodied placeholders, and commented-out designs do not merge. Intent goes to `docs/BACKLOG.md` (with the design sketch if there is one) or to a branch that is deleted or merged within 30 days.
4. **`TODO` has a format or it does not exist.** `# TODO(BL-nn): …` referencing a `BACKLOG.md` entry. Ruff's `TD` rules enforce the shape; a CI grep enforces the reference.
5. **Every feature or fix ships with a test.** Deterministic only. A bug fix adds the test that would have caught it *first*.
6. **Every decision with lasting consequences gets a `DECISIONS.md` entry** in the same PR.
7. **Every hand-implemented technique gets a `LEARNING_LOG.md` entry** in the same PR — that is the point of the project.
8. **Commit messages:** `<scope>: <imperative summary>` where scope is a package (`math`, `ml`, `data`, `docs`, `ci`, `chore`). Body explains *why* when non-obvious.
9. **Definition of done** for a PR: CI green · tests added · docs/decision/learning entries updated as applicable · no new `TODO` without a backlog id · branch deleted after merge.

---

## 3. Initiative triage

Rule applied: D-4. "Delete" = removed in Phase 3 and recorded in `BACKLOG.md`. "Keep+TODO" = stays, gets a `BL-nn` id. Re-entry cost is the estimate to rebuild from the backlog entry.

| ID | Item | Location | Verdict | Rationale | Re-entry |
|---|---|---|---|---|---|
| BL-01 | FastAPI contract mock | `fastapi_app/`, `Dockerfile`, `buildspec.yaml`, `initDummyDataForStrategy.json` | **Delete** | Owner decision D-3. No consumers; the Pydantic contract shape is worth saving in the backlog entry. | 2–3 h |
| BL-02 | Neural network | `mlDomain/neuralNetwork.py` (0 lines) | **Delete** | Empty file. | 0 (nothing to restore) |
| BL-03 | CSV source controller | `dataDomain/sourceControllers/CsvController.py` (0 lines) | **Delete** | Empty file. | 0 |
| BL-04 | Iterative deepening search | `algorithmImplementations/iterativeDeepening.py` (0 lines) | **Delete** | Empty; BFS/DFS on `abstractGraphAlgorithm` show the pattern. | 30 min |
| BL-05 | `algorithmImplementations/testScript.py` | (0 lines) | **Delete** | Empty scratch file. | 0 |
| BL-06 | AI domain | `aiDomain/` (empty `__init__.py`) | **Delete** | Declared-but-empty package. Intent recorded: the future join point for agent/LLM-driven model selection. | 0 |
| BL-07 | Sum rule / product rule | `probabilityBased/sumRule.py`, `productRule.py` (9 lines each, `pass`) | **Keep+TODO** | *Tested* (`test_probabilityBased.py` imports both) — tests currently assert only the placeholder exists. Under D-4 tested modules stay. Implement in the CS 6344 pairing. | — |
| BL-08 | InformationGain / ChiSquare split criteria | `graphBased/splitFunction.py` | **Keep+TODO** | Placeholder subclasses beside a real `Gini`; removing them means editing the class hierarchy and `DecisionTree`'s injection points (> 1 h clean). | — |
| BL-09 | Declarative `TransformerPipeline` | `dataDomain/DataOrchestrator.py:41,53` (commented out) | **Keep+TODO → R1** | Owner decision. This is Phase 6. | — |
| BL-10 | Regression/classification enum for cost/loss/regularization | `costFunction.py:7`, `lossFunction.py:9`, `regularizationFunction.py:4` | **Keep+TODO → R3** | Folded into the metadata/introspection phase. | — |
| BL-11 | Exporter for trained models | `projectSpecificFiles/adClickPredictionLogReg.py:29,52` | **Keep+TODO** | Real need (persist fitted weights); design not started. | 1–2 h |
| BL-12 | `hypothesis.py:43` — call expander to reshape data | `mathDomain/hypothesis.py` | **Keep+TODO → R2** | Belongs with the shared gradient-descent base. | — |
| BL-13 | `decisionTree.py` refactor onto graphBased utilities; missing no-split error; "abstracted later" | `mlDomain/decisionTree.py:23,115,133` | **Keep+TODO** | > 1 h; the no-split error handling becomes a Phase 3 bug fix with a test. | — |
| BL-14 | Never-imported modules: `visualizer.py`, `bayesRule.py`, `gaussianPrior.py`, `depthFirstSearch.py`, `regularizationFunction.py`, `linearAlgebraHelpers.py` | various | **Keep, add smoke tests** | Owner: some untested modules stay. All are real implementations (8–34 lines). Phase 3 adds one smoke test each so the keep rule is satisfied going forward. | — |
| BL-15 | Duplicate DFS | `graphBased/searchAlgorithms/DFS.py` (13 lines) vs `algorithmImplementations/depthFirstSearch.py` (33 lines, on the ABC) | **Delete `searchAlgorithms/DFS.py`** | Superseded by the template-method version. | 0 |
| BL-16 | `~25 × # TODO: review metadata (auto-generated)` | across the tree | **Delete markers → R3** | Replaced wholesale by introspected metadata in Phase 5. | — |
| BL-17 | `projectScripts/sandbox.py`, `testScript.py`, `KNNClassifierTestScript.py` | `projectScripts/` | **Move to `examples/`**, delete `sandbox.py` | Scripts are composition roots and belong in the reference brain; `sandbox.py` is scratch. | 0 |
| BL-18 | `graphBased/tests/nxGraphExample.py` | (66 lines, not a test) | **Move to `examples/`** | Not collected by pytest; it is a worked example. | 0 |
| BL-19 | tradePlatform plugin seam | (not in code) | **Backlog only** | Recorded so the north star is written down: descriptors introspected at the boundary (see `docs/reviews/2026-07-18-architecture-review.md` §Cross-codebase). | — |
| BL-20 | Logistic/linear duplication | `logisticRegression.py:10,36` | **Keep+TODO → R2** | Phase 5. | — |
| BL-21 | pandas ≥ 3 incompatibility | `genericEvaluator.py:136-137` | **Fix in Phase 3** | Copy-on-write makes `.to_numpy()` views read-only; in-place `[... == -1] = 0` raises `ValueError: assignment destination is read-only`. Baseline runs pinned to `pandas<3` until fixed. | — |
| BL-22 | Confusion-matrix FP/FN swapped | `genericEvaluator.py:141,145` | **Fix in Phase 3** | "falsePositives" counts target=1/pred=0 (a false negative) and vice versa, so reported precision is really recall. Snapshot must be regenerated deliberately when fixed. | — |
| BL-23 | Training predicts all-ones on ad-click data | `logisticRegression.gridFit` / `hypothesis.computeClassification` | **Investigate in Phase 5** | Every permutation of the smoke grid (and the historic example output) yields TP=1317, TN=0: accuracy equals the majority-class rate. Likely threshold/feature-scaling issue. Not a migration blocker; the snapshot pins the behaviour so the refactor can't hide it. | — |

---

## 4. Phases and slices

Conventions for this section: every slice is one PR on a branch named as shown, lands squash-merged, and must leave `main` green. Estimates are for Xavier doing the work with Claude assisting. **Order within a phase is the order to do them; phases are strictly sequential.**

### Phase 0 — Repository hygiene (≈ 1.5 h) · *no code changes*

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 0.1 | *(local only)* | Remove accidental outer repo: `rm -rf ~/Desktop/BotMaker/.git ~/Desktop/BotMaker/.idea`. Delete `~/RiderProjects/botMaker`. Optionally move the project to `~/develop/botMaker` next to `orchestrator`. | `git -C ~/Desktop/BotMaker rev-parse` fails; only one checkout exists |
| 0.2 | `chore/gitignore` | Root `.gitignore` from the standards Python template (`__pycache__/`, `.venv/`, `.DS_Store`, `.idea/`, `*.egg-info`, `.pytest_cache/`, `.ruff_cache/`). Nothing was tracked that needed un-tracking. | `git status` clean after a test run |
| 0.3 | `chore/remove-fastapi` | Delete `fastapi_app/`, `Dockerfile`, `buildspec.yaml`, `initDummyDataForStrategy.json`, root `__init__.py`. Add BL-01 to a first-draft `docs/BACKLOG.md` with the Pydantic models pasted in as the saved contract. | Tree contains only `MlLib/`, `docs/`, config |
| 0.4 | `chore/docs-and-assets` | Move `.docx`/`.pdf` → `docs/reports/`, `MLFall25HW1.ipynb` → `notebooks/`, and archive the architecture review as `docs/reviews/2026-07-18-architecture-review.md` (references updated). `.pptx` stays ignored on disk (A-7 optional). | `git ls-files` shows no binaries outside `docs/reports/` and `data/` |

### Phase 1 — Framework documents (≈ 3 h)

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 1.1 | *(new repo)* `engineering-standards` | Scaffold per §2.2: templates, Python configs, `CONTRIBUTING.md`, `SOURCES.md` with citations to ruff / uv / pytest / pre-commit / GitHub Actions / Anthropic CLAUDE.md docs. | Repo pushed; README explains adoption in ≤ 10 lines |
| 1.2 | `docs/framework` | Add `CLAUDE.md`, `README.md`, `CONTRIBUTING.md` (copied), `docs/DECISIONS.md` (seeded with D-1…D-13, A-1…A-7 as confirmed), `docs/BACKLOG.md` (full §3 table), `docs/LEARNING_LOG.md` (seeded with the topics already implemented: linear/logistic regression, perceptron, SVM, decision trees, KNN, BFS/DFS, polynomial features), `.claude/agents/testing-agent.md`. | Opening the repo in Claude Code and asking "how do I add a model?" gets a correct answer from the docs alone |
| 1.3 | `docs/architecture` | Promote the archived review (`docs/reviews/2026-07-18-architecture-review.md`, moved in 0.4) into a living `docs/ARCHITECTURE.md`: decomposition, contracts, extension points, the R1–R3 targets. | — |
| 1.4 | *(orchestrator)* | `projects/botmaker.md`: update `## Thesis` to mention the framework, add `## Schedule` line-items for Phases 0–7 with `~` estimated dates, set `next` to Phase 0.1. Log the change. Run `python3 build/build.py`; `/publish`. | Docket shows the phase ladder |

### Phase 2 — Tooling gates (≈ 2.5 h)

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 2.1 | `chore/uv-pyproject` | Move `MlLib/pyproject.toml` → root `pyproject.toml`; `requires-python = ">=3.12"`; `uv lock`; `uv sync`; commit `uv.lock`. Delete the 3.9 `.venv`. Dev deps: `pytest`, `pytest-cov`, `ruff`, `pre-commit`, `hypothesis`, `pandas`. | `uv run pytest` reproduces the §1 baseline (27 pass / 1 skip / 1 error) |
| 2.2 | `chore/ruff` | `[tool.ruff]` from standards: `line-length = 100`, select `E,F,W,I,N,UP,B,SIM,TD,RUF`; per-file ignores for `N` on the not-yet-renamed modules (removed in Phase 4). `ruff format`. Commit the format-only diff **separately** from config. | `ruff check` and `ruff format --check` pass |
| 2.3 | `chore/pre-commit` | `.pre-commit-config.yaml`, `.editorconfig`; `pre-commit install`. | A commit with a lint error is rejected locally |
| 2.4 | `ci/github-actions` | `.github/workflows/ci.yml`: `astral-sh/setup-uv` → `uv sync --frozen` → ruff → pytest with coverage summary. Then in GitHub settings: branch protection on `main` (require PR, require `ci` status, no force-push). | The PR for this slice is the first one that shows the check |

### Phase 3 — Green baseline and strip (≈ 3 h)

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 3.0 | *(gate)* | Baseline test green before and after **every** Phase 3–6 slice; it joins the CI job in 2.4. | `test_training_baseline.py` → 7 passed |
| 3.1 | `fix/test-import-root` | Make the three `algorithmImplementations/tests` import `MlLib.…`; remove `pythonpath = [".."]` in favor of `uv sync` installing the package (editable). | `uv run pytest` → 0 errors, 30 collected |
| 3.2 | `fix/evaluator-set-literal` | F5: dict literal in base `persistEvaluationRecord`, plus a test that calls the base implementation. | Test fails before, passes after |
| 3.3 | `fix/logreg-super-init` | F4: `super().__init__()` chain in `adClickPredictionLogReg.LogisticRegression`; test that `.fit()` works without `gridFit`. | Same |
| 3.4 | `fix/decision-tree-no-split` | BL-13 partial: raise a typed error when no split is possible; test. | Same |
| 3.4b | `fix/evaluator-readonly-arrays` | BL-21: `np.array(..., copy=True)` on entry to `updateTestingPredictionData`; drop the `pandas<3` pin. | Baseline green on pandas 3 |
| 3.4c | `fix/confusion-matrix-labels` | BL-22: swap FP/FN; unit test on a 4-row hand-built case; regenerate the baseline snapshot with `BASELINE_UPDATE=1` in the same PR and say so. | Precision/recall match sklearn on the same arrays |
| 3.5 | `chore/strip-unstarted` | Delete BL-02…06, BL-15, `sandbox.py`. Move BL-17/18 to `examples/`. Each deletion's backlog entry is already written (1.2). | Tree has no 0-line modules; `grep -rn "pass$"` shows only intentional placeholders BL-07/08 with `# TODO(BL-07)` tags |
| 3.6 | `test/smoke-untested-modules` | One smoke test per BL-14 module. | Coverage report lists every module ≥ 1 test |
| 3.7 | `chore/todo-format` | Rewrite the surviving `TODO`s to `# TODO(BL-nn): …`; add the CI grep that fails on a `TODO` without an id. | CI green with the grep on |

### Phase 4 — Structure and naming (≈ 3 h) · *mechanical, high-churn, do in one sitting*

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 4.1 | `refactor/src-layout` | `MlLib/` → `src/mllib/`; tests → top-level `tests/` mirroring the package; `data/`, `examples/`, `notebooks/`, `docs/` at root. `ProjectSpecificDataClasses/*.json` → `data/configs/`. Update `pyproject` `[tool.setuptools]`/`[tool.pytest]`. | `uv run pytest` green; `import mllib` works from a fresh `uv sync` |
| 4.2 | `refactor/snake-case-modules` | `git mv` every module to `snake_case` (`hypothesis_expander.py`, `data_orchestrator.py`, `abstract_graph_algorithm.py`, …). Rewrite imports with ruff's `I` + a scripted sed; **no logic changes.** Drop the Phase-2 `N` per-file ignores. | `ruff check --select N` clean; tests green; diff is renames + import lines only |
| 4.3 | `refactor/class-and-method-names` | Class names stay PascalCase (already are); method names to `snake_case` (`computePrediction` → `compute_prediction`, `gridFit` → `grid_fit`, `persistEvaluationRecord` → …). One slice per domain if the diff exceeds 300 lines: `math`, `ml`, `data`. | `N802/N803/N806` clean; tests green |
| 4.4 | `docs/update-after-rename` | `README`, `ARCHITECTURE`, `LEARNING_LOG`, `BACKLOG` paths refreshed. | `grep -rn "MlLib/" docs/` returns nothing stale |

### Phase 5 — R3: introspected metadata + R2: shared gradient-descent base (≈ 5 h)

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 5.1 | `refactor/metadata-introspection` | A single `describe(obj)` helper deriving `{name, kind, doc, signature, params}` from the class; remove every hand-typed `metadata` dict and the 25 auto-generated markers (BL-16). Tests for the helper and for one instance of each kind. | No `metadata = {` literals remain; evaluator records still serialize |
| 5.2 | `refactor/task-kind-enum` | BL-10: `TaskKind = Enum("REGRESSION", "CLASSIFICATION")` on loss/cost/regularization; replace the "later" TODOs. | Enum used by models; TODOs gone |
| 5.3 | `refactor/gradient-descent-base` | R2/BL-20/BL-12: `GradientDescentModel` base parameterized by the hypothesis's predict method; `LinearRegression` and `LogisticRegression` become thin subclasses; `train/test split` moves into `grid_fit`. Property-based test (hypothesis lib) that both converge on a known synthetic dataset. | Line count of the two models drops by > 50 %; tests green |
| 5.3b | `fix/degenerate-classifier` | BL-23: find why every permutation predicts all-ones (threshold, feature scaling, or gradient sign); property test that the model separates a linearly separable synthetic set. Regenerate snapshot deliberately. | Best smoke-grid accuracy > majority rate |
| 5.4 | `docs/learning-log-gd` | `LEARNING_LOG.md` entry: gradient descent as a template method — what generalizes across regression/classification and why. | — |

### Phase 6 — R1: declarative `TransformerPipeline` (≈ 4 h)

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 6.1 | `refactor/transformer-abc` | Make `abstract_transformer.py` a real ABC with `fit/transform/fit_transform`; two concrete transformers extracted from the current `temp*Transformer` methods. Tests. | — |
| 6.2 | `refactor/transformer-pipeline` | Un-comment and finish `TransformerPipeline`: JSON config → list of transformers loaded by class name; `DataOrchestrator` uses it. | `PurchasePredictionDataTransformations.json` drives the ad-click example end to end |
| 6.3 | `refactor/retire-model-ladder` | Delete `temp*Transformer` methods and the `if model == …` ladder. | `DataOrchestrator` has no model-name strings |
| 6.4 | `docs/learning-log-pipelines` | Learning-log entry; close BL-09. | — |

### Phase 7 — Close-out (≈ 1 h)

| Slice | Branch | Change |
|---|---|---|
| 7.1 | `docs/close-refactor` | Move this file to `docs/plans/2026-08-refactor.md` with a "what changed vs. plan" section; `DECISIONS.md` entry recording completion; orchestrator `projects/botmaker.md` log + next pointer to the first CS 6344 topic (PCA). |
| 7.2 | `docs/standards-retro` | Back-port anything learned into `engineering-standards` (configs that needed tweaking, rules that were wrong). |

**Total: ≈ 24 h across 7 phases, 35 slices.** Phases 0–3 (≈ 10 h) get `main` to "professional and green"; 4–6 are the real refactors and can be paced around coursework.

---

## 5. Risks and how the plan handles them

| Risk | Mitigation |
|---|---|
| Rename churn (Phase 4) makes `git blame` noisy | `.git-blame-ignore-revs` listing the rename and format-only commits; GitHub honors it. |
| Ruff format diff hides real changes | 2.2 commits config and format separately; 4.2 forbids logic changes. |
| Phase 5/6 refactors stall mid-way, recreating the "half-built on main" problem | Each is behind a branch; rule 3 in §2.3 (30-day branch limit) applies to Xavier too. The backlog entry keeps the design if the branch is abandoned. |
| Standards repo becomes a second unfinished project | It ships in slice 1.1 with only what Phase 1–2 need. It grows by back-port (7.2), never speculatively. |
| Orchestrator schema pressure from tracking too finely | A-2: only phase-level items go in the orchestrator. |
| Coursework interrupts | Phases 0–3 first, in one or two sittings; everything after is independently pausable. |

---

## 6. Immediate next actions

1. Confirm or overrule A-1…A-7 (reply with the ids).
2. Run Phase 0.1 (local deletes — the plan does not perform these for you).
3. Open the first branch: `chore/gitignore`.

---

## 7. What changed vs. the plan (written at close-out, 2026-08-29)

**Schedule.** Planned ≈24 h over ~5 weeks (~2026-09-30); executed in two sittings on 2026-08-28/29 as 21 PRs.
The per-slice estimates were about right; the calendar estimate assumed evenings, not sessions.

**Slices added.** 3.4b (BL-21 pandas-3 read-only arrays), 3.4c (BL-22 swapped FP/FN), 5.3b (BL-23 degenerate
classifier) — all three found by *building the baseline test* in the pre-work, which is the single most valuable
thing this plan did. 3.0 (the baseline as a gate on every code slice).

**Slices absorbed.** 4.4, 5.4 and 6.4 (docs-after-the-fact) never ran as separate PRs: every rename/refactor slice
updated the docs in the same PR, which is what CONTRIBUTING rule 6/7 demands anyway.

**Findings that changed the record.** The survey overstated BL-14 (only 2 of 6 "never-imported modules" were real
implementations; DFS was a stub → BL-25, implemented in 3.6). `.idea/` was never tracked. The base
`MyLogisticRegression.__init__` had always been broken (missing `degree` argument) — F4 was hiding a `TypeError`.
The ad-click dataset has no linear signal (sklearn logistic 0.650 CV vs 0.650 majority), so the "degenerate
classifier" was half data; the model half (label-space mismatch) was real and is fixed.

**Process corrections.** Two self-inflicted errors were caught by gates rather than review: a 178 MB `.pptx` that
the `src/` move re-exposed (caught by `check-added-large-files`), and an over-broad rename that touched attributes
in a renames-only slice (caught by the imports-only diff check). One process error was mine: syncing `main` on
"continue" before verifying the PR had actually been merged — fixed by checking `gh pr list` first, every time.

**Backlog at close.** 26 entries: 16 closed, 10 open — BL-01 (web seam, build in tradePlatform), BL-07/08/11/13
(learning-paired implementations), BL-14 (record), BL-19 (plugin seam), BL-24 (lint debt, counts in `pyproject`),
BL-26 (perceptron/SVM onto the descent base). None block the CS 6344 work; BL-07 is its natural first pairing.

**Numbers.** Tests 27 → 87 (+ baseline); `DataOrchestrator` 378 → 87 lines; `MyLinearRegression` 76 → 11;
46 metadata dicts → 46 docstrings + `describe()`; 415 naming violations → 0; ruff/pre-commit/CI gates on;
`main` protected.
