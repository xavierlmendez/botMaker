# Interactive walkthroughs from injected recorders

Status: **accepted** (2026-09-10, grilled with Xavier) · Written 2026-09-10 · Owner: Xavier
Target: four stacked slices from `feat/visualization-recorder` into `main` (`main` @ `1cc6fb1`);
slice 3 stacks on `feat/two-hot-span-harness` (`docs/plans/2026-09-two-hot-span.md` slice 2.3).
Phase 2 (§ 6) adds three more stacked branches on top of those four:
`feat/visualization-explain` → `feat/visualization-explain-two-hot` → `feat/visualization-explain-traversal`.
Backlog entry: **BL-43**. Decisions: **D-32**, **D-33**, **D-34**.

---

## 0. Why

The library certifies landmark selections and optimizes spanning sets, and neither run can be watched.
A `SearchResult` says what was proved; it says nothing about how the frontier moved, which children were
pruned, or when the incumbent last improved — and the two-hot optimizer's V is only ever seen at the last
step. Both are teaching artefacts as much as engines (CLAUDE.md: learn by hand, showcase, reference brain),
and a run a reader can step through is the form that carries all three.

What exists today is `math/graph/visualizer.py`, a matplotlib animation nothing imports, and the evaluator
record, which is one row per run. Neither is per-step observation. This plan adds observation as a
*collaborator injected into the algorithm* — a recorder — and a renderer that turns a run's frames into one
offline HTML page. The engines keep their shape: guarded call sites only, no new result field (D-28), and a
run with the null recorder identical to today's, to the byte.

## 1. Decisions and assumptions

| # | Decision |
|---|---|
| P-1 | Observation is an **injected recorder**, off by default. An algorithm takes `recorder: AbstractRecorder` defaulting to a fresh `NullRecorder`; when it is off the hot loop pays exactly one `enabled` read per step and nothing else. No global switch, no registry, no result field (**D-32**, D-28). |
| P-2 | The abstract recorder (`AbstractRecorder`, `Frame`, `NullRecorder`) lives in `math/recorder.py`; the per-problem children live in `visualization/recorders/`. **The child extracts and captions**; the engine calls it with the raw objects it already holds (the expanded state and its bound, the priced children, the full frontier, the expansion count, plus variant extras through a `_recorder_extras()` hook the base calls). math never imports visualization. |
| P-3 | A **recording** is a versioned JSON document (`schema_version`, `problem`, `configuration`, `frames`, `result`) written when the run ends, with exact floats, nested lists, and **no timestamps anywhere** — so a recording is diffable and a fixture can be committed like a baseline snapshot. |
| P-4 | A **walkthrough** is a single offline HTML file: vanilla JS, inline SVG, no CDN, no build step. One generic template owns the stepper (previous, next, slider, play with speed, keyboard arrows), the caption and a raw-fields panel; each problem supplies a **view** — a JS drawing function plus the layout data Python computed for it. |
| P-5 | Tests assert on **data, never pixels**: frame counts and their agreement with the result, round-tripped recordings, the presence of the recording and the stepper in the rendered page. No image comparison, no headless browser. |
| P-6 | A frame records the **full frontier**, not a truncation. Walkthrough inputs are small by design (fixture-scale runs, tens to hundreds of expansions); the cells that make the frontier large are grid runs, which are not walkthroughs. |
| P-7 | Every sentence on a page is **derived from the recording by the view's Python at render time** and embedded as data under `layout["explain"]`. No prose is authored per run, and the JS never composes a sentence — it displays the string the frame's index selects (**D-34**). |
| P-8 | Hover tooltips come from one dictionary (`visualization/glossary.py`) whose keys are `CONTEXT.md` `**Term**:` headings, spelled identically; a test parses `CONTEXT.md` and fails on any key that is not a heading (**D-34**). |

| # | Assumption | Why |
|---|---|---|
| A-1 | The null-recorder guard keeps both baselines byte-identical. | `tests/ml/test_training_baseline.py` and `tests/ml/test_nystrom_search_baseline.py` are the only guarantee that instrumentation changed no number; the guard is one boolean read on a branch that already exists. Verified per slice, not once — every slice that touches an engine re-runs both. |

## 2. Current state (evidence)

Captured 2026-09-10 at `1cc6fb1`.

- **No observation seam exists.** No observer, callback, listener or hook anywhere under `src/mllib`
  (`git grep -Ei 'observer|callback|on_step|hook' 1cc6fb1 -- src/mllib` is empty).
- The only opt-in instrumentation in the tree is the pruned A\* **bound-drop counter**
  (`count_bound_drops`, `bound_drop_slack`, `BoundDropCounter` in `math/algorithms/a_star_search.py`,
  BL-33): off by default, left on the *instance* as `bound_drops`, and deliberately never on
  `SearchResult`. That is the precedent this plan generalizes.
- `src/mllib/math/graph/visualizer.py` (matplotlib animation of a traversal) is imported by nothing but
  `tests/math/test_smoke_modules.py`; it is a BL-14 "kept, smoke-tested" module, not a used one.
- No visualization **decision**, **backlog item** or **extension point** exists: `docs/DECISIONS.md`
  ends at D-31, `docs/BACKLOG.md` at BL-42, `docs/ARCHITECTURE.md` §1 has four layers and §4 lists no
  observation extension point.
- Dependencies are `numpy`, `pandas`, `scikit-learn`, `networkx`, `matplotlib` only (plus the opt-in
  `torch` group, D-31). Nothing here adds one: the pages are vanilla JS.

## 3. Phases and slices

Each slice: branch off the previous slice's branch; `uv run ruff check . && uv run ruff format --check .`;
`uv run pytest`; reviewer-agent; Xavier commits and opens the PR. Tests and records ship in the same PR as
the code they cover. "Done when" is in addition to that.

**Acceptance, verbatim, on all four slices:**

- "Math never imports ML; ML composes math; scripts (composition roots) wire data to models"
  (`docs/ARCHITECTURE.md` §1)
- "a run with the null recorder is byte-identical to today"
- "never on a result type (D-28)"

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 1 | `feat/visualization-recorder` | `math/recorder.py` (`AbstractRecorder`, frozen slotted `Frame`, `NullRecorder`); `AStarSearch.__init__` takes `recorder`, defaulting to a fresh `NullRecorder` (never a shared mutable default), with **two** guarded moments in the base `_search` loop — an expansion (after the expanded state's children are priced and pushed) and the goal pop the search returns on, so a walkthrough ends at the answer; variants supply extras through `_recorder_extras()` and never override `_search`. `visualization/recorders/astar_landmark.py` (`AStarRecorder`, `AStarFrame`). Records: BL-43, this plan, D-32, CONTEXT.md area, learning log; process edits (CLAUDE.md `viz` scope, CI/pre-commit TODO grep over `*.html`/`*.js`, pyproject package-data). | Both baselines byte-identical with the default recorder; `NullRecorder.enabled` is `False` and `AStarRecorder.enabled` `True`, each asserted; on ≥ 2 fixture kernels `len(recorder.frames) == result.nodes_expanded` and the last frame is consistent with the returned result (its state, its bound, its expansion count); pruned frames carry what the expansion declined to store, and the incumbent — which the anytime engine inherits, since it tracks nothing of its own mid-run (its certified gap is computed at the stop); a frame's caption is a full sentence. |
| 2 | `feat/visualization-walkthrough` | `visualization/recording.py` (`Recording`, `save`, `load`), `visualization/html_renderer.py` + `template.html` + `walkthrough.js`, `visualization/views/astar_landmark.py`, `examples/astar_landmark_walkthrough.py` (RBF chain 8×8 at k = 3; SPECTF n = 14, k = 2), one committed fixture recording, CLI `python -m mllib.visualization.render`. | A recording round-trips through `save`/`load` with every float bit-identical and no timestamp in the file; `load` refuses an unknown `schema_version` with a clear error; the rendered page contains the recording's JSON verbatim and the stepper controls (previous, next, slider, play, keyboard handler); the example writes both a recording and a page to `--output-dir`; the CLI re-renders a committed fixture to a page byte-identical to the example's. |
| 3 | `feat/visualization-two-hot` | Stacked on `feat/two-hot-span-harness`. `visualization/recorders/two_hot_span.py` (every step: the training loss; a full frame every `frame_every` steps, default 1 — V, R(v_j), the rounded pairs, component labels, E\*, Ê), `visualization/views/two_hot_span.py` (V heatmap; pair graph on embedded positions — ladder layout for the roach graph, seeded spring layout otherwise; component count and Ê over steps), `examples/two_hot_span_walkthrough.py` (roach G_5 λ = 10 spectral; karate λ = 1). | Every frame's E\* and Ê equal the problem module's `pinv` functions recomputed on that frame's V (P-2 of the two-hot plan, D-31: never the ridge projector); the per-step losses equal the optimizer's `loss_history` element for element; the layout is seeded and a rerun produces the same positions; torch-dependent tests `pytest.importorskip` so the default suite is unaffected. |
| 4 | `feat/visualization-graph-search` | BFS and DFS take the recorder through the same base seam; `math/graph/visualizer.py` deleted; `examples/graph_search_vs_networkx.py` switched to writing a recording plus a page. | BFS and DFS each produce frames whose visit order equals the traversal they return; `visualizer.py` is gone, its BL-14 entry amended to say so with the closing commit, and `tests/math/test_smoke_modules.py` no longer imports it; the BFS example's current output was fingerprinted **before** the switch and the new example reproduces those traversals; the matplotlib comment in `pyproject.toml` no longer names a deleted module. |

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Recordings grow large enough to be unshippable (P-6 records the full frontier). | Walkthrough cells are small by design and are chosen in the examples, not by a caller; committed fixtures obey the 1 MB cap the pre-commit `check-added-large-files` hook already enforces and the D-19 rule for data. If a cell exceeds it, the cell is wrong for a walkthrough, not the recorder. |
| The page and the schema drift apart, so an old recording renders wrongly and silently. | The schema is versioned and `load` refuses an unknown `schema_version` (P-3); the renderer reads the version too, and the round-trip plus render tests run on the committed fixtures, which age with the schema. |
| Instrumentation slows the hot loops even when off. | The `enabled` guard is one boolean read per step on a branch that already exists (P-1), and the frames are built by the child, off the loop's path. Measure once, on `examples/nystrom_batched_bounds.py --no-timings` cells, comparing wall time with and without the default recorder; record the number in the slice-1 PR. |
| A future problem is instrumented by editing an engine. | D-32's rule: a new problem adds a child recorder and a view. If a slice finds itself editing `_search`, the extras hook is the missing piece, not a second call site. |
| Narration drifts from the recording, so a page says something the frames do not. | Tests assert the narration strings for fixture frames — the opening's first line, frame 0, one mid-run frame and the ending's first line, as string equality — and recompute every key moment's frame index from the frames themselves (the peak is `argmax(frontier_size)`, not a number typed into the view). A narration is never edited by hand: the fix for a wrong sentence is the derivation, and re-recording the run re-narrates it (P-7, D-34). |
| A narration needs a value the frame lacks. | The known absences, stated rather than papered over. **A\***: the certified gap exists only in the result, never per frame — the per-frame `gap` quantity is the incumbent over the frontier minimum, which is not the same number. **Two-hot**: drift is only the run-wide `max_zero_sum_violation` in the result; the column↔pair correspondence is lost on frames where a column rounds to zero (`rounded_pairs` is then shorter than r); the harness datum is not in the recording at all. **Traversal**: parent pointers are absent, so no sentence may say which node discovered which. Mitigation: a narration says only what the frame holds, the ending panel names the absence in prose, and **no frame field is added in this phase** — a need for one is a backlog line, not a call site (D-32, D-34). |

## 5. Immediate next actions

1. Slice 1 opens from `main` @ `1cc6fb1` on `feat/visualization-recorder`, carrying BL-43, this plan,
   D-32, the CONTEXT.md area and the process edits.
2. Measure the guarded loop once (risk 3) and put the number in the slice-1 PR body.
3. Note the CI/pre-commit TODO-grep change for back-port to `engineering-standards` in the same PR.
4. Slice 3 waits on `feat/two-hot-span-harness` (two-hot plan slice 2.3) landing.

## 6. Phase 2 — the explainability layer (2026-09-10)

### 6.1 Why

Phase 1 made a run watchable; it did not make it legible. The pages show the run and do not explain it:
a reader watches the frontier move without being told what a bound is, sees Ê fall without knowing what
it is read against, and reaches the last frame with no statement of why the run stopped there. The
recorder's caption says what happened in one frame; nothing says what the algorithm is doing, which
frames decided the run, or how to read the ending. The bar this phase is held to: **a reader with the
page and nothing else must be able to say what the algorithm did and why the run ended where it did** —
without the plan, the source, or a person to narrate it.

### 6.2 The seven elements

Each is added once, in the shared layer, and filled by every view (or left empty).

1. **Opening panel** — four lines above the stepper: the instance, the algorithm in one sentence, the
   question the run answers, and how to read the page.
2. **Frame narration** — one sentence per frame, derived from that frame and the one before it, above
   the recorder's own caption, which stays and is restyled muted.
3. **Key moments** — the frames a view derives as the ones that decide the run, drawn as labelled
   markers under the slider; clicking one jumps to its frame, and its reason joins that frame's narration.
4. **Legend and hover** — a legend panel naming every drawn element and the quantity it stands for, and
   a `<title>` on each drawn element giving that element's own numbers.
5. **Quantity strip** — a row of term/value pairs between the stepper and the stage, one value per frame,
   `—` where the frame lacks it; hidden for views with no quantities.
6. **Ending panel** — two to four lines, always visible: why the run stopped, and what to read the result
   against, including the values the recording does not hold.
7. **Glossary tooltips** — a `CONTEXT.md` term appearing in the prose is wrapped in an `<abbr>` carrying
   its one-line definition, from the shared glossary dictionary.

### 6.3 Slices

Each slice: branch off the previous slice's branch; `uv run ruff check . && uv run ruff format --check .`;
`uv run pytest`; reviewer-agent; Xavier commits and opens the PR. Both baselines stay byte-identical —
nothing below `visualization/` changes in this phase — and no recorder call site or frame field is added.

| Slice | Branch | Change | Done when |
|---|---|---|---|
| A | `feat/visualization-explain` | The shared layer: `visualization/explain.py` (the frozen `Moment`, `LegendEntry`, `Quantity`, `Explanation` dataclasses and the `fmt_state` / `fmt_num` / `fmt_delta` helpers), `visualization/glossary.py` (the whole `GLOSSARY`, including the two-hot and traversal terms, and `used_terms`), the template's fixed slots (opening, moments, quantities, narration, legend, ending), the marker strip under the slider and the quantity strip, and `walkthrough.js` reading `layout.explain`. Plus the A\* view's own content: narration, key moments, legend and hovers, quantities; a third cell `rbf_chain_8x8_k3_capped` in `examples/astar_landmark_walkthrough.py` on `AnytimeAStarSearch(max_expansions=7)` so the incumbent line and the capped ending appear on a real page (7, not 5: the first complete selection is priced on the fifth expansion, whose frame a cap of 5 stops the run before recording, so that page would hold no incumbent and no pruned child); and the docs — this section, D-34, the `CONTEXT.md` two-hot area (the glossary test reads those `**Term**:` headings, so they ship in this slice), BL-43's phase-2 line, `examples/README.md`, the learning-log entry. | A layout without `explain` still renders, with every new element hidden (the phase-1 fallback is kept and tested); the four asserted A\* strings match; every moment's frame index is recomputed from the frames in the test; the legend's keys equal the `data-legend="…"` values regexed out of `astar_landmark.js`, in both directions; `Explanation.glossary` keys are a subset of `GLOSSARY` and each occurs in the page's prose; every `GLOSSARY` key is a `**Term**:` heading of `CONTEXT.md`; the page still contains no `http`, no `fetch` and no `type="module"`, and its `</script>` count is unchanged; the capped cell renders an ending naming the expansion cap and the certified gap. |
| B | `feat/visualization-explain-two-hot` (on A) | The two-hot view's explanation: the 2-hot rule as a named constant `TWO_HOT_SHARE = 0.95` — a column is 2-hot when its two largest squared coordinates hold at least 95 % of its squared norm, so its roundability is at most 0.05, and a zero column is never 2-hot; narration and moments over the full frames; the quantity strip (E\*, Ê, Ê − Σλ, component count, 2-hot columns, mean R(v_j), training loss); the legend, with **dashed chords** for rounded pairs that are not edges of the graph, so a pair the graph does not contain is visible as such. | The 2-hot count, the distinct pairs and the pairs-that-are-edges are recomputed in the test from the fixture's `spanning_set` and `problem.layout.edges` and match the narration's numbers; light frames narrate the training loss only and say which full frame the picture is from; the ending states the run-wide `max_zero_sum_violation` as drift and says the datum is not in the recording; the torch-dependent parts stay behind `pytest.importorskip`, and the committed two-hot fixture renders without torch. |
| C | `feat/visualization-explain-traversal` (on B) | BFS and DFS get elements 1, 2, 3, 4 and 6 only — opening, narration, key moments, legend and hover, ending. No quantity strip: `Explanation.quantities` is `()` and the strip is hidden. | The narration never names a discovering node (there are no parent pointers in a traversal frame, and a test greps the view for the words that would imply one); the `target pending` moment's frame index is the first frame whose pending contains the target, recomputed in the test; the ending distinguishes "target visited" from "the queue emptied"; the strip is `hidden` on a traversal page. |

### 6.4 Key moments per view

| View | Moments |
|---|---|
| A\* landmark | `goal priced` — the first frame where a child at goal depth is pushed; `incumbent` — each frame where the incumbent first appears or improves; `peak` — the frame where the frontier is largest (first occurrence); `proved` / `returned` — the goal frame, or `capped` on the last frame of a run the expansion cap stopped. One marker per frame; when two coincide the terminal moment wins, then `incumbent`, `goal priced`, `peak`. |
| Two-hot span | `first 2-hot` — the first full frame with at least one 2-hot column; `all 2-hot` — the first full frame where every column is 2-hot; `K reached` — the first full frame whose component count equals K (absent if it never does); `Ê drop` — the full frame with the largest fall of Ê from the previous full frame; `Ê settled` — the last full frame at which Ê changed, when a full, non-end frame follows it (the end frame restates the last step and is excluded). Priority: `K reached`, `all 2-hot`, `first 2-hot`, `Ê drop`, `Ê settled`. |
| Graph traversal | `target pending` — the first frame whose pending container holds the target; `peak` — the frame where the container is longest (first occurrence); `target visited` — the frame that visits the target. |
