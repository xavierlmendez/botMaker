<!-- transcribed from the research repo's evidence-to-plan run 2026-09-09..10; §5 of that plan was the source -->
# Two-hot spanning sets — minimal rcut prototype

Status: **accepted** (2026-09-10, slice 2.1 opened) · Written 2026-09-09 · Owner: Xavier
Target: three stacked slices from `feat/two-hot-span-problem` into `main` (`main` @ `85f2205`).
Why the code exists: `~/develop/research/candidates/constrained-eigenvectors.md` and Prof.
Schweitzer's handoff brief §20 ("the exact next step").

---

## 0. Why

The handoff asks for "the smallest possible direct PyTorch implementation of this objective, not a
full autoencoder and not yet the alternating analytical solver". Two facts settled in the research
session make that implementable without guesswork: R(u) ≤ 1/2 for zero-sum u with equality exactly at
2-hot vectors (proved), and Ê of a rounded within-block spanning forest is exactly the RatioCut of the
partition, with the λ = 0 minimum of E over r = n − K vectors equal to the sum of the K smallest
eigenvalues of XXᵀ (proved, machine precision on five random weighted graphs). So the harness has a
floor to report against and a tiny-graph oracle that is a partition enumeration, not a search.

`main` @ `85f2205` has no torch: absent from `pyproject.toml` dependencies and groups, zero matches
in `uv.lock`, no import under `src/`. `networkx>=3.2` is already a core dependency, so the connected
components of the rounded-pair graph come free. The module triple is mirrored **by role** —
problem / optimizer / harness, as `nystrom_landmark_problem.py` (374 lines) /
`nystrom_landmark_selectors.py` (273) / `nystrom_uci_harness.py` (336) — not by base class:
this is a gradient-optimized continuous relaxation, so `AbstractGraphProblem` and
`SearchCostFunction` do not apply and are not subclassed.

## 1. Decisions and assumptions

| # | Decision |
|---|---|
| P-1 | torch enters as a PEP 735 `[dependency-groups] torch` group with an explicit CPU index (`docs/DECISIONS.md` **D-31**), not as an extra: extras are published metadata and this library publishes no torch consumer. |
| P-2 | The **training** loss uses the ridge projector `V(VᵀV + εI)⁻¹Vᵀ`; every **reported** E\* and Ê uses the exact `pinv` projector in numpy. The objective is never reported through a regularized projector. |
| P-3 | The rounded-cut oracle for n ≤ 10 is a brute-force set-partition enumeration, not the A\* engine. |
| P-4 | K = number of clusters, r = n − K = number of spanning vectors. `k` is not used in this module family. |

| # | Assumption | Why |
|---|---|---|
| A-1 | Adam on the unconstrained ratio penalty reaches a usable V on toy graphs. | No published result covers plain autodiff on an ℓ1/ℓ2 ratio penalty; the field uses ADMM, ℓ1−αL2 bisection, or moving-balls, all to critical points only. Slice 2.2's no-NaN test is the check; Hoyer's projection step is the named fallback. |
| A-2 | The CPU wheels run on the R620 (no AVX2). | PyTorch does not document an AVX2 requirement; issue #94021 shows some ops already fail without it. Recorded as **BL-42**; first runs are on the Mac, so nothing depends on it. |

## 2. Current state (evidence)

Captured 2026-09-09 at `85f2205`. No torch anywhere in the tree. `networkx>=3.2` core. Tests green.
No module in `math/` or `ml/` optimizes a continuous objective by gradient; this is the first.

## 3. Phases and slices

Each slice: branch off the previous slice's branch; `uv run ruff check . && uv run ruff format --check .`;
`uv run pytest`; reviewer-agent; Xavier commits and opens the PR. Each slice is ≤ ~300 changed lines
including tests, with its tests and records in the same PR. "Done when" is in addition to that.

**Acceptance, verbatim, on all three slices of phase 2:**

- "VᵀV = I is never imposed (source §3, §17)"
- "gradient track only; the alternating solver is parked (source §19)"
- "Ê is the headline number; R(v_j) is a per-column diagnostic (source §13)"
- "rcut first; ncut only in phase 4 (source §20)"

| Slice | Branch | Change | Done when |
|---|---|---|---|
| 2.1 | `feat/two-hot-span-problem` | `math/graph/two_hot_span_problem.py`, numpy only: X = E for rcut from a networkx graph; `projector_residual` via `pinv`; `round_columns` → (argmax, argmin) pairs; `rounded_cut` = Ê; `spectral_floor` = Σλ; `brute_force_rcut` by partition enumeration; the four test graphs (roach G_5 with n = 20 and m = 23, karate, seeded planted partition, two-moons kNN). | Ê from a rounded within-block spanning forest equals RatioCut on ≥ 5 random weighted graphs (n = 7–11, K = 2–3) to 1e-10; `brute_force_rcut` matches an independent enumeration at n ≤ 8; roach G_5 has 4k vertices and 5k−2 edges for k = 5. |
| 2.2 | `feat/two-hot-span-optimizer` | `math/algorithms/two_hot_span_optimizer.py`: trainable W, V = W − c(cᵀW)/(cᵀc), optional column normalization, loss E_ridge(V) − λ Σ_j R(Cv_j), Adam. `pyproject.toml` gains the `torch` group + explicit CPU index; CI runs `uv sync --locked --dev --group torch`; torch-dependent tests use `pytest.importorskip`. **D-31** and **BL-42** in the same PR. | `cᵀv_j = 0` to 1e-12 at every step of a seeded run; a test asserts the reporting path never uses `epsilon`; a 20-step run on the roach graph is finite (no NaN) with finite gradients; on a seeded 200-step run on one n ≤ 8 graph, the reported Ê is ≥ the brute-force optimum from 2.1 and the report states both (S6). |
| 2.3 | `feat/two-hot-span-harness` | `ml/projects/two_hot_span_harness.py`: the four graphs × a λ sweep; the datum computed three ways; one JSON report per graph. | Each report carries E\*, Ê, Ê − E\*, R(v_j) per column, the clustering, Σλ, Ê − Σλ, and the datum's rounded rcut under `kmeans`, `discretize` and `cluster_qr`; at n ≤ 10 it also carries the brute-force optimum and asserts Ê ≥ it. |
| 2.4 | `feat/two-hot-span-schedule-adjacency` | experiment: a learning-rate scheduler and an optional per-column graph term (Laplacian form or edge product), both off by default so the default objective stays the brief's §20 objective; the probe example | done when: defaults bit-identical, schedule history tested, both forms tested against hand values, the probe's tables in the PR body |
| 2.5 | `feat/two-hot-span-diversity` | experiment: an optional column-diversity term `+ ν Σ_i ℓ_i²` on the vertex load ℓ_i = Σ_j p_ij of the brief's §9 distributions, off by default. The **first term that couples the columns**: everything before it (the span term, R(v_j), the graph term) scores a column on its own, so nothing could tell two columns apart from one column used twice. Harness pass-through and the probe example. | done when: defaults bit-identical, the hand values tested (¼ Σ_i deg_i² on exact 2-hot columns, the r²/n floor on a uniform spread), the probe's tables in the PR body |
| 2.6 | `feat/two-hot-span-roach-g20` | a second cockroach instance, from He, Gu & Zhang 2012 (`docs/REFERENCES.md` **L5**): `roach_g20_instance()` = `roach_graph(20)`, n = 80, m = 98, **K = 3** with planted labels ladder / top antenna / bottom antenna, appended as the fifth `default_test_graphs()` entry. The walkthrough, both probes and `run_default_suite` cover it; the ladder layout and the probes' cross-path column read k and n/2 off the graph instead of the k = 5 literals. | done when: n/m/connectivity, planted RatioCut = 0.15, the K = 2 antennae-vs-ladder cut = 0.10, and the three planted blocks as the components of G minus the two antenna edges are all tested; the paper's λ2/λ3/λ4 = 0.0057/0.0062/0.0246 are reproduced to four decimals; the harness and both probes run on it and their tables are in the PR body; the roach G₅ visualization fixture stays byte-identical. |
| 2.7 | `feat/two-hot-stress` | the stress ladder (**BL-45**, research seed `sessions/2026-09-18-two-hot-stress.md`): `ml/projects/two_hot_span_stress.py`, a harness-tier runner over a **pre-registered** grid — rung 0 the prototype's three graphs at three seeds, rung 1 the planted partition at (100, 2), (201, 3), (500, 5) × three cluster strengths by expected cross-block degree (1/3/6 against a within-block degree of 12) × three seeds, rung 2 the same generator at n = 1000/2000 (cloud only), rung 3 polbooks / football / email-Eu-core. λ = 10, lr 0.05, 3000 steps, checkpoints (300, 1000, 3000); couplings (ν, μ) ∈ {(0,0), (10,0.3), (10,0), (0,0.3)} × {random, spectral} everywhere plus a ν × μ sweep on rung 1 alone. One JSONL record per graph, per cell and per oracle check; a subclassed `AbstractStepRecorder` takes the checkpoints and enforces the per-cell cap; a spawn pool with `maxtasksperchild=1` gives each cell its own peak RSS. **No engine change**: the problem, optimizer and harness modules are byte-identical, and the rung-0 oracle is what proves it. | done when: the oracle reproduces the prototype's `roach_g5`/`karate`/`roach_g20` reports to 1e-8 (exact on labels and component counts) and re-reaches Ê = 4/15 at K = 2 on the tuned cell, and rungs 1-3 refuse to run without it; the cell counts 72/432/16/72 and the rung-1 generator parameters are pinned by tests; Ê ≥ Σλ and E\* ≥ Σλ are checked per cell; the memory budget and the laptop's refusal of rung 2 are tested as scheduling logic; the run is resumable by cell key and every cell is recorded, failures included. |
| 2.8 | `feat/two-hot-span-visualization-stack` (this branch) | the `two_triangles` instance and its walkthrough pages: `two_triangles_graph()` and `two_triangles_instance()` in `math/graph/two_hot_span_problem.py` — n = 6, K = 2, two weight-1 triangles joined by the weight-0.1 bridge (2, 3) — appended as the **sixth** `default_test_graphs()` entry by the 2.6 index-stability rule. It is the first default-suite graph inside `BRUTE_FORCE_NODE_LIMIT`, so its optimum is enumerated (1/15, at the planted labels) rather than referenced, and all three datum roundings reach it: a run that misses 1/15 here has missed it for reasons of its own, which is what makes the instance a test of the *optimizer* and not of the objective. It is also where λ gets its scale — λ·r/2 = 20 at λ = 10 against Σλ = 0.064 — recorded as an observation in `docs/LEARNING_LOG.md` and as **BL-46**, not as a decision. The walkthrough gains the graph at λ = 0.1 with a schedule of its own (5000 steps, lr 0.01, applied only when the CLI left `--steps`/`--learning-rate` at their defaults) and an explicit six-node layout, plus three rendered pages: spectral, random seed 0 and random seed 3. | done when: n/m/weights/connectivity, Σλ = 0.063771, the enumerated optimum 1/15 at the planted labels and the three datum roundings are all tested; the default suite is six instances with `two_triangles` last and the harness asserts Ê ≥ 1/15 on its rows; the older three graphs carry no schedule of their own and the roach G₅ walkthrough fixture stays unchanged; the three pages are rendered into `visualization/walkthroughs` and are under the 1 MB pre-commit limit. |
| 4.1 | `feat/two-hot-span-ncut` | **Conditional** on 2.3's rcut behaving reliably (source §20). c_i = √d_i, C = diag(c), X = D^{-1/2}E threaded as parameters of the same three modules. | Ê = NCut on ≥ 5 random weighted graphs to 1e-10; the harness emits ncut rows beside rcut rows; the diff adds parameters, it does not copy modules. If this slice is not reached by 2026-09-25, it is **BL-41** and parked as P23 in the research repo. |

Slice 2.6 carries **no** `docs/DECISIONS.md` entry either, for the same reason as 2.4 and 2.5: it
adds an instance to report on, not a settled default — nothing about the objective, the rounding or
the defaults changes with it. The two choices a reader might want to re-litigate are argued where
they are made rather than here: appending `roach_g20` as the *last* `default_test_graphs()` entry,
so every index-based reference to that tuple still names the instance it named, and taking planted
labels that are **not** the RatioCut optimum (0.15, against 0.10 for the K = 2 merge of the two
antennae) as the instance's reference — both in `roach_g20_instance`'s docstring and in the
`docs/LEARNING_LOG.md` entry "The eighty-node cockroach: three parts, not two".

Slice 2.5 carries **no** `docs/DECISIONS.md` entry either, for the same reason and by the same
test: a term that is off by default is an experiment, and it becomes a decision the day one value of
ν is what the harness runs by default. On the 2026-09-10 probe no single value earned that. The
result is nonetheless the strongest the prototype has produced — on the roach, λ = 10, random init,
`edge_product` μ = 0.3 and ν = 10 round to exactly K = 2 components at Ê = 0.2667, the brute-force
optimum, and stay there at 1000 and 3000 steps — but on karate the best Ê is still ν = 0, so what is
owed next is a second graph where ν wins, not a default. See `docs/LEARNING_LOG.md`.

Slice 2.4 carries **no** `docs/DECISIONS.md` entry: a schedule and a graph term that are both off by default are an experiment, not a settled decision. One becomes a decision the day a form becomes the default objective — and on the 2026-09-10 probe neither did: the best Ê on both the roach and karate is μ = 0 under the constant schedule.

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Adam produces NaN at zero entries of ‖v‖₁. | Slice 2.2's finite-value test on the roach graph is the gate; the fallback is Hoyer's projected-gradient step, which never imposes orthogonality. |
| V goes rank-deficient and the projector's gradient breaks. | P-2: ridge in training, `pinv` in reporting. QR is excluded — reduced mode returns an n×r Q whatever the rank, so QQᵀ over-spans a rank-deficient V and is a different function. |
| The prototype's numbers get quoted before the identity test passes. | 2.1 lands the identity test before any optimizer exists; the harness asserts Ê ≥ the brute-force optimum at n ≤ 10 on every run. |
| torch bloats CI. | The group is opt-in; only the `--group torch` job installs it, and the CPU index keeps the Linux wheel off the GPU build. |

## 5. Immediate next actions

1. Xavier reviews § 1 and § 3; amends or accepts; plan status → `accepted`.
2. Slice 2.1 opens from `main`.
