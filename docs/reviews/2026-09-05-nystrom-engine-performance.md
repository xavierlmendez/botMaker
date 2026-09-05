# Nyström search engine — performance options · 2026-09-05

Context for BL-31. Written after the first EXP-09a attempt (research program, truncation
calibration) was paused for memory at n = 1,000. Every number below was measured in that session on
this machine (Apple Silicon, NumPy on Accelerate); none is quoted from a paper.

## What is expensive, and why

**Memory: the frontier.** A* stores every child it has priced and not yet expanded. At n = 1,000 an
expansion pushes up to a thousand entries and pops one, so after E expansions the frontier holds
about E × 1,000 entries. The cell that finished (sat, scale 0.25, k = 4, δ = 1e-4) needed 40,631
expansions. An entry today is a `(bound, sequence, state tuple)` heap tuple: **124 bytes** measured.
This is the EXP-01 failure mode, and the downdate made it worse by making expansions cheap enough to
reach it inside a ten-minute cap. The arithmetic (kernel, reduced coordinates, per-parent buffers)
is about 8 MB each and is freed per parent; it is not where memory goes.

**Time: one expansion at n = 1,000** (sat kernel, scale 1; heap pushes are the search's own cost):

| δ | retained rank | `eigh` of the parent | secular solve, all children | heap pushes | total |
|---|---|---|---|---|---|
| 0 | 1000 | 89 ms | 110 ms | 0.1 ms | 199 ms |
| 1e-4 | 530 | 19 ms | 54 ms | 0.2 ms | 82 ms |
| 1e-3 | 241 | 4 ms | 28 ms | 0.2 ms | 34 ms |
| 1e-2 | 61 | 0.2 ms | 8 ms | 0.1 ms | 8 ms |

At full rank the secular solve costs more than the decomposition it was meant to make cheap: plan
decision P-5's trigger ("revisit Gragg only if the solver, not the parent `eigh`, dominates") has
fired at r = n.

**Frontier entry size by representation** (n = 1,000, k = 4, measured with `tracemalloc`):

| representation | bytes per entry |
|---|---|
| today: `(float, int, tuple[int, ...])` | 124 |
| bitmask over 1,000 points as a Python int | 253 |
| four indices packed into one int | 151 |
| NumPy arrays: 8-byte bound + four 2-byte indices (CSR-style, fixed row width) | 16 |

Any per-entry Python object stays near 120 bytes; only contiguous arrays change the order of
magnitude. In the sparse-matrix vocabulary of the Data Representations handouts: the frontier is a
binary matrix with k − 1 ones per row (density 0.3%); each state is stored the way LIL stores one
row (an index list, no values), but the container is a heap, not a list of rows; CSR with an implicit
`indptr` is the array layout above, and the handout's own remark applies — the win is contiguous
arrays and compiled code, not asymptotics. COO would add a row index per nonzero for nothing.

## Options

Grouped by what each attacks. "Baseline-safe" means expansion counts and returned subsets are
unchanged, so `tests/ml/nystrom_search_snapshot.json` and the BL-27 reference cells prove it.

### Memory

| # | Option | Effect | Cost | Baseline-safe |
|---|---|---|---|---|
| M1 | **Prune at push time.** A greedy selection before the search gives an incumbent cost; a child whose bound is strictly above it is never stored. A* stops on the first goal popped, so such a child would never have been expanded. | Removes most of the frontier on flat kernels. | ~20 lines in `AStarSearch` + an optional incumbent; ½ day | yes (ties: prune only strictly above) |
| M2 | **Array-backed frontier.** States stay tuples at every boundary; the queue is fixed-width integer index arrays + a bound array with a binary heap written over them, decoded on pop. | 124 → 16 bytes per entry (~8×). | ~60 lines inside `AStarSearch`, a state-width hook; 1 day | yes (FIFO tie-break preserved) |
| M3 | Bitmask / packed-int states everywhere | **Worse** (253 / 151 bytes) and touches every consumer of `state` (11 sites in the Nyström module, 32 readers of `result.state`). | — | rejected |
| M4 | Per-run memory / expansion caps | Protects the machine; labels a run `memory_cap`. Already in the EXP-09a runner (`--max-rss-gb`). | done | n/a (no answer changes) |

### Time per expansion

| # | Option | Effect | Cost | Baseline-safe |
|---|---|---|---|---|
| T1 | **Spectrum truncation δ** (D-26, built) | 199 → 34 ms at δ = 1e-3; bound gets looser (EXP-09a measures the extra expansions). | done | no by design (δ > 0 changes counts, never the optimum) |
| T2 | **Gragg / Melman iteration for the secular solve** (P-5's trigger has fired) | ~halves the expansion at δ = 0 (110 ms → a few ms for the solve). | small; `secular_equation.py` + T1 tests unchanged in contract | yes (same roots to rounding) |
| T3 | GPU port of `eigh` + secular solve (CuPy) | 5–10× per expansion at full rank; nothing for memory. | rented hardware (no CUDA here; Metal has no symmetric eigensolver, D-24); port + per-expansion transfer | yes in principle |
| T4 | Parent-from-grandparent downdate (spec §7, deferred) | one `eigh` per grandparent | new bound path + equivalence tests | yes if done right |

### Throughput across cells

| # | Option | Note |
|---|---|---|
| C1 | More workers on this machine | limited by memory, not cores (large runs already cut to 2 workers) |
| C2 | EC2 memory-optimised instance (r7i.8xlarge, 32 vCPU / 256 GB, ~$2/h on-demand; spot ~⅓) | ~$13–21 for an overnight run; buys RAM and cores, delays the frontier wall only by RAM ratio on flat cells; Linux OpenBLAS rounds ties differently (cost-based agreement checks are unaffected) |
| C3 | Expand several parents per step (batched A*) | would let a GPU batch decompositions, but changes expansion counts and needs its own certificate argument; not a drop-in |

### Not viable

- Sparse storage of the kernel (COO/LIL/CSR): RBF kernels have no exact zeros and the eigendecomposition needs the dense matrix; thresholding changes the objective.
- Renting a GPU before any engine change: flat n = 1,000 cells need 10⁵–10⁶ expansions; a faster expansion reaches the memory cap sooner.

## Suggested order

M1 → T2 → M2 → rerun EXP-09a → C2 for the overnight large set. M1, M2 and T2 are each a normal
slice with the "cost changed, search did not" proof.

## Where the evidence lives

- Timings and entry sizes: this session, 2026-09-05; scripts were one-off and are reproduced by the
  tables above (the per-expansion timing uses `NystromCssCostFunction._downdated_bounds` pieces on
  the sat n = 1,000 kernel at scale 1, seed 7).
- EXP-09a runner and partial results: `~/Desktop/BotMaker/nystrom-grid/exp09a.py`, `exp09a_results.jsonl`
  (104 runs; every δ = 0 reference run reproduced S1's expansion count and optimum).
- Research-side context: `~/develop/research/nystrom/experiments/EXP-09a-truncation-calibration.md`.
