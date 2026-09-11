"""The two-hot span stress ladder: the composition root for the pre-registered grid run.

**The ladder** (`docs/plans/2026-09-two-hot-span.md` slice 2.7, research seed
`~/develop/research/sessions/2026-09-18-two-hot-stress.md`). Rung 0 is the prototype's own three
graphs — roach G_5, karate and the eighty-node cockroach — and doubles as the runner's oracle: the
frozen JSON reports are reproduced to 1e-8 before any new graph is allowed to run. Rung 1 is the
planted partition at (n, K) = (100, 2), (201, 3) and (500, 5), three cluster strengths by expected
cross-block degree (clear 1, moderate 3, weak 6) against a within-block degree of 12, three seeds
each. Rung 2 is the same generator at moderate strength and n = 1000 (K = 4) and n = 2000 (K = 8),
one seed, cloud only. Rung 3 is the field's own instances: polbooks, football and email-Eu-core.

**What is fixed in advance, and stays fixed.** λ = 10 everywhere; the coupling grid is the four
fixed points (ν, μ) ∈ {(0, 0), (10, 0.3), (10, 0), (0, 0.3)} at both inits on every rung, plus a
ν ∈ {3, 10, 30} x μ ∈ {0.1, 0.3, 1} sweep at random init on rung 1 alone; learning rate 0.05 on the
constant schedule; 3000 steps with checkpoints at 300, 1000 and 3000. `edge_product` is the
adjacency form wherever μ ≠ 0. Nothing is tuned per graph: a number from an unlisted setting is not
a number this run may report, which is why no coupling reaches this script as a flag.

**The judge.** Ê, the rounded cut, is the headline — it is the RatioCut of the partition the run
actually produces. E\\* and the spectral floor Σλ are reported beside it, never instead of it, and
R(v_j) is a per-column diagnostic and never the criterion. Every cell is reported, including the
ones that reach fewer than K components, time out, exceed the memory budget or raise.

Two invocations, and only two:

    # laptop, rungs 0, 1 and 3, detached overnight
    nohup uv run --group torch python examples/two_hot_span_stress.py \\
        --rung 0 --rung 1 --rung 3 --hours 10 > /dev/null 2>&1 &

    # cloud, the whole ladder including rung 2
    uv run --group torch python examples/two_hot_span_stress.py \\
        --host cloud --workers 8 --memory-budget-gb 32 \\
        --rung 0 --rung 1 --rung 2 --rung 3

`--allow-rung-2` runs rung 2 on the laptop host, which is where it ends up: the cloud instance's
oracle fails on Linux — the spectral-init rows differ under OpenBLAS's eigh basis and one karate
random-init row flips its labels at a near-tie — so the ladder's top rung stays on the machine whose
arithmetic the prototype's numbers were established on. The flag lifts the host refusal and nothing
else; the memory budget still admits at most three 2000-node cells (2.1 GB each) at once at 8 GB.

Results land in `examples/two_hot_span_stress_results/` (gitignored), one JSON line per finished
cell, resumable by cell key; the frozen copy lives in the research repo under `nystrom/data/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mllib.ml.projects.two_hot_span_stress import main

DEFAULT_RESULTS = (
    Path(__file__).resolve().parent / "two_hot_span_stress_results" / "two_hot_stress_results.jsonl"
)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--results" not in argv:
        DEFAULT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        argv = [*argv, "--results", str(DEFAULT_RESULTS)]
    raise SystemExit(main(argv))
