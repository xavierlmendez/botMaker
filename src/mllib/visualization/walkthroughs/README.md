# Rendered walkthroughs

One offline HTML page per recorded run, opened from `file://` with no server. Each page embeds its
own recording, so the `.json` beside it in the render folder is not kept here. Every sentence on a
page is derived from the recording at render time (D-34).

Rendered 2026-09-10 from this branch's code with the commands below, into this folder. Several runs
use a sparser full-frame cadence than the default (`--frame-every 5`) so that each page stays under
the 1 MB pre-commit limit; the light frame at every step, and so the loss curve, is unaffected.

The three `two_triangles` pages are rendered at `--steps 1500` rather than at the 5000 steps that
instance's own schedule carries, and that is the one place where a page is not the run the plan
quotes. `--frame-every` cannot make a 5000-step page fit: a page costs about 530 bytes per *step* —
a light frame is recorded at every step whatever the full-frame cadence (≈ 374 B, most of it the
null-valued keys a light frame still writes out), and the explain layer derives one narration
sentence per frame on top (≈ 160 B) — so the 5000-step pages come to 2.9 MB and 1 MB is reached at
about 1600 steps. Nothing a reader looks at is lost at 1500: Ê settles at step 200 on the spectral
run, 1100 on random seed 0 and 25 on random seed 3, so all three pages end on the same Ê, the same
component count and the same partition as the 5000-step runs, and only E\* differs — in the fourth
decimal, still creeping toward the floor. The fix that would let the full run be the committed page
is a light-frame cadence on `TwoHotSpanRecorder` plus matching narration thinning; that is a slice
of its own, not a flag this example has.

| Page | Script and flags |
| --- | --- |
| `rbf_chain_8x8_k3.html`, `spectf_n14_k2.html`, `rbf_chain_8x8_k3_capped.html` | `examples/astar_landmark_walkthrough.py` |
| `bfs.html`, `dfs.html` | `examples/graph_search_vs_networkx.py` |
| `roach_g5.html` | `examples/two_hot_span_walkthrough.py --graph roach_g5` |
| `karate.html` | `… --graph karate --frame-every 15` |
| `roach_g20.html` | `… --graph roach_g20 --frame-every 150` |
| `roach_g5_diversity.html` | `… --graph roach_g5 --init random --collision-weight 10 --adjacency-form edge_product --adjacency-weight 0.3 --diversity-weight 10 --output-name roach_g5_diversity` |
| `roach_g20_diversity.html` | the same knobs with `--graph roach_g20 --frame-every 150 --output-name roach_g20_diversity` |
| `two_triangles.html` | `… --graph two_triangles --frame-every 25 --steps 1500` — spectral init, Ê = 1/15 on 2 components |
| `two_triangles_random_seed0.html` | `… --init random --seed 0 --output-name two_triangles_random_seed0` — Ê = 3.0833 on 3 components, the miss |
| `two_triangles_random_seed3.html` | `… --init random --seed 3 --output-name two_triangles_random_seed3` — Ê = 1/15 again, from the same random start at a different seed |

```
D=src/mllib/visualization/walkthroughs
uv run python examples/astar_landmark_walkthrough.py --output-dir $D
uv run python examples/graph_search_vs_networkx.py --output-dir $D
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph roach_g5
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph karate --frame-every 15
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph roach_g20 --frame-every 150
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph two_triangles --frame-every 25 --steps 1500
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph two_triangles --frame-every 25 --steps 1500 --init random --seed 0 --output-name two_triangles_random_seed0
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph two_triangles --frame-every 25 --steps 1500 --init random --seed 3 --output-name two_triangles_random_seed3
rm $D/*.json
```

Re-render whenever a view or an example changes, so the committed page of a run is the current one.
