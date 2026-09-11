# Rendered walkthroughs

One offline HTML page per recorded run, opened from `file://` with no server. Each page embeds its
own recording, so the `.json` beside it in the render folder is not kept here. Every sentence on a
page is derived from the recording at render time (D-34).

Rendered 2026-09-10 from this branch's code with the commands below, into this folder. Several runs
use a sparser full-frame cadence than the default (`--frame-every 5`) so that each page stays under
the 1 MB pre-commit limit; the light frame at every step, and so the loss curve, is unaffected.

**Open, for Xavier: the three `two_triangles` pages are 2.9 MB each and do not meet that limit.**
No `--frame-every` can bring them under it. A page costs about 600 bytes per *step*, because a
light frame is recorded at every step whatever the full-frame cadence, and that instance's schedule
is 5000 steps (`examples/two_hot_span_walkthrough.py`, `GRAPHS["two_triangles"]`); 1 MB is reached
at roughly 1600 steps. The two ways out are a light-frame cadence on `TwoHotSpanRecorder`, which it
does not have, and a shorter schedule, which would no longer be the run the side-by-side against
Xavier's own engine was done at. Until that is settled these three pages cannot be committed as
they stand.

| Page | Script and flags |
| --- | --- |
| `rbf_chain_8x8_k3.html`, `spectf_n14_k2.html`, `rbf_chain_8x8_k3_capped.html` | `examples/astar_landmark_walkthrough.py` |
| `bfs.html`, `dfs.html` | `examples/graph_search_vs_networkx.py` |
| `roach_g5.html` | `examples/two_hot_span_walkthrough.py --graph roach_g5` |
| `karate.html` | `… --graph karate --frame-every 15` |
| `roach_g20.html` | `… --graph roach_g20 --frame-every 150` |
| `roach_g5_diversity.html` | `… --graph roach_g5 --init random --collision-weight 10 --adjacency-form edge_product --adjacency-weight 0.3 --diversity-weight 10 --output-name roach_g5_diversity` |
| `roach_g20_diversity.html` | the same knobs with `--graph roach_g20 --frame-every 150 --output-name roach_g20_diversity` |
| `two_triangles.html` | `… --graph two_triangles --frame-every 25` — spectral init, Ê = 1/15 on 2 components |
| `two_triangles_random_seed0.html` | `… --init random --seed 0 --output-name two_triangles_random_seed0` — Ê = 3.0833 on 3 components, the miss |
| `two_triangles_random_seed3.html` | `… --init random --seed 3 --output-name two_triangles_random_seed3` — Ê = 1/15 again, from the same random start at a different seed |

```
D=src/mllib/visualization/walkthroughs
uv run python examples/astar_landmark_walkthrough.py --output-dir $D
uv run python examples/graph_search_vs_networkx.py --output-dir $D
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph roach_g5
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph karate --frame-every 15
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph roach_g20 --frame-every 150
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph two_triangles --frame-every 25
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph two_triangles --frame-every 25 --init random --seed 0 --output-name two_triangles_random_seed0
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph two_triangles --frame-every 25 --init random --seed 3 --output-name two_triangles_random_seed3
rm $D/*.json
```

Re-render whenever a view or an example changes, so the committed page of a run is the current one.
