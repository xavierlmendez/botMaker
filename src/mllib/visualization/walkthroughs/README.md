# Rendered walkthroughs

One offline HTML page per recorded run, opened from `file://` with no server. Each page embeds its
own recording, so the `.json` beside it in the render folder is not kept here. Every sentence on a
page is derived from the recording at render time (D-34).

Rendered 2026-09-10 from this branch's code with the commands below, into this folder. Three runs
use a sparser full-frame cadence than the default (`--frame-every 5`) so that each page stays under
the 1 MB pre-commit limit; the light frame at every step, and so the loss curve, is unaffected.

| Page | Script and flags |
| --- | --- |
| `rbf_chain_8x8_k3.html`, `spectf_n14_k2.html`, `rbf_chain_8x8_k3_capped.html` | `examples/astar_landmark_walkthrough.py` |
| `bfs.html`, `dfs.html` | `examples/graph_search_vs_networkx.py` |
| `roach_g5.html` | `examples/two_hot_span_walkthrough.py --graph roach_g5` |
| `karate.html` | `… --graph karate --frame-every 15` |
| `roach_g20.html` | `… --graph roach_g20 --frame-every 150` |
| `roach_g5_diversity.html` | `… --graph roach_g5 --init random --collision-weight 10 --adjacency-form edge_product --adjacency-weight 0.3 --diversity-weight 10 --output-name roach_g5_diversity` |
| `roach_g20_diversity.html` | the same knobs with `--graph roach_g20 --frame-every 150 --output-name roach_g20_diversity` |

```
D=src/mllib/visualization/walkthroughs
uv run python examples/astar_landmark_walkthrough.py --output-dir $D
uv run python examples/graph_search_vs_networkx.py --output-dir $D
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph roach_g5
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph karate --frame-every 15
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir $D --graph roach_g20 --frame-every 150
rm $D/*.json
```

Re-render whenever a view or an example changes, so the committed page of a run is the current one.
