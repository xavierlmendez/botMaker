# Examples

Composition roots: each script wires data to a model or an engine and writes its output. Three of them
also write a **walkthrough** — one offline HTML page holding a whole run, frame by frame, that opens
from `file://` with no server and no network.

```
uv run python examples/astar_landmark_walkthrough.py --output-dir src/mllib/visualization/walkthroughs
uv run python examples/graph_search_vs_networkx.py  --output-dir src/mllib/visualization/walkthroughs

# the two-hot optimizer needs the torch group (D-31), which the default sync does not install
uv sync --dev --group torch
uv run --group torch python examples/two_hot_span_walkthrough.py --output-dir src/mllib/visualization/walkthroughs

# re-render any saved recording; the view is chosen from the document's problem kind
uv run python -m mllib.visualization.render --recording run.json --output run.html
```

Each script writes a recording (`.json`) and a page (`.html`) per cell into `--output-dir`. The
rendered pages live in the repository at `src/mllib/visualization/walkthroughs/` (its README lists
the flags each page was rendered with); the recordings beside them are not kept, because a page
embeds its own. Re-render into that folder whenever a view or an example changes, so the page of a
cell is always the current one.

The markers under the slider are the run's **key moments**: the frames the view derived from the
recording as the ones that decided it — the frontier's peak, an incumbent improvement, the goal
expansion. Click one to jump to its frame. Every sentence on a page is derived from the recording when
the page is rendered, never typed by hand (D-34), so a re-recorded run re-narrates itself; underlined
terms carry their definition from `CONTEXT.md`.
