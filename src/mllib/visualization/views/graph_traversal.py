"""The view for a graph traversal: the graph's geometry, placed once, in page coordinates.

The recording already carries the layout the example computed — a seeded spring layout in the
unit-ish square networkx returns — so nothing here decides *where* the graph is shaped, only where
that shape lands on the page. The scaling is done in Python rather than in the browser for the
reason every view does it: a picture whose geometry is computed at render time is a picture that can
move between two openings of the same document, and the whole point of a recording is that it does
not.

The drawing itself is ``graph_traversal.js``, beside this module and read verbatim into the page.
It draws one picture per frame from four things the frame already holds: the node just visited, the
order so far, what is pending, and — on the last frame — whether the run found what it was looking
for.

Both algorithms share this view, as they share a recorder. A reader who opens the BFS page and the
DFS page of the same graph sees the same nodes in the same places, which is what makes the only real
difference between the two — the order they go in, and which end of the pending list they take from
— the only difference on screen.

The second half of the module is the page's prose: ``explain`` derives the opening, one sentence
per frame, the key moments, the legend and the ending from the recording and nothing else. One
absence shapes every sentence it writes. A frame holds what is pending, never *why* — no parent
pointer, no discovering node — so the narration counts the entries that were not pending a frame
ago and says only that they joined. "Node 4 discovered nodes 6 and 7" would be a guess dressed as a
fact, and on a graph with a cross-link it would frequently be the wrong guess. The recorder is
called at the pop, before the visited node's neighbours are pushed, so a frame's ``pending`` is the
container the visit was taken *out of* — whatever is new in it arrived during the *previous* visit,
and the narration credits nothing to the visit it is narrating.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from mllib.visualization.explain import (
    Explanation,
    LegendEntry,
    Moment,
    sort_moments,
    text_of,
    used_terms,
)
from mllib.visualization.recording import Recording
from mllib.visualization.views import View

VIEW_KIND = "graph_traversal"

# The drawing area, in user units; the page scales it to whatever width it is given.
CANVAS_WIDTH = 780
CANVAS_HEIGHT = 470

# Room around the graph for the banner above it and the node labels at its edges.
MARGIN_X = 48
MARGIN_TOP = 76
MARGIN_BOTTOM = 40

NODE_RADIUS = 15

ALGORITHM_NAMES = {"bfs": "Breadth-first search", "dfs": "Depth-first search"}

# The colours the drawing uses, repeated here because the legend has to show them beside a name.
# The drawing is the source of the pixels; this is the source of the words about them.
RULE_COLOUR = "#d8dce3"
UNTOUCHED_COLOUR = "#eef0f4"
VISITED_COLOUR = "#2f5ea8"
CURRENT_COLOUR = "#b4462f"
PENDING_COLOUR = "#a2761f"
FOUND_COLOUR = "#2f7a4f"

# Every drawn element that stands for something, keyed by the ``data-legend`` tag the view's
# JavaScript puts on it. The two lists are compared by a test, so an element that is drawn without
# an entry — or explained without being drawn — fails before the page is opened.
LEGEND: tuple[LegendEntry, ...] = (
    LegendEntry(
        key="edge",
        swatch=RULE_COLOUR,
        name="Edge",
        meaning="a pair of nodes the graph joins, which the traversal may or may not use",
    ),
    LegendEntry(
        key="crossed_edge",
        swatch=VISITED_COLOUR,
        name="Crossed edge",
        meaning="an edge with both ends already visited, so the walk's shape accumulates",
    ),
    LegendEntry(
        key="node_visited",
        swatch=VISITED_COLOUR,
        name="Visited node",
        meaning="a node already in the traversal order, with the visit it was taken at",
    ),
    LegendEntry(
        key="node_current",
        swatch=CURRENT_COLOUR,
        name="Current node",
        meaning="the node this frame is about, the one just taken out and visited",
    ),
    LegendEntry(
        key="node_pending",
        swatch=PENDING_COLOUR,
        name="Pending node",
        meaning="a node waiting in the queue or on the stack, not yet visited",
    ),
    LegendEntry(
        key="node_untouched",
        swatch=UNTOUCHED_COLOUR,
        name="Untouched node",
        meaning="a node the traversal has neither visited nor put in the container yet",
    ),
    LegendEntry(
        key="pending_strip",
        swatch=PENDING_COLOUR,
        name="Pending strip",
        meaning="the queue or stack in the container's own order, with each node's depth",
    ),
    LegendEntry(
        key="banner",
        swatch=FOUND_COLOUR,
        name="Banner",
        meaning="what this frame is: the visit it records, or why the traversal ended",
    ),
)


def _view_javascript() -> str:
    """The drawing function's source, read from the sibling file that ships as package data."""
    return files("mllib.visualization.views").joinpath("graph_traversal.js").read_text()


def _placed_nodes(layout: dict[str, Any]) -> list[dict[str, Any]]:
    """The recording's positions scaled into the canvas box, one entry per node id.

    The two axes are scaled independently so the graph fills the box it is given; a spring layout
    carries no distance a reader is meant to measure, only which nodes are near which, and that
    survives an anisotropic scale. The y axis is flipped because SVG counts downwards and networkx
    does not.
    """
    node_ids = [int(node_id) for node_id in layout["node_ids"]]
    positions = [(float(x), float(y)) for x, y in layout["positions"]]
    if not positions:
        return []

    xs = [x for x, _ in positions]
    ys = [y for _, y in positions]
    x_span = max(xs) - min(xs) or 1.0
    y_span = max(ys) - min(ys) or 1.0
    width = CANVAS_WIDTH - 2 * MARGIN_X
    height = CANVAS_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    return [
        {
            "id": node_id,
            "x": MARGIN_X + (x - min(xs)) / x_span * width,
            "y": MARGIN_TOP + (1.0 - (y - min(ys)) / y_span) * height,
        }
        for node_id, (x, y) in zip(node_ids, positions, strict=True)
    ]


def _container(algorithm: str) -> str:
    """What the pending nodes are held in, which is the only thing the two algorithms disagree on."""
    return "stack" if algorithm == "dfs" else "queue"


def _pending_ids(frame: dict[str, Any]) -> list[int]:
    """The nodes waiting on this frame, in the container's own order."""
    return [int(entry[0]) for entry in frame.get("pending") or []]


def _joined(frame: dict[str, Any], previous: dict[str, Any] | None) -> list[int]:
    """The nodes pending now that were not pending a frame ago, sorted.

    These arrived during the *previous* visit: the recorder snapshots the container at the pop,
    before this visit's neighbours are pushed. The frames carry no parent pointer either, so the
    most the sentence built from this list can say is that they joined between the two frames —
    which is what it says.
    """
    before = set(_pending_ids(previous)) if previous is not None else set()
    return sorted(node_id for node_id in _pending_ids(frame) if node_id not in before)


def _order_text(order: list[int]) -> str:
    """A traversal order as prose reads it: the ids in the order they were visited."""
    return ", ".join(str(int(node_id)) for node_id in order)


def _visit_count(recording: Recording) -> int:
    """How many nodes the run visited, from the result when it says and from the frames when not."""
    stated = recording.result.get("visit_count")
    if stated is not None:
        return int(stated)
    return len(recording.result.get("traversal_order") or [])


def _target_visit(recording: Recording) -> int | None:
    """Which visit the target was, counting from one, or ``None`` if it was never visited."""
    target = recording.problem.get("target_node_id")
    order = [int(node_id) for node_id in recording.result.get("traversal_order") or []]
    if target is None or int(target) not in order:
        return None
    return order.index(int(target)) + 1


def _opening(recording: Recording) -> tuple[str, ...]:
    """What the page is, in four lines: the instance, the algorithm, the question, how to read it."""
    problem = recording.problem
    container = _container(str(problem.get("algorithm", "bfs")))
    if container == "stack":
        algorithm = (
            "Algorithm: depth-first search — the next node visited is the one most recently "
            "pushed on the stack, so the traversal runs deep before it runs wide."
        )
    else:
        algorithm = (
            "Algorithm: breadth-first search — the next node visited is the one that has waited "
            "longest in the queue, so nodes are visited in order of depth."
        )

    return (
        f"Instance: a graph of {problem.get('node_count')} nodes; the traversal starts at node "
        f"{recording.configuration.get('start')} and looks for node "
        f"{problem.get('target_node_id')}.",
        algorithm,
        f"Question: is node {problem.get('target_node_id')} reached, and which nodes are visited "
        "before it?",
        "How to read: one frame per visit. Filled nodes are visited, the ringed node is the "
        "current visit, outlined nodes are pending; the strip below the graph is the "
        f"{container} with the next node first.",
    )


def _narrate_visit(
    frame: dict[str, Any], previous: dict[str, Any] | None, container: str, target: int | None
) -> str:
    """The sentence for one visit: which node left the container, what joined it, what waits."""
    visit = len(frame["traversal_order"])
    pending = _pending_ids(frame)
    left = f"Visit {visit}: node {frame['node_id']} at depth {frame['depth']} left the {container}"

    # The first visit is taken out of a container holding only the start node, and the frame is
    # written before its neighbours are pushed: the empty pending list is the state of the run,
    # not a traversal that found nothing.
    if not pending:
        return f"{left}, which is empty until its neighbours are pushed."

    joined = _joined(frame, previous)
    if not joined:
        arrivals = "nothing joined it since the previous visit"
    else:
        noun = "node" if len(joined) == 1 else "nodes"
        arrivals = f"since the previous visit {noun} {_order_text(joined)} joined it"

    verb = "is" if len(pending) == 1 else "are"
    noun = "node" if len(pending) == 1 else "nodes"
    waiting = f"{len(pending)} {noun} {verb} pending ({_order_text(pending)})"
    tail = f"; node {target} is among them" if target is not None and int(target) in pending else ""

    return f"{left}; {arrivals}, and {waiting}{tail}."


def _narrate_end(recording: Recording, frame: dict[str, Any]) -> str:
    """The last frame's sentence: how long the walk was, whether it stopped on what it wanted."""
    visits = len(frame["traversal_order"])
    target_visit = _target_visit(recording)
    outcome = (
        f"target found at visit {target_visit}"
        if frame.get("found") and target_visit is not None
        else "target not found"
    )
    return (
        f"Traversal ended after {visits} visit{'' if visits == 1 else 's'}: {outcome}; "
        f"order {_order_text(frame['traversal_order'])}."
    )


def _narration(recording: Recording) -> tuple[str, ...]:
    """One sentence per frame, each read from that frame and the one before it."""
    container = _container(str(recording.problem.get("algorithm", "bfs")))
    target = recording.problem.get("target_node_id")
    lines: list[str] = []
    for index, frame in enumerate(recording.frames):
        if frame.get("end"):
            lines.append(_narrate_end(recording, frame))
        else:
            lines.append(
                _narrate_visit(
                    frame, recording.frames[index - 1] if index else None, container, target
                )
            )
    return tuple(lines)


def _moments(recording: Recording) -> tuple[Moment, ...]:
    """The frames that decide the run, in the order a frame claimed by two of them is resolved."""
    frames = recording.frames
    container = _container(str(recording.problem.get("algorithm", "bfs")))
    target = recording.problem.get("target_node_id")
    candidates: list[Moment] = []

    if target is not None:
        visited_at = next(
            (index for index, frame in enumerate(frames) if frame.get("node_id") == target), None
        )
        if visited_at is not None:
            candidates.append(
                Moment(
                    frame_index=visited_at,
                    label="target visited",
                    reason="The target is visited here and the traversal stops.",
                )
            )
        pending_at = next(
            (index for index, frame in enumerate(frames) if target in _pending_ids(frame)), None
        )
        if pending_at is not None:
            candidates.append(
                Moment(
                    frame_index=pending_at,
                    label="target pending",
                    reason=f"Node {target} is in the {container} from here; every node ahead of "
                    "it is visited first.",
                )
            )

    sizes = [len(_pending_ids(frame)) for frame in frames]
    if sizes and max(sizes) > 0:
        peak = sizes.index(max(sizes))
        candidates.append(
            Moment(
                frame_index=peak,
                label="peak",
                reason=f"The {container} was never longer: {sizes[peak]} nodes waited.",
            )
        )
    return sort_moments(candidates)


def _ending(recording: Recording) -> tuple[str, ...]:
    """Why the run stopped, and the walk it stopped after."""
    frames = recording.frames
    container = _container(str(recording.problem.get("algorithm", "bfs")))
    target = recording.problem.get("target_node_id")
    visits = _visit_count(recording)
    found = bool(frames[-1].get("found")) if frames else False

    if found:
        stopped = f"Stopped because node {target} was visited, after {visits} visits."
    else:
        stopped = (
            f"Stopped because the {container} emptied without visiting node {target}, after "
            f"{visits} visits."
        )
    order = recording.result.get("traversal_order") or []
    return (stopped, f"Visit order: {_order_text(order)}.")


def explain(recording: Recording) -> Explanation:
    """Everything the page says in words about this run, derived from the run and nothing else."""
    opening = _opening(recording)
    narration = _narration(recording)
    ending = _ending(recording)
    return Explanation(
        opening=opening,
        narration=narration,
        moments=_moments(recording),
        legend=LEGEND,
        # A traversal has no per-frame number a reader tracks: the depth and the pending count are
        # in every sentence already, and a strip repeating them would be a row of the narration
        # with the words removed. The page hides the strip when this is empty.
        quantities=(),
        ending=ending,
        glossary=used_terms(text_of((opening, narration, ending, LEGEND))),
    )


def graph_traversal_view(recording: Recording) -> View:
    """The graph-traversal view: its drawing function and the placed graph that function reads."""
    problem = recording.problem
    layout_data = problem["layout"]
    algorithm = str(problem.get("algorithm", "bfs"))
    target = problem.get("target_node_id")
    subtitle = (
        f"{ALGORITHM_NAMES.get(algorithm, algorithm)} over {problem.get('node_count', '?')} nodes"
        + ("" if target is None else f", looking for node {target}")
    )

    layout: dict[str, Any] = {
        "width": CANVAS_WIDTH,
        "height": CANVAS_HEIGHT,
        "subtitle": subtitle,
        "algorithm": algorithm,
        "target_node_id": target,
        "node_radius": NODE_RADIUS,
        "nodes": _placed_nodes(layout_data),
        "edges": [[int(u), int(v)] for u, v in layout_data["edges"]],
        "banner": {"x": 16, "y": 30, "width": CANVAS_WIDTH - 32, "height": 30},
        "legend": {"x": 16, "y": CANVAS_HEIGHT - 14},
        "explain": explain(recording).to_dict(),
    }

    return View(
        kind=VIEW_KIND,
        title=f"{ALGORITHM_NAMES.get(algorithm, algorithm)} walkthrough",
        javascript=_view_javascript(),
        layout=layout,
    )
