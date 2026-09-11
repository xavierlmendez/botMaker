"""The view for a two-hot span run: where the graph is drawn, and the scales its panels share.

The drawing itself is ``two_hot_span.js``, beside this module and read verbatim into the page. What
is computed here is everything the JavaScript would otherwise have to derive by walking a whole
recording in a browser: the node positions in the coordinates the panel actually uses, and the four
run-wide ranges the pictures are scaled against.

The positions come from the recording, not from here. Where a node belongs is a property of the
instance — a ladder for the roach, a seeded spring layout for anything else — and the example that
built the instance is the only place that knows which. This module only maps those coordinates into
the panel's box, each axis independently: a roach ladder scaled to preserve its aspect ratio would
be a two-pixel-tall smear across the page, and what a reader needs to see is which side of the
ladder a node ended up on.

Every scale is the run's rather than the frame's, for the reason the A* view fixes its bar scale:
a heatmap normalised to its own frame's largest entry looks identical on the first step and the
last, and the one thing V does over a run is grow. So ``value_max`` is the largest ``|V_ij|``
anywhere in the run, and a cell that darkens between frames means the entry grew.

The second half of the module is the page's prose: ``explain`` derives the opening, one sentence
per frame, the key moments, the legend, the per-frame quantities and the ending from the recording
and nothing else. It is here rather than in the JavaScript because the recording is a Python object
at render time and a JSON blob in a browser, and a sentence composed in the browser is a sentence
no test can read.
"""

from __future__ import annotations

from importlib.resources import files
from itertools import pairwise
from typing import Any

import numpy as np

from mllib.visualization.explain import (
    MINUS_SIGN,
    Explanation,
    LegendEntry,
    Moment,
    Quantity,
    fmt_num,
    sort_moments,
    text_of,
    used_terms,
)
from mllib.visualization.recording import Recording
from mllib.visualization.views import View

VIEW_KIND = "two_hot_span"

# The drawing area, in user units; the page scales it to whatever width it is given.
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 560

# The pair graph's box, in the same user units. Positions are mapped into it here, inset by a
# margin wide enough that a node circle drawn on the extreme coordinate still sits inside the box.
GRAPH_BOX = {"x": 16, "y": 96, "width": 452, "height": 236}
GRAPH_MARGIN = 18.0

# How much of a column's squared norm its two largest coordinates must hold before the page calls
# that column 2-hot — equivalently, a roundability of at most 0.05. A threshold rather than an
# exact test because a gradient run never reaches exactly two non-zero coordinates: it approaches
# a vertex pair, and this is the line past which a reader would call the approach arrived.
TWO_HOT_SHARE = 0.95

# The colours the drawing uses, repeated here because the legend has to show them beside a name.
# The drawing is the source of the pixels; this is the source of the words about them.
INK_COLOUR = "#16181d"
RULE_COLOUR = "#d8dce3"
LOSS_COLOUR = "#2f5ea8"
CUT_COLOUR = "#2f7a4f"
FLOOR_COLOUR = "#a2761f"
POSITIVE_COLOUR = "#b4462f"

# Every drawn element that stands for a quantity, keyed by the ``data-legend`` tag the view's
# JavaScript puts on it. The two lists are compared by a test, so an element that is drawn without
# an entry — or explained without being drawn — fails before the page is opened.
LEGEND: tuple[LegendEntry, ...] = (
    LegendEntry(
        key="graph_edge",
        swatch=RULE_COLOUR,
        name="Graph edge",
        meaning="an edge of the instance itself, drawn faint as the background the pairs are read "
        "against",
    ),
    LegendEntry(
        key="pair_edge",
        swatch=INK_COLOUR,
        name="Rounded pair on an edge",
        meaning="a rounded pair whose two vertices are joined by an edge of the graph, drawn solid",
    ),
    LegendEntry(
        key="pair_chord",
        swatch=INK_COLOUR,
        name="Rounded pair off the graph",
        meaning="a rounded pair whose two vertices share no edge, drawn dashed because the "
        "relaxation is free to pair any two vertices",
    ),
    LegendEntry(
        key="vertex",
        swatch=LOSS_COLOUR,
        name="Vertex",
        meaning="a vertex of the instance, filled by the component the rounded pairs put it in",
    ),
    LegendEntry(
        key="heat_cell",
        swatch=POSITIVE_COLOUR,
        name="Entry of V",
        meaning="one entry of the spanning set, red positive and blue negative; a column with two "
        "dark cells and nothing else has a low roundability",
    ),
    LegendEntry(
        key="loss_curve",
        swatch=LOSS_COLOUR,
        name="Training loss",
        meaning="the ridge training loss at every step, the quantity the optimizer descends",
    ),
    LegendEntry(
        key="cut_curve",
        swatch=CUT_COLOUR,
        name="Rounded cut Ê",
        meaning="the rounded cut at each full frame, the value the relaxation actually delivers",
    ),
    LegendEntry(
        key="floor_line",
        swatch=FLOOR_COLOUR,
        name="Spectral floor Σλ",
        meaning="the sum of the K smallest Laplacian eigenvalues, which no cut can go below",
    ),
    LegendEntry(
        key="end_banner",
        swatch=CUT_COLOUR,
        name="End banner",
        meaning="the run's last frame, carrying the numbers the run itself reported",
    ),
)


def _view_javascript() -> str:
    """The drawing function's source, read from the sibling file that ships as package data."""
    return files("mllib.visualization.views").joinpath("two_hot_span.js").read_text()


def _full_frames(recording: Recording) -> list[dict[str, Any]]:
    """The frames carrying the derived numbers; a light frame has ``None`` in every one of them."""
    return [frame for frame in recording.frames if frame.get("full")]


def _scaled_positions(positions: list[list[float]]) -> list[list[float]]:
    """The recording's node coordinates mapped into the graph box, y flipped for SVG.

    Each axis is scaled on its own extent. An axis with no extent — every node on one line, which
    is what a one-column layout or a single node gives — is centred rather than divided by zero.
    """
    if not positions:
        return []
    box_left = GRAPH_BOX["x"] + GRAPH_MARGIN
    box_top = GRAPH_BOX["y"] + GRAPH_MARGIN
    span_x = GRAPH_BOX["width"] - 2 * GRAPH_MARGIN
    span_y = GRAPH_BOX["height"] - 2 * GRAPH_MARGIN

    xs = [float(position[0]) for position in positions]
    ys = [float(position[1]) for position in positions]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_extent = x_max - x_min
    y_extent = y_max - y_min

    scaled = []
    for x, y in zip(xs, ys, strict=True):
        across = 0.5 if x_extent == 0 else (x - x_min) / x_extent
        up = 0.5 if y_extent == 0 else (y - y_min) / y_extent
        # SVG's y grows downward, so a node high in the layout is drawn near the top of the box.
        scaled.append([box_left + across * span_x, box_top + (1.0 - up) * span_y])
    return scaled


def _two_hot_columns(spanning_set: Any) -> tuple[int, int, list[tuple[int, int]]]:
    """How many of V's columns are 2-hot, out of how many, and the vertex pair each one names.

    The test is on squared coordinates because the objective is: a column's energy is what the
    projector sees, so "two coordinates hold this column" means those two hold the energy. A zero
    column holds nothing and is not 2-hot — it is a column the run has switched off, and rounding
    it would invent a pair the optimizer never chose.
    """
    matrix = np.asarray(spanning_set, dtype=float)
    if matrix.size == 0:
        return 0, int(matrix.shape[1]) if matrix.ndim == 2 else 0, []

    squared = matrix * matrix
    norms = squared.sum(axis=0)
    # Stable, so two coordinates of equal magnitude are broken by row index and the same recording
    # always names the same pair.
    order = np.argsort(-np.abs(matrix), axis=0, kind="stable")
    columns = np.arange(matrix.shape[1])
    largest, second = order[0], order[1]
    held = squared[largest, columns] + squared[second, columns]
    two_hot = (norms > 0.0) & (held >= TWO_HOT_SHARE * norms)

    pairs = [
        (int(min(largest[column], second[column])), int(max(largest[column], second[column])))
        for column in columns
        if two_hot[column]
    ]
    return int(two_hot.sum()), int(matrix.shape[1]), pairs


def _pair_summary(frame: dict[str, Any], edges: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    """A full frame's 2-hot count, its column count, its distinct pairs and how many are edges."""
    count, columns, pairs = _two_hot_columns(frame["spanning_set"])
    distinct = sorted(set(pairs))
    return count, columns, len(distinct), sum(1 for pair in distinct if pair in edges)


def _edge_set(recording: Recording) -> set[tuple[int, int]]:
    """The instance's edges as sorted index pairs, the form a rounded pair is compared against."""
    embedded = recording.problem.get("layout") or {}
    return {
        (int(min(first, second)), int(max(first, second)))
        for first, second in embedded.get("edges", [])
    }


def _mean_collision(frame: dict[str, Any]) -> float:
    """The mean R(v_j) over the frame's columns; a diagnostic, never a criterion (`CONTEXT.md`)."""
    measures = frame.get("collision_measures") or []
    return float(np.mean(measures)) if len(measures) else 0.0


def _frame_every(recording: Recording) -> int:
    """How many steps apart the full frames were asked for, read back off the frames themselves.

    The cadence is a recorder argument and the recording does not carry it, so it is recovered
    from the steps the full frames landed on. The last step is always full whatever the cadence
    divides, which makes the final gap a short one; the commonest gap is the cadence.
    """
    steps = [int(frame["step"]) for frame in recording.frames if frame.get("full")]
    gaps = [later - earlier for earlier, later in pairwise(steps) if later > earlier]
    if not gaps:
        return 1
    return max(sorted(set(gaps)), key=gaps.count)


def _full_series(recording: Recording) -> list[tuple[int, dict[str, Any]]]:
    """The full frames the run's picture actually advances on, paired with their frame index.

    The end frame is left out. It restates the last step's numbers rather than taking a step, so
    counting it would report a change of zero at a step that already had one.
    """
    return [
        (index, frame)
        for index, frame in enumerate(recording.frames)
        if frame.get("full") and not frame.get("end")
    ]


def _opening(recording: Recording) -> tuple[str, ...]:
    """What the page is, in four lines: the instance, the algorithm, the question, how to read it."""
    problem = recording.problem
    return (
        f"Instance: {problem.get('graph')}, n = {problem.get('n')} vertices, "
        f"K = {problem.get('cluster_count')} clusters asked, "
        f"λ = {problem.get('collision_weight')} on the collision measure, "
        f"{problem.get('init')} initialisation.",
        "Algorithm: gradient steps on a spanning set V with n rows and r columns, lowering the "
        "relaxed objective E* while the collision measure R(v_j) rewards each column for becoming "
        "2-hot: two non-zero coordinates of opposite sign, a vertex pair. At every full frame the "
        "columns are rounded to their pairs and the rounded cut Ê is measured.",
        "Question: do the columns become 2-hot, and do their pairs give a cut with K components "
        "whose Ê is close to E* and to the spectral floor Σλ?",
        f"How to read: one frame per step; every {_frame_every(recording)}th step is a full frame "
        "with V, its pairs and Ê, the steps between carry only the training loss and keep the last "
        "full picture. Left, the graph with rounded pairs drawn bold; right, V as a heatmap, red "
        "positive, blue negative; below, the training loss every step and Ê at full frames against "
        "Σλ.",
    )


def _movement(now: float, before: float, fell: str = "fell", rose: str = "rose") -> str:
    """How a number moved since the last time the page showed one, as a reader sees the two.

    Compared at display precision, not as doubles: two numbers a reader reads as the same number
    did not change on this page, and "fell by 0.0000" is a sentence that makes every other one
    suspect.
    """
    if fmt_num(now) == fmt_num(before):
        return "unchanged"
    verb = fell if now < before else rose
    return f"{verb} by {fmt_num(abs(now - before))}"


def _narrate_full(
    frame: dict[str, Any],
    previous: dict[str, Any] | None,
    previous_count: int | None,
    cluster_count: int,
    spectral_floor: float,
) -> str:
    """A full frame: where the two objectives stand, what the rounding gives, how 2-hot V is."""
    count, columns, _ = _two_hot_columns(frame["spanning_set"])
    step = int(frame["step"])
    relaxed = float(frame["relaxed_objective"])
    cut = float(frame["rounded_cut"])
    components = int(frame["component_count"])
    mean = fmt_num(_mean_collision(frame))

    if previous is None:
        return (
            f"Step {step}: E* {fmt_num(relaxed)} against a floor of {fmt_num(spectral_floor)}, "
            f"Ê {fmt_num(cut)}, {components} components of K = {cluster_count}; {count} of "
            f"{columns} columns are 2-hot; mean R(v_j) {mean}."
        )

    was_step = int(previous["step"])
    if components == cluster_count:
        reached = f"reached K = {cluster_count}"
    elif components < cluster_count:
        reached = f"still short of K = {cluster_count}"
    else:
        reached = f"more than K = {cluster_count}"

    gained = count - int(previous_count or 0)
    if gained == 0:
        two_hot = f"unchanged since step {was_step}"
    else:
        sign = "+" if gained > 0 else MINUS_SIGN
        two_hot = f"{sign}{abs(gained)} since step {was_step}"

    return (
        f"Step {step}: E* {fmt_num(relaxed)} "
        f"({_movement(relaxed, float(previous['relaxed_objective']))} since step {was_step}), "
        f"Ê {fmt_num(cut)} ({_movement(cut, float(previous['rounded_cut']))}), "
        f"{components} components ({reached}); {count} of {columns} columns are 2-hot "
        f"({two_hot}); mean R(v_j) {mean}."
    )


def _narrate_end(recording: Recording, frame: dict[str, Any], edges: set[tuple[int, int]]) -> str:
    """The last frame: the run's own reported numbers, and what the columns rounded to."""
    count, columns, distinct, on_edges = _pair_summary(frame, edges)
    steps = len(recording.result.get("loss_history") or [])
    plural = "" if on_edges == 1 else "s"
    return (
        f"Run ended after {steps} steps: E* {fmt_num(frame['relaxed_objective'])}, "
        f"Ê {fmt_num(frame['rounded_cut'])}, Ê {MINUS_SIGN} Σλ "
        f"{fmt_num(frame['rounded_cut_minus_floor'])}, {frame['component_count']} components; "
        f"{count} of {columns} columns 2-hot on {distinct} distinct pairs, {on_edges} of them "
        f"edge{plural} of the graph."
    )


def _narration(recording: Recording) -> tuple[str, ...]:
    """One sentence per frame: a full frame against the last full one, a light frame against its own
    predecessor, because a light frame's only new number is its training loss."""
    problem = recording.problem
    cluster_count = int(problem.get("cluster_count", 0))
    spectral_floor = float(problem.get("spectral_floor", 0.0))
    edges = _edge_set(recording)

    lines: list[str] = []
    previous_full: dict[str, Any] | None = None
    previous_count: int | None = None
    for index, frame in enumerate(recording.frames):
        if frame.get("end"):
            lines.append(_narrate_end(recording, frame, edges))
            continue
        if frame.get("full"):
            lines.append(
                _narrate_full(frame, previous_full, previous_count, cluster_count, spectral_floor)
            )
            previous_count = _two_hot_columns(frame["spanning_set"])[0]
            previous_full = frame
            continue

        previous = recording.frames[index - 1]
        loss = float(frame["training_loss"])
        lines.append(
            f"Step {frame['step']}: training loss {fmt_num(loss)} "
            f"({_movement(loss, float(previous['training_loss']), 'down', 'up')} since step "
            f"{previous['step']}); the picture is from step "
            f"{previous_full['step'] if previous_full else frame['step']}."
        )
    return tuple(lines)


def _moments(recording: Recording) -> tuple[Moment, ...]:
    """The frames that decide the run, listed in the order a frame claimed by two is resolved."""
    cluster_count = int(recording.problem.get("cluster_count", 0))
    series = _full_series(recording)
    counted = [(index, frame, _two_hot_columns(frame["spanning_set"])) for index, frame in series]
    candidates: list[Moment] = []

    for index, frame, _ in counted:
        if int(frame["component_count"]) == cluster_count:
            candidates.append(
                Moment(
                    frame_index=index,
                    label="K reached",
                    reason="The rounded pairs first give exactly K components.",
                )
            )
            break

    for index, _frame, (count, columns, _pairs) in counted:
        if columns and count == columns:
            candidates.append(
                Moment(
                    frame_index=index,
                    label="all 2-hot",
                    reason="Every column is now a vertex pair; from here the run only re-arranges "
                    "pairs.",
                )
            )
            break

    for index, _frame, (count, _columns, _pairs) in counted:
        if count:
            candidates.append(
                Moment(
                    frame_index=index,
                    label="first 2-hot",
                    reason="A column concentrated on a vertex pair for the first time.",
                )
            )
            break

    drops = [
        (float(before["rounded_cut"]) - float(frame["rounded_cut"]), index, before, frame)
        for (_, before), (index, frame) in pairwise(series)
    ]
    if drops:
        fall, index, before, frame = max(drops, key=lambda entry: entry[0])
        if fall > 0.0:
            candidates.append(
                Moment(
                    frame_index=index,
                    label="Ê drop",
                    reason=f"Ê's largest single fall: {fmt_num(before['rounded_cut'])} to "
                    f"{fmt_num(frame['rounded_cut'])}.",
                )
            )

    changed = [
        index
        for (_, before), (index, frame) in pairwise(series)
        if fmt_num(before["rounded_cut"]) != fmt_num(frame["rounded_cut"])
    ]
    if changed and changed[-1] != series[-1][0]:
        candidates.append(
            Moment(
                frame_index=changed[-1],
                label="Ê settled",
                reason="Ê did not change after this frame.",
            )
        )
    return sort_moments(candidates)


def _quantities(recording: Recording) -> tuple[Quantity, ...]:
    """The seven numbers the page reads out for whichever frame is showing.

    Dense, one entry per frame, and ``None`` wherever a light frame simply has no such number: a
    page that carried the last full frame's Ê on a light frame would be reporting a measurement
    the recorder did not take.
    """
    relaxed: list[str | None] = []
    cuts: list[str | None] = []
    above_floor: list[str | None] = []
    components: list[int | None] = []
    two_hot: list[str | None] = []
    collision: list[str | None] = []
    losses: list[str | None] = []

    for frame in recording.frames:
        losses.append(fmt_num(float(frame["training_loss"])))
        if not frame.get("full"):
            relaxed.append(None)
            cuts.append(None)
            above_floor.append(None)
            components.append(None)
            two_hot.append(None)
            collision.append(None)
            continue
        count, columns, _ = _two_hot_columns(frame["spanning_set"])
        relaxed.append(fmt_num(float(frame["relaxed_objective"])))
        cuts.append(fmt_num(float(frame["rounded_cut"])))
        above_floor.append(fmt_num(float(frame["rounded_cut_minus_floor"])))
        components.append(int(frame["component_count"]))
        two_hot.append(f"{count} of {columns}")
        collision.append(fmt_num(_mean_collision(frame)))

    return (
        Quantity(
            key="relaxed_objective",
            term="Relaxed objective",
            definition="E*, the ratio-cut value at the unrounded V through the exact projector, never "
            "the training loss",
            values=tuple(relaxed),
        ),
        Quantity(
            key="rounded_cut",
            term="Rounded cut",
            definition="Ê, the ratio-cut value of the partition the rounded pairs induce",
            values=tuple(cuts),
        ),
        Quantity(
            key="cut_minus_floor",
            term="Spectral floor",
            definition=f"Ê {MINUS_SIGN} Σλ, how far the rounded cut sits above the spectral floor",
            values=tuple(above_floor),
        ),
        Quantity(
            key="component_count",
            term="Component count",
            definition="how many connected components the rounded pairs induce, against the K "
            "clusters asked for",
            values=tuple(components),
        ),
        Quantity(
            key="two_hot_columns",
            term="2-hot vector",
            definition="columns whose two largest squared coordinates hold 95 % of the norm",
            values=tuple(two_hot),
        ),
        Quantity(
            key="mean_collision",
            term="Collision measure",
            definition="the mean R(v_j) over the columns, a diagnostic of how 2-hot they have "
            "become",
            values=tuple(collision),
        ),
        Quantity(
            key="training_loss",
            term="Training loss",
            definition="the ridge criterion minus λ times the summed collision measure, the "
            "quantity the steps descend",
            values=tuple(losses),
        ),
    )


def _fmt_drift(value: float) -> str:
    """A drift, which is honest only if it is not rounded to the zero it very nearly is.

    Four decimals is the page's precision for a quantity a reader compares; a zero-sum violation
    is a quantity a reader checks the *size* of, and one printed as 0.0000 is indistinguishable
    from a violation that never happened.
    """
    if value != 0.0 and abs(value) < 10.0**-4:
        return f"{value:.2e}"
    return fmt_num(value)


def _ending(recording: Recording) -> tuple[str, ...]:
    """Why the run stopped, what it ended holding, and what the page cannot tell the reader."""
    problem = recording.problem
    result = recording.result
    frames = recording.frames
    edges = _edge_set(recording)
    last = frames[-1]
    steps = len(result.get("loss_history") or [])
    count, columns, distinct, _on_edges = _pair_summary(last, edges)

    series = _full_series(recording)
    changed = [
        (index, frame)
        for (_, before), (index, frame) in pairwise(series)
        if fmt_num(before["rounded_cut"]) != fmt_num(frame["rounded_cut"])
    ]
    if not changed:
        settled = f"Ê never changed after the first full frame at step {series[0][1]['step']}."
    else:
        last_change = int(changed[-1][1]["step"])
        if changed[-1][0] == series[-1][0]:
            settled = f"Ê last changed at step {last_change}, on the last full frame."
        else:
            held = max(steps - 1 - last_change, 0)
            settled = (
                f"Ê last changed at step {last_change}, then held for the remaining {held} steps."
            )

    return (
        f"Stopped because the step budget of {steps} steps was spent; the optimizer never stops "
        "early.",
        f"Final: E* {fmt_num(result.get('relaxed_objective'))}, "
        f"Ê {fmt_num(result.get('rounded_cut'))}, Ê {MINUS_SIGN} Σλ "
        f"{fmt_num(result.get('rounded_cut_minus_floor'))}, {result.get('component_count')} "
        f"components against K = {problem.get('cluster_count')} asked; {count} of {columns} "
        f"columns 2-hot on {distinct} distinct pairs.",
        settled,
        f"Largest zero-sum violation over the run "
        f"{_fmt_drift(float(result.get('max_zero_sum_violation', 0.0)))} (drift; the recording "
        "holds only this maximum).",
        "The datum this run is judged against is not in the recording.",
    )


def explain(recording: Recording) -> Explanation:
    """Everything the page says in words about this run, derived from the run and nothing else."""
    opening = _opening(recording)
    narration = _narration(recording)
    moments = _moments(recording)
    quantities = _quantities(recording)
    ending = _ending(recording)
    return Explanation(
        opening=opening,
        narration=narration,
        moments=moments,
        legend=LEGEND,
        quantities=quantities,
        ending=ending,
        glossary=used_terms(text_of((opening, narration, ending, LEGEND, quantities))),
    )


def two_hot_span_view(recording: Recording) -> View:
    """The two-hot span view: its drawing function and the layout that function reads."""
    problem = recording.problem
    graph_name = str(problem.get("graph", "a graph"))
    node_total = int(problem.get("n", 0))
    collision_weight = problem.get("collision_weight", "?")
    init = str(problem.get("init", "?"))
    spectral_floor = float(problem.get("spectral_floor", 0.0))

    embedded = problem.get("layout") or {}
    positions = [list(position) for position in embedded.get("positions", [])]
    edges = [[int(first), int(second)] for first, second in embedded.get("edges", [])]

    full = _full_frames(recording)
    # The end frame restates the loss of the step it follows, so it is not a point of the loss
    # series: counting it would let the same number decide the range twice and, in the drawing,
    # put a second point on top of the last one.
    losses = [float(frame["training_loss"]) for frame in recording.frames if not frame.get("end")]
    cuts = [float(frame["rounded_cut"]) for frame in full]
    entries = [
        abs(float(value)) for frame in full for row in frame["spanning_set"] for value in row
    ]
    column_count = max((len(row) for frame in full for row in frame["spanning_set"]), default=0)

    subtitle = (
        f"{graph_name} — {node_total} nodes, K = {problem.get('cluster_count', '?')}, "
        f"λ = {collision_weight}, {init} init"
    )
    # The experiment knobs of slices 2.4 and 2.5 reach `problem` only when they are switched on, so
    # naming them here costs a default recording nothing and tells an experiment's page what it is.
    experiments = []
    if "adjacency_weight" in problem:
        experiments.append(f"μ = {problem['adjacency_weight']} ({problem.get('adjacency_form')})")
    if "diversity_weight" in problem:
        experiments.append(f"ν = {problem['diversity_weight']}")
    if "learning_rate_schedule" in problem:
        experiments.append(str(problem["learning_rate_schedule"]))
    if experiments:
        subtitle += " · " + ", ".join(experiments)

    layout: dict[str, Any] = {
        "width": CANVAS_WIDTH,
        "height": CANVAS_HEIGHT,
        "subtitle": subtitle,
        "positions": _scaled_positions(positions),
        "edges": edges,
        "node_count": len(positions),
        "column_count": column_count,
        # The run-wide scales. Every one of them spans the whole run so that a panel that changes
        # between frames changed because the run did.
        "value_max": max(entries, default=1.0) or 1.0,
        "cut_min": min([*cuts, spectral_floor]),
        "cut_max": max([*cuts, spectral_floor]),
        "loss_min": min(losses, default=0.0),
        "loss_max": max(losses, default=1.0),
        "component_max": max((int(frame["component_count"]) for frame in full), default=1),
        "step_count": max((int(frame["step"]) for frame in recording.frames), default=0) + 1,
        "spectral_floor": spectral_floor,
        "banner": {"x": 16, "y": 30, "width": CANVAS_WIDTH - 32, "height": 30},
        "graph": dict(GRAPH_BOX),
        "heatmap": {"x": 500, "y": 96, "width": 384, "height": 236},
        "loss_curve": {"x": 60, "y": 420, "width": 380, "height": 96},
        "cut_curve": {"x": 540, "y": 420, "width": 344, "height": 96},
        "explain": explain(recording).to_dict(),
    }

    return View(
        kind=VIEW_KIND,
        title=f"Two-hot span rcut — {subtitle}",
        javascript=_view_javascript(),
        layout=layout,
    )
