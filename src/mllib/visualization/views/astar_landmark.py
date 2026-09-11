"""The view for an A* landmark search: the panel geometry and the ranges its drawing needs.

The drawing itself is ``astar_landmark.js``, beside this module and read verbatim into the page.
What is computed here is everything the JavaScript would otherwise have to derive by walking the
whole recording in a browser: the bound range every bar and the curve are scaled against, how many
expansions the run took, and where the three panels sit.

The bound range is the one that matters. Scaling each frame's bars to that frame's own extremes
would make every frontier look the same — the cheapest state always a stub, the dearest always full
width — and the reader would lose the one thing the bars are for, which is watching the frontier's
bounds climb as the cheap states are consumed. So the scale is the run's, fixed across every frame,
and a bar that grows between frames means the bound grew.

The second half of the module is the page's prose: ``explain`` derives the opening, one sentence per
frame, the key moments, the legend, the per-frame quantities and the ending from the recording and
nothing else. It is here rather than in the JavaScript for the reason the layout is: the recording
is a Python object at render time and a JSON blob in a browser, and a sentence composed in the
browser is a sentence no test can read.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from mllib.visualization.explain import (
    Explanation,
    LegendEntry,
    Moment,
    Quantity,
    fmt_num,
    fmt_state,
    sort_moments,
    text_of,
    used_terms,
)
from mllib.visualization.recording import Recording
from mllib.visualization.views import View

VIEW_KIND = "astar_landmark"

# The drawing area, in user units; the page scales it to whatever width it is given. The height
# leaves the frontier panel a row past its last bar: a full frontier writes "+ N more waiting"
# under the list, and the curve's own title used to be drawn on the same line as that note.
CANVAS_WIDTH = 780
CANVAS_HEIGHT = 492

# The colours the drawing uses, repeated here because the legend has to show them beside a name.
# The drawing is the source of the pixels; this is the source of the words about them.
FRONTIER_COLOUR = "#9aa6ba"
EXPANDED_COLOUR = "#2f5ea8"
PUSHED_COLOUR = "#2f7a4f"
PRUNED_COLOUR = "#b4462f"
INCUMBENT_COLOUR = "#a2761f"

# Every drawn element that stands for a quantity, keyed by the ``data-legend`` tag the view's
# JavaScript puts on it. The two lists are compared by a test, so an element that is drawn without
# an entry — or explained without being drawn — fails before the page is opened.
LEGEND: tuple[LegendEntry, ...] = (
    LegendEntry(
        key="expanded_bar",
        swatch=EXPANDED_COLOUR,
        name="Expanded state",
        meaning="the state this frame took off the frontier, drawn at its lower bound",
    ),
    LegendEntry(
        key="frontier_bar",
        swatch=FRONTIER_COLOUR,
        name="Frontier state",
        meaning="a state still waiting on the frontier after this expansion, at its lower bound",
    ),
    LegendEntry(
        key="new_bar",
        swatch=PUSHED_COLOUR,
        name="New on the frontier",
        meaning="a frontier state this expansion priced and pushed",
    ),
    LegendEntry(
        key="child_bar",
        swatch=PUSHED_COLOUR,
        name="Pushed child",
        meaning="a child priced from the expanded state that joined the frontier",
    ),
    LegendEntry(
        key="pruned_bar",
        swatch=PRUNED_COLOUR,
        name="Pruned child",
        meaning="a child left off the frontier, with the mechanism that declined it",
    ),
    LegendEntry(
        key="bound_curve",
        swatch=EXPANDED_COLOUR,
        name="Bound curve",
        meaning="the lower bound of every expanded state so far, against its expansion",
    ),
    LegendEntry(
        key="incumbent_line",
        swatch=INCUMBENT_COLOUR,
        name="Incumbent line",
        meaning="the smallest residual trace of any goal state seen so far",
    ),
    LegendEntry(
        key="goal_banner",
        swatch=PUSHED_COLOUR,
        name="Goal banner",
        meaning="the frame at which a goal state was expanded and the run ended",
    ),
)

# How a pruning mechanism is spelled in a sentence: the frames carry the engine's own identifier.
PRUNE_REASONS = {
    "goal_sibling": ("goal sibling", "goal siblings"),
    "above_incumbent": ("above the incumbent", "above the incumbent"),
}


def _view_javascript() -> str:
    """The drawing function's source, read from the sibling file that ships as package data."""
    return files("mllib.visualization.views").joinpath("astar_landmark.js").read_text()


def _bound_range(recording: Recording) -> tuple[float, float]:
    """The smallest and largest value anywhere in the run that the drawing places on that scale.

    Every bound the run produced is included rather than only the expanded ones, because a pruned
    child's bar is drawn on the same scale as the frontier's and would otherwise run off the panel.
    An anytime run's incumbent is on the scale too — it is drawn as a horizontal line across the
    curve — and an incumbent below every bound would otherwise be a line drawn off the bottom of it.
    """
    bounds: list[float] = []
    for frame in recording.frames:
        bounds.append(float(frame["bound"]))
        bounds.extend(float(child[1]) for child in frame["children"])
        bounds.extend(float(entry[0]) for entry in frame["frontier"])
        incumbent = frame["extras"].get("incumbent")
        if incumbent is not None:
            bounds.append(float(incumbent))
    if not bounds:
        return 0.0, 1.0
    return min(bounds), max(bounds)


def _frontier_min(frame: dict[str, Any]) -> float | None:
    """The smallest bound waiting after this expansion; ``None`` when nothing is waiting."""
    frontier = frame["frontier"]
    return float(frontier[0][0]) if frontier else None


def _pruned_clause(frame: dict[str, Any]) -> str:
    """How many children were declined, and by which mechanism, or nothing when none were."""
    pruned = frame["extras"].get("pruned") or []
    if not pruned:
        return ""
    counts: dict[str, int] = {}
    for entry in pruned:
        counts[str(entry[2])] = counts.get(str(entry[2]), 0) + 1
    parts = []
    for reason in sorted(counts):
        count = counts[reason]
        singular, plural = PRUNE_REASONS.get(reason, (reason.replace("_", " "),) * 2)
        parts.append(f"{count} {singular if count == 1 else plural}")
    return f", pruned {len(pruned)} ({', '.join(parts)})"


def _pushed_goal(frame: dict[str, Any], goal_depth: int) -> tuple[list[int], float] | None:
    """The cheapest complete selection this expansion pushed, if it pushed one at all.

    The cheapest, because goal siblings priced in one expansion are indistinguishable to the reader
    and only the smallest-bound one can ever be the first goal expanded (`CONTEXT.md`, goal sibling).
    """
    goals = [
        (list(child[0]), float(child[1]))
        for child in frame["children"]
        if child[2] and len(child[0]) == goal_depth
    ]
    return min(goals, key=lambda entry: (entry[1], entry[0])) if goals else None


def _pushed_bound(recording: Recording, frame: dict[str, Any]) -> float | None:
    """The bound the expanded state was pushed onto the frontier with, or ``None`` if it never was.

    Not the frame's own ``bound``: at a goal state that is the objective recomputed when the state
    came off the heap, and the heap ordered the state by the bound it went on with. The two agree
    to every digit a reader sees and can differ in the last, which is exactly the difference that
    decides whether a tie-break was ever consulted. The root is the one state nothing pushed.
    """
    wanted = list(frame["expanded_state"])
    for earlier in recording.frames[: frame["index"]]:
        for child in earlier["children"]:
            if list(child[0]) == wanted:
                return float(child[1])
    return None


def _goal_frame_index(recording: Recording) -> int | None:
    """The frame at which the run ended on a goal, or ``None`` for a run stopped by its cap."""
    for index in range(len(recording.frames) - 1, -1, -1):
        if recording.frames[index]["goal"]:
            return index
    return None


def _opening(recording: Recording) -> tuple[str, ...]:
    """What the page is, in four lines: the instance, the algorithm, the question, how to read it."""
    problem = recording.problem
    configuration = recording.configuration
    engine = str(configuration.get("engine", "AStarSearch"))

    algorithm = (
        "Algorithm: A* over landmark subsets, expanding the frontier state with the smallest "
        "lower bound on the residual trace"
    )
    if engine == "AStarSearch":
        algorithm += (
            "; a subset of k landmarks is a goal state, and the first goal expanded is proved "
            "optimal because every other state's bound is at least its residual trace."
        )
    else:
        algorithm += (
            "; children whose bound is not below the incumbent are pruned instead of joining the "
            "frontier."
        )
    if engine == "AnytimeAStarSearch":
        algorithm += (
            f" The run stops after {configuration.get('max_expansions')} expansions if no goal "
            "has been expanded and reports the incumbent with a certified gap."
        )

    return (
        f"Instance: cell {problem.get('cell')}, a kernel on n = {problem.get('n')} points; "
        f"the search chooses k = {problem.get('landmark_count')} landmarks.",
        algorithm,
        "Question: which k landmarks give the smallest residual trace, and is that answer proved?",
        "How to read: one frame per expansion. Left, the frontier after the expansion, smallest "
        "bound first; right, the children priced from the expanded state; below, the bound of "
        "every expanded state so far. Markers under the slider are the moments that decide the run.",
    )


def _narrate_goal(recording: Recording, frame: dict[str, Any]) -> str:
    """The sentence for a frame that expanded a goal state, in each of the three ways that ends."""
    state = fmt_state(frame["expanded_state"])
    cost = fmt_num(frame["bound"])
    expansions = frame["expansions"]
    superseded = frame["extras"].get("superseded_goal")
    if superseded is not None:
        return (
            f"Goal {fmt_state(superseded)} at residual trace "
            f"{fmt_num(frame['extras'].get('superseded_cost'))} was within the tie tolerance of "
            f"the held incumbent {state} at {cost}, so the incumbent is returned after "
            f"{expansions} expansions."
        )

    # A goal can be expanded and then set aside: under a tie tolerance the pruned engine pops one
    # goal and returns the incumbent it already holds (D-29), leaving a frame after this one. The
    # optimality sentence would be false here, and the frontier under it says so — it holds cheaper
    # states — so this frame gets the sentence its own frame supports.
    following = recording.frames[frame["index"] + 1 :]
    if following and following[0]["extras"].get("superseded_goal") == list(frame["expanded_state"]):
        return (
            f"Goal {state} expanded at residual trace {cost} after {expansions} expansions, but "
            f"it is within the tie tolerance of the held incumbent "
            f"{fmt_state(frame['extras'].get('incumbent_state'))} at "
            f"{fmt_num(frame['extras'].get('incumbent'))}, so it is set aside."
        )

    remaining = int(frame["frontier_size"])
    waiting = (
        "the frontier is empty"
        if remaining == 0
        else f"{remaining} states remain on the frontier, none with a bound below {cost}"
    )
    proof = (
        f"Goal {state} expanded at residual trace {cost} after {expansions} expansions: "
        f"{waiting}, so {state} is proved optimal"
    )

    # States tied with the answer are the interesting half of a certificate: the search proved the
    # value, and something other than the value decided which member of the optimum set came back
    # (`CONTEXT.md`, optimum set). Ties are counted at display precision, because two bounds a
    # reader sees as the same number are the same number as far as the sentence under them is.
    tied = [entry for entry in frame["frontier"] if fmt_num(entry[0]) == cost]
    if not tied:
        return proof + "."

    # Which mechanism actually decided is a question about the raw bounds, not the printed ones.
    # The heap key is the bound a state was *pushed* with, so that is what is compared: the
    # frame's own ``bound`` is the goal cost recomputed at the pop and can differ from the key in
    # the last digits. Only when the keys are exactly equal — or when a tie tolerance made them
    # equal for the heap's purposes — was the tie-break consulted at all.
    tolerance = float(recording.configuration.get("tie_tolerance") or 0.0)
    pushed_at = _pushed_bound(recording, frame)
    exactly_equal = pushed_at is not None and any(float(entry[0]) == pushed_at for entry in tied)
    verb = "ties" if len(tied) == 1 else "tie"
    if tolerance > 0 or exactly_equal:
        return (
            f"{proof}; {len(tied)} of them {verb} with it at {cost}, and with equal bounds the "
            f"tie-break ({recording.configuration.get('tie_break')}) decided which was expanded "
            "first."
        )
    return (
        f"{proof}; {len(tied)} of them {verb} with it at {cost} to four decimals; the expanded "
        "one had the smaller bound in the digits not shown, so the tie-break was never consulted."
    )


def _narrate_expansion(
    frame: dict[str, Any], previous: dict[str, Any] | None, goal_depth: int, seen_goal: bool
) -> str:
    """The sentence for an ordinary expansion: what it took, what it priced, what that left."""
    children = frame["children"]
    pushed = sum(1 for child in children if child[2])
    bound = fmt_num(frame["bound"])
    size = int(frame["frontier_size"])
    minimum = _frontier_min(frame)

    if previous is None:
        return (
            f"Expanded {fmt_state(frame['expanded_state'])} at bound {bound}; priced "
            f"{len(children)} children and pushed {pushed}, so the frontier opens with {size} "
            f"states and a minimum bound of {fmt_num(minimum)}."
        )

    change = size - int(previous["frontier_size"])
    if change > 0:
        movement = f"grew by {change} to {size} states"
    elif change < 0:
        movement = f"shrank by {-change} to {size} states"
    else:
        movement = f"stayed at {size} states"

    # Compared as the reader sees them, not as the doubles they are: two bounds that differ in the
    # fifteenth decimal are the same number on this page, and "rose from 0.6629 to 0.6629" is a
    # sentence that makes a reader distrust every other one.
    was = _frontier_min(previous)
    shown, was_shown = fmt_num(minimum), fmt_num(was)
    if minimum is None:
        bounds = "and it holds no bound"
    elif was is None:
        bounds = f"and its minimum bound is {shown}"
    elif shown == was_shown:
        bounds = f"and its minimum bound stayed at {shown}"
    elif minimum > was:
        bounds = f"and its minimum bound rose from {was_shown} to {shown}"
    else:
        bounds = f"and its minimum bound fell from {was_shown} to {shown}"

    sentences = [
        f"Expanded {fmt_state(frame['expanded_state'])} "
        f"(depth {len(frame['expanded_state'])}) at bound {bound}, the smallest on the frontier; "
        f"priced {len(children)} children, pushed {pushed}{_pruned_clause(frame)}.",
        f"The frontier {movement} {bounds}.",
    ]

    goal = _pushed_goal(frame, goal_depth)
    if goal is not None and not seen_goal:
        sentences.append(
            f"A goal was priced for the first time: {fmt_state(goal[0])} at {fmt_num(goal[1])}."
        )

    incumbent = frame["extras"].get("incumbent")
    was_incumbent = previous["extras"].get("incumbent")
    if incumbent is not None and was_incumbent is None:
        sentences.append(
            f"The first incumbent is {fmt_state(frame['extras'].get('incumbent_state'))} at "
            f"{fmt_num(incumbent)}."
        )
    elif incumbent is not None and float(incumbent) < float(was_incumbent):
        sentences.append(
            f"The incumbent improved from {fmt_num(was_incumbent)} to {fmt_num(incumbent)} with "
            f"{fmt_state(frame['extras'].get('incumbent_state'))}."
        )
    return " ".join(sentences)


def _narration(recording: Recording, goal_depth: int) -> tuple[str, ...]:
    """One sentence per frame, each read from that frame and the one before it."""
    lines: list[str] = []
    seen_goal = False
    for index, frame in enumerate(recording.frames):
        if frame["goal"]:
            lines.append(_narrate_goal(recording, frame))
        else:
            previous = recording.frames[index - 1] if index else None
            lines.append(_narrate_expansion(frame, previous, goal_depth, seen_goal))
        if _pushed_goal(frame, goal_depth) is not None:
            seen_goal = True
    return tuple(lines)


def _moments(recording: Recording, goal_depth: int) -> tuple[Moment, ...]:
    """The frames that decide the run, in the order a frame claimed by two of them is resolved.

    The order below *is* the priority order, since ``sort_moments`` keeps the first claim on a
    frame. The terminal moment leads it: a short run ends where something else also happened — the
    frontier's peak, or the last improvement of the incumbent — and a page whose last frame carries
    no marker is a page that never says where the run stopped. Then the incumbent, which on a
    capped run is the only marker that says the answer moved; then the first complete selection;
    then the peak, which is a fact about memory rather than about the answer.
    """
    frames = recording.frames
    candidates: list[Moment] = []

    goal_index = _goal_frame_index(recording)
    if goal_index is not None:
        superseded = frames[goal_index]["extras"].get("superseded_goal") is not None
        candidates.append(
            Moment(
                frame_index=goal_index,
                label="returned" if superseded else "proved",
                reason="The run ends: the state expanded here is the answer.",
            )
        )
    elif frames:
        candidates.append(
            Moment(
                frame_index=len(frames) - 1,
                label="capped",
                reason="The expansion cap stopped the run here; the incumbent is reported with "
                "its certified gap.",
            )
        )

    previous: float | None = None
    for index, frame in enumerate(frames):
        incumbent = frame["extras"].get("incumbent")
        if incumbent is not None and (previous is None or float(incumbent) < previous):
            candidates.append(
                Moment(
                    frame_index=index,
                    label="incumbent",
                    reason="The best complete selection seen so far improved, tightening the bar "
                    "every child must clear.",
                )
            )
        if incumbent is not None:
            previous = float(incumbent)

    for index, frame in enumerate(frames):
        goal = _pushed_goal(frame, goal_depth)
        if goal is not None:
            answered = list(recording.result.get("state") or []) == goal[0]
            candidates.append(
                Moment(
                    frame_index=index,
                    label="goal priced",
                    reason=(
                        "The eventual answer entered the frontier here and then waited for every "
                        "cheaper state to be expanded."
                        if answered
                        else "A first complete selection entered the frontier here; the answer "
                        "came later."
                    ),
                )
            )
            break

    sizes = [int(frame["frontier_size"]) for frame in frames]
    if sizes:
        peak = max(sizes)
        candidates.append(
            Moment(
                frame_index=sizes.index(peak),
                label="peak",
                reason=f"The frontier was never larger: {peak} states waited here, the run's "
                "memory cost.",
            )
        )
    return sort_moments(candidates)


def _quantities(recording: Recording) -> tuple[Quantity, ...]:
    """The six numbers the page reads out for whichever frame is showing.

    Formatted here, not in the browser: the page's job is to put a string in a cell, and a number
    rounded in two places is a number that will one day be rounded two ways.
    """
    frames = recording.frames
    bounds: list[str | None] = []
    minima: list[str | None] = []
    sizes: list[int | None] = []
    incumbents: list[str | None] = []
    gaps: list[str | None] = []
    expansions: list[int | None] = []

    for frame in frames:
        minimum = _frontier_min(frame)
        incumbent = frame["extras"].get("incumbent")
        bounds.append(fmt_num(frame["bound"]))
        minima.append(None if minimum is None else fmt_num(minimum))
        sizes.append(int(frame["frontier_size"]))
        incumbents.append(None if incumbent is None else fmt_num(incumbent))
        gaps.append(
            None
            if incumbent is None or minimum is None
            else fmt_num(float(incumbent) - float(minimum))
        )
        expansions.append(int(frame["expansions"]))

    return (
        Quantity(
            key="bound",
            term="Lower bound",
            label="bound of the expanded state",
            definition="the smallest residual trace any completion of the expanded state can reach",
            values=tuple(bounds),
        ),
        Quantity(
            key="frontier_min",
            term="Frontier",
            label="frontier minimum",
            definition="the smallest bound waiting on the frontier after this expansion",
            values=tuple(minima),
        ),
        Quantity(
            key="frontier_size",
            term="Frontier",
            label="frontier size",
            definition="how many states were waiting on the frontier after this expansion",
            values=tuple(sizes),
        ),
        Quantity(
            key="incumbent",
            term="Incumbent",
            label="incumbent",
            definition="the smallest residual trace of any goal state seen so far",
            values=tuple(incumbents),
        ),
        Quantity(
            key="gap",
            term="Certificate",
            label="gap",
            definition="how far the incumbent sits above the frontier's minimum; the optimum lies "
            "in between",
            values=tuple(gaps),
        ),
        Quantity(
            key="expansions",
            term="Expansion",
            label="expansions",
            definition="how many states the search had taken off the frontier by this frame",
            values=tuple(expansions),
        ),
    )


def _frontier_peak(recording: Recording) -> int:
    """The largest the frontier grew, from the result when it says and from the frames when not.

    The engines leave ``frontier_peak`` unset on a result they did not measure it on, and the
    frames hold every frontier size the run passed through, so the number is in the recording
    either way — which is what the ending's last line claims about every number above it.
    """
    stated = recording.result.get("frontier_peak")
    if stated is not None:
        return int(stated)
    return max((int(frame["frontier_size"]) for frame in recording.frames), default=0)


def _ending(recording: Recording) -> tuple[str, ...]:
    """Why the run stopped, and what the reader should hold the answer against."""
    result = recording.result
    frames = recording.frames
    peak = _frontier_peak(recording)
    goal_index = _goal_frame_index(recording)
    state = fmt_state(result.get("state"))
    cost = fmt_num(result.get("cost"))
    expansions = result.get("nodes_expanded")
    lines: list[str] = []

    if goal_index is None:
        lines.append(
            f"Stopped because the cap of {recording.configuration.get('max_expansions')} "
            "expansions was reached before a goal was expanded."
        )
        lines.append(
            f"The incumbent {state} at residual trace {cost} is within "
            f"{fmt_num(result.get('certified_gap'))} of the optimum (certified gap)."
        )
        lines.append(f"Frontier peak {peak} states.")
    else:
        goal_frame = frames[goal_index]
        superseded = goal_frame["extras"].get("superseded_goal")
        if superseded is None:
            lines.append(
                f"Stopped because a goal state was expanded: {state} at residual trace {cost} is "
                f"proved optimal after {expansions} expansions."
            )
        else:
            lines.append(
                f"Stopped because a goal state was expanded: {fmt_state(superseded)} at residual "
                f"trace {fmt_num(goal_frame['extras'].get('superseded_cost'))} was within the tie "
                f"tolerance, so the held incumbent {state} at residual trace {cost} was returned "
                f"after {expansions} expansions."
            )
        remaining = int(goal_frame["frontier_size"])
        lines.append(
            f"Frontier peak {peak} states; {remaining} states were still waiting and never needed "
            "expanding."
        )
        gap = result.get("certified_gap")
        if gap is None or float(gap) == 0.0:
            lines.append(
                "The certificate is exact: the gap between the answer and the best remaining "
                "bound is 0."
            )

    lines.append("Every number above is read from the recording's frames and result.")
    return tuple(lines)


def explain(recording: Recording) -> Explanation:
    """Everything the page says in words about this run, derived from the run and nothing else."""
    goal_depth = int(recording.problem.get("landmark_count", 0))
    opening = _opening(recording)
    narration = _narration(recording, goal_depth)
    moments = _moments(recording, goal_depth)
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


def astar_landmark_view(recording: Recording) -> View:
    """The A* landmark view: its drawing function and the layout that function reads."""
    bound_min, bound_max = _bound_range(recording)
    expansion_max = max((int(frame["expansions"]) for frame in recording.frames), default=1)
    problem = recording.problem
    subtitle = (
        f"{problem.get('cell', 'a kernel')} — choosing {problem.get('landmark_count', '?')} "
        f"landmarks from {problem.get('n', '?')} columns"
    )

    layout: dict[str, Any] = {
        "width": CANVAS_WIDTH,
        "height": CANVAS_HEIGHT,
        "subtitle": subtitle,
        "bound_min": bound_min,
        "bound_max": bound_max,
        "expansion_max": expansion_max,
        "banner": {"x": 16, "y": 32, "width": CANVAS_WIDTH - 32, "height": 30},
        "frontier": {
            "x": 16,
            "y": 88,
            "width": 372,
            "label_width": 92,
            "row_height": 19,
            "bar_height": 13,
            "max_rows": 12,
        },
        "children": {
            "x": 408,
            "y": 88,
            "width": 356,
            "label_width": 92,
            "row_height": 19,
            "bar_height": 13,
            "max_rows": 12,
        },
        # One frontier row below where it used to sit, so the panel's overflow note and this
        # panel's title never land on the same line (a full frontier writes both).
        "curve": {"x": 60, "y": 368, "width": 700, "height": 96},
        "explain": explain(recording).to_dict(),
    }

    return View(
        kind=VIEW_KIND,
        title=f"A* landmark search — {subtitle}",
        javascript=_view_javascript(),
        layout=layout,
    )
