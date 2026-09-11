"""What a problem looks like: the drawing half of a walkthrough, one module per problem.

The template and the stepper are the same for every recording — previous, next, a slider, a play
button, a caption, a raw-fields panel. Everything *inside* the picture is not, and a renderer that
tried to draw a frontier and a heatmap from one function would be a renderer nobody could add a
problem to. So the split is: the page owns the chrome, a **view** owns the drawing.

A view is two things travelling together. The JavaScript is the drawing function itself, inlined
into the page, which registers under the problem's kind in ``window.walkthroughViews``. The layout
is whatever that function should not have to work out for itself in a browser — positions, ranges,
the extent of a curve — computed once in Python, where numpy is, and embedded beside the recording.
Splitting them this way keeps the JavaScript small enough to read in one screen and keeps every
number the picture depends on visible in the document rather than hidden in an axis calculation.

``view_for`` is the whole registry: a recording names its ``kind`` and this module maps that string
to a builder. An unknown kind raises rather than falling back to a generic picture, because a
generic picture of a search is a picture that is wrong in a way nobody notices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mllib.visualization.recording import Recording


@dataclass(frozen=True, slots=True)
class View:
    """One problem's drawing: the JavaScript that renders a frame and the data it renders from.

    ``kind`` matches the recording's ``problem["kind"]`` and is the key the JavaScript registers
    itself under. ``title`` is what the page calls itself. ``javascript`` is the source of the view
    function, inlined verbatim into the page: it must assign
    ``window.walkthroughViews[kind] = function (frame, recording, svg, layout, draw) { … }`` and
    must not reach outside the page for anything. ``layout`` is the precomputed data that function
    reads, embedded as JSON, so it must be JSON-serialisable — a numpy scalar in here fails at
    render time, which is the right time for it to fail.
    """

    kind: str
    title: str
    javascript: str
    layout: dict[str, Any] = field(default_factory=dict)


def view_builders() -> dict[str, Callable[[Recording], View]]:
    """Every problem kind that can be drawn, mapped to the builder that draws it.

    The imports are inside the function on purpose: a view module imports ``View`` from this
    package, so importing the view modules at the top of this one would close a cycle. Deferring
    them costs one import on the first render and keeps both modules free to import the other's
    half of the contract.
    """
    from mllib.visualization.views.astar_landmark import astar_landmark_view
    from mllib.visualization.views.graph_traversal import graph_traversal_view
    from mllib.visualization.views.two_hot_span import two_hot_span_view

    return {
        "astar_landmark": astar_landmark_view,
        "graph_traversal": graph_traversal_view,
        "two_hot_span": two_hot_span_view,
    }


def view_for(recording: Recording) -> View:
    """The view for a recording's problem kind, or a refusal naming the kinds that exist."""
    builders = view_builders()
    kind = recording.kind
    if kind not in builders:
        known = ", ".join(sorted(builders)) or "none"
        raise ValueError(
            f"No walkthrough view is registered for problem kind {kind!r} (known: {known})."
        )
    return builders[kind](recording)
