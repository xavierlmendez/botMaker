"""Putting a recording, a view and the stepper into one file that works with nothing else present.

The constraint that decides every choice here is that the page has to open from ``file://`` on a
machine that has never heard of this repository. That rules out a build step, a module script, a
CDN, a stylesheet beside the file, and ``fetch`` — all of which fail or are blocked under the
file-URL origin. What is left is string substitution: the template, the stepper's JavaScript, the
view's JavaScript and two JSON documents, concatenated into one file.

The recording travels as ``<script type="application/json">`` rather than as a JavaScript literal
because a JSON block is inert. Nothing in it is evaluated, so a caption is a caption even when a
state's label happens to look like code, and a reader (or a test) can lift the document straight
back out of the page and compare it to the one on disk. The one thing that *can* escape such a block
is the character pair that opens a closing tag, so it is written as ``<\\/``, which JSON reads back
as the same string — the escape is invisible by the time anything reads the document.

The substitution is done with ``str.replace`` and not ``str.format`` or an f-string, because the
template is mostly CSS and CSS is mostly braces.
"""

from __future__ import annotations

import html
import json
import re
from importlib.resources import files
from pathlib import Path

from mllib.visualization.recording import Recording
from mllib.visualization.views import View

TEMPLATE_NAME = "template.html"
STEPPER_NAME = "walkthrough.js"

# The tokens the template carries. Named rather than inlined so that a template edit that drops one
# fails here, at render time, with the token's name in the message.
TOKENS = (
    "{{TITLE}}",
    "{{WIDTH}}",
    "{{HEIGHT}}",
    "{{RECORDING_JSON}}",
    "{{LAYOUT_JSON}}",
    "{{VIEW_JS}}",
    "{{WALKTHROUGH_JS}}",
)

# The tokens as one alternation, so the whole template is substituted in a single pass.
TOKEN_PATTERN = re.compile("|".join(re.escape(token) for token in TOKENS))


def _asset(name: str) -> str:
    """One of the package's shipped text assets (see ``[tool.setuptools.package-data]``)."""
    return files("mllib.visualization").joinpath(name).read_text()


def escape_for_script(payload: str) -> str:
    """Make a JSON document safe to sit inside a ``<script>`` element.

    Only ``</`` can end the element early, and ``\\/`` is a JSON escape for ``/``, so replacing the
    pair leaves a document that parses back to exactly the same values.
    """
    return payload.replace("</", "<\\/")


def render_walkthrough(recording: Recording, view: View) -> str:
    """The whole page as one string: chrome, stepper, view and the recording it draws."""
    if view.kind != recording.kind:
        raise ValueError(
            f"View {view.kind!r} was asked to draw a {recording.kind!r} recording; "
            "the view is chosen from the recording's problem kind."
        )
    for source, name in (
        (view.javascript, f"view {view.kind!r}"),
        (_asset(STEPPER_NAME), "stepper"),
    ):
        if "</script" in source.lower():
            raise ValueError(f"The {name} JavaScript closes the script element it is inlined into.")

    layout = dict(view.layout)
    page = _asset(TEMPLATE_NAME)
    substitutions = {
        "{{TITLE}}": html.escape(view.title),
        "{{WIDTH}}": str(layout.get("width", 780)),
        "{{HEIGHT}}": str(layout.get("height", 470)),
        "{{RECORDING_JSON}}": escape_for_script(json.dumps(recording.to_dict(), sort_keys=True)),
        "{{LAYOUT_JSON}}": escape_for_script(json.dumps(layout, sort_keys=True)),
        "{{VIEW_JS}}": view.javascript,
        "{{WALKTHROUGH_JS}}": _asset(STEPPER_NAME),
    }
    for token in TOKENS:
        if token not in page:
            raise ValueError(f"The walkthrough template no longer carries the token {token}.")

    # One pass over the template, not one pass per token. Replacing them in turn would let a value
    # substituted early be scanned again for a later token, so a recording holding the literal text
    # of a token — in a caption, a cell name, anything — would have that text expanded into the
    # page. ``re.sub`` with a callback never re-examines what it has already written.
    return TOKEN_PATTERN.sub(lambda match: substitutions[match.group(0)], page)


def write_walkthrough(recording: Recording, view: View, path: str | Path) -> Path:
    """Render the page and write it, creating the directory it goes in."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_walkthrough(recording, view))
    return destination
