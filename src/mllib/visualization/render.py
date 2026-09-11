"""Render a saved recording as a walkthrough page.

    uv run python -m mllib.visualization.render --recording run.json --output run.html

The example that produced a recording is a composition root and knows the cell it ran; this is the
other direction — a document on disk, rendered by whoever has it, with no access to the run. That is
the whole reason a recording carries its problem kind: the view is chosen from the document, so a
recording made months ago by code nobody has any more still draws itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mllib.visualization.html_renderer import write_walkthrough
from mllib.visualization.recording import Recording
from mllib.visualization.views import view_for


def main(argv: list[str] | None = None) -> Path:
    """Load a recording, pick the view its problem kind names, write the page; return its path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recording", required=True, type=Path, help="the recording JSON to render"
    )
    parser.add_argument("--output", required=True, type=Path, help="the HTML page to write")
    arguments = parser.parse_args(argv)

    recording = Recording.load(arguments.recording)
    written = write_walkthrough(recording, view_for(recording), arguments.output)
    print(f"{written} — {len(recording.frames)} frames of {recording.kind}")
    return written


if __name__ == "__main__":
    main()
