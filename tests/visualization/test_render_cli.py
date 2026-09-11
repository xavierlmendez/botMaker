"""The CLI: a recording on disk, rendered by someone who did not run it.

This is the path that proves a recording is self-describing. The command is given a file and nothing
else — no cell, no view name, no kernel — and everything it needs to draw the run has to come out of
the document.
"""

import json
import re

import pytest

from mllib.visualization.render import main

EMBEDDED = re.compile(r'<script id="recording" type="application/json">(.*?)</script>', re.DOTALL)


def test_the_cli_renders_a_recording_it_is_only_given_the_path_of(
    fixture_path, fixture_recording, tmp_path
):
    written = main(["--recording", str(fixture_path), "--output", str(tmp_path / "page.html")])

    assert written == tmp_path / "page.html"
    assert json.loads(EMBEDDED.search(written.read_text()).group(1)) == fixture_recording.to_dict()


def test_the_cli_refuses_a_document_it_cannot_read(fixture_recording, tmp_path):
    payload = fixture_recording.to_dict()
    payload["schema_version"] = 99
    (tmp_path / "future.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema_version"):
        main(["--recording", str(tmp_path / "future.json"), "--output", str(tmp_path / "p.html")])
