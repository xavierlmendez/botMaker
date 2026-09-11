"""The recording document: what it is assembled from, and what survives a trip through disk.

The claims here are the ones a committed fixture rests on. A recording that lost a float's last bits
on the way out would make every baseline comparison a comparison of rounding, and a recording that
carried a clock would differ from itself on every regeneration.
"""

import json

import pytest

from mllib.visualization.recording import SCHEMA_VERSION, Recording

# Anything a reader would call a timestamp. A key matching one of these in a saved document means a
# recording has stopped being reproducible, whoever added it and for whatever reason.
TIMESTAMP_WORDS = ("time", "date", "when", "clock", "stamp", "ran_at", "created")


def test_from_recorder_takes_the_frames_and_the_result_from_the_recorder(
    small_recording, recorded_run
):
    result, recorder, _ = recorded_run

    assert small_recording.schema_version == SCHEMA_VERSION
    assert small_recording.frames == recorder.frame_dicts()
    assert small_recording.result == recorder.describe_result(result)


def test_the_frames_are_dense_and_in_the_order_the_run_produced_them(small_recording):
    assert [frame["index"] for frame in small_recording.frames] == list(
        range(len(small_recording.frames))
    )


def test_saving_and_loading_returns_every_float_bit_for_bit(small_recording, tmp_path):
    # `==` on floats is the assertion, not `approx`: a recording that rounds is a recording that
    # cannot be diffed against a committed one.
    small_recording.save(tmp_path / "run.json")

    assert Recording.load(tmp_path / "run.json").to_dict() == small_recording.to_dict()


def test_a_saved_recording_ends_with_a_newline(small_recording, tmp_path):
    path = small_recording.save(tmp_path / "run.json")

    assert path.read_text().endswith("}\n")


def test_a_saved_recording_names_no_timestamp_anywhere(small_recording, tmp_path):
    small_recording.save(tmp_path / "run.json")

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    named = [
        key
        for key in keys(json.loads((tmp_path / "run.json").read_text()))
        if any(word in key.lower() for word in TIMESTAMP_WORDS)
    ]
    assert named == []


def test_a_schema_version_this_library_does_not_know_is_refused(small_recording, tmp_path):
    payload = small_recording.to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    (tmp_path / "future.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema_version"):
        Recording.load(tmp_path / "future.json")


def test_a_recording_whose_problem_names_no_kind_is_refused(small_recording, tmp_path):
    payload = small_recording.to_dict()
    payload["problem"] = {"cell": "rbf_chain_4x4_k2"}
    (tmp_path / "kindless.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="kind"):
        Recording.load(tmp_path / "kindless.json")


def test_a_document_missing_a_top_level_key_is_refused(tmp_path):
    (tmp_path / "partial.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION}))

    with pytest.raises(ValueError, match="missing the key"):
        Recording.load(tmp_path / "partial.json")
