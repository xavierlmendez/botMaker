"""The document a run leaves behind: its frames, what it was, and what it returned.

A recording is written once, when a run ends, and read by something that was not there when it ran —
a renderer, a diff, a test that committed it as a fixture. That is the whole reason it is a plain
versioned JSON document rather than a pickle or an object graph: the only thing a reader needs in
order to open one is a JSON parser and the version number at the top.

Three properties are load-bearing and each costs something to keep.

*Exact floats.* ``json`` writes a float through ``repr``, which round-trips every finite double
exactly, and this module never rounds or formats one on the way out. A bound that reads ``0.8317``
in a caption is still the full double in the frame, so a recording can be compared to another
recording bit for bit and a disagreement means the search disagreed.

*No timestamps.* Nothing here records when it ran. A document with a clock in it differs from itself
on every regeneration, which would make a committed recording unreviewable and a fixture useless as
a baseline — the same reason the behavioural baselines carry no wall clock (CONTRIBUTING § tests).
The cost is that a recording cannot say when it was made; the run's identity lives in ``problem``
and ``configuration`` instead, which is what actually decides whether two recordings are comparable.

*A version that is checked.* ``schema_version`` is refused when it is not one this module knows, so
an old document meeting a newer renderer fails at ``load`` with a sentence, rather than rendering
half a page and leaving a reader to wonder which half is true.

The type stays a dumb document. It does not know how to draw itself (that is a view), it does not
know which algorithm produced it (that is ``problem["kind"]``, a string), and it holds frames as
plain dicts rather than as ``Frame`` objects, because the recorder that owned those objects has
already converted them and re-inflating them here would only give a second definition of the same
shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mllib.math.recorder import AbstractRecorder

# The version of the document shape below. Bumped when a reader of an existing recording would be
# wrong to assume it, never when a problem adds a field inside ``problem`` or a frame.
SCHEMA_VERSION = 1

# Every version this module can read. It is a set rather than a comparison so that dropping support
# for an old shape is a deletion here, not a rewritten inequality somewhere else.
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

# The keys a document must have to be one of these at all.
REQUIRED_KEYS = ("schema_version", "problem", "configuration", "frames", "result")


@dataclass(frozen=True, slots=True)
class Recording:
    """The frames of one run plus its metadata and the run's own result, as a JSON document.

    ``problem`` says what was solved and must name a ``kind`` — the string a renderer dispatches a
    view on — beside whatever size and name fields that problem needs to be identified (for the
    landmark search: the cell, ``n`` and the landmark count). ``configuration`` says how the engine
    was set up, which is what makes two recordings of the same problem comparable or not.
    ``frames`` are the recorder's own dicts, dense and in order, and ``result`` is what the run
    returned, described by the recorder that watched it.
    """

    schema_version: int
    problem: dict[str, Any]
    configuration: dict[str, Any]
    frames: list[dict[str, Any]]
    result: dict[str, Any]

    @classmethod
    def from_recorder(
        cls,
        recorder: AbstractRecorder,
        *,
        problem: dict[str, Any],
        configuration: dict[str, Any],
        result: object,
    ) -> Recording:
        """Assemble the document from the recorder a run was watched with and that run's result.

        The recorder converts: ``frame_dicts`` for the frames it collected and ``describe_result``
        for the result object, so this method never learns what an ``AStarFrame`` or a
        ``SearchResult`` is. That is also why ``result`` is typed ``object`` — the recorder knows
        which type it is describing, and this module deliberately does not.
        """
        return cls(
            schema_version=SCHEMA_VERSION,
            problem=dict(problem),
            configuration=dict(configuration),
            frames=recorder.frame_dicts(),
            result=recorder.describe_result(result),
        )

    @property
    def kind(self) -> str:
        """The problem kind a renderer dispatches a view on."""
        return str(self.problem["kind"])

    def to_dict(self) -> dict[str, Any]:
        """The document as plain Python, in the shape ``save`` writes and ``load`` reads."""
        return {
            "schema_version": self.schema_version,
            "problem": self.problem,
            "configuration": self.configuration,
            "frames": self.frames,
            "result": self.result,
        }

    def save(self, path: str | Path) -> Path:
        """Write the document as JSON: indented, key-sorted, newline-terminated, exact floats.

        The formatting is not cosmetic. Sorted keys and a fixed indent make a committed recording
        diff line by line when a number changes, and the trailing newline is what the
        ``end-of-file-fixer`` pre-commit hook expects of a committed file.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json())
        return destination

    def to_json(self) -> str:
        """The document as the exact JSON text ``save`` writes, trailing newline included."""
        # allow_nan=False: JSON has no NaN and the page's JSON.parse would refuse the document.
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        """Read a document back, refusing one this module cannot honestly claim to understand."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, payload: object) -> Recording:
        """Build a recording from a parsed document, checking the two things a reader relies on.

        The version, because a shape this module does not know is a shape it must not guess at; and
        ``problem["kind"]``, because without it no view can be chosen and the failure would
        otherwise surface much later, as a missing key inside a renderer.
        """
        if not isinstance(payload, dict):
            raise ValueError(
                f"A recording is a JSON object; this file holds {type(payload).__name__}."
            )
        missing = [key for key in REQUIRED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"This recording is missing the key(s) {', '.join(missing)}.")

        version = payload["schema_version"]
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            known = ", ".join(str(number) for number in sorted(SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(
                f"Recording schema_version {version!r} is not one this library reads "
                f"(known: {known}). The document was written by a different version of "
                "mllib.visualization."
            )

        problem = payload["problem"]
        if not isinstance(problem, dict) or not problem.get("kind"):
            raise ValueError(
                "This recording's 'problem' does not name a 'kind', "
                "so no view can be chosen for it."
            )

        return cls(
            schema_version=int(version),
            problem=dict(problem),
            configuration=dict(payload["configuration"]),
            frames=list(payload["frames"]),
            result=dict(payload["result"]),
        )
