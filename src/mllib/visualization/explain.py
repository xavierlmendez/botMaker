"""What a walkthrough says about itself: the sentences a view derives from a recording.

A caption tells a reader what happened in a frame. It does not tell them what the run is, what the
numbers mean, which frames decide the answer, or why the run stopped — and a page that leaves those
four to the reader is a page only its author can read. This module is the shape those four take.

The rule the shape exists to enforce is that **nothing here is authored per run**. An
``Explanation`` is a function of a recording: the same document produces the same sentences, so a
regenerated page differs exactly when the run differed, and a narration that has drifted from the
frames is a failing test rather than stale prose. The corollary is that the page's JavaScript never
composes a sentence either. It receives finished strings and puts them in the document; every
number in them was formatted in Python, where the recording is, against the same precision the
recorder captioned with.

The formatting helpers are here rather than in each view because a page that prints a bound to four
decimals in one panel and to full precision in another is a page that looks like two runs. ``Δ`` is
spelled "by" in prose ("the frontier grew by 4"), and a signed number uses a real minus sign, which
is what a reader's eye reads as one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# Decimals on a number in prose. The same precision the recorders caption with, so a bound reads
# the same in the narration as in the caption under it.
NUMBER_PRECISION = 4

# A hyphen is a hyphen; a minus sign is what a negative number starts with. Spelled as its escape
# because it is otherwise indistinguishable from the hyphen beside it in a source file.
MINUS_SIGN = "\u2212"


def fmt_num(value: float | int | None) -> str:
    """A number as prose reads it: an integer stays an integer, anything else gets four decimals."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{NUMBER_PRECISION}f}"


def fmt_delta(value: float | int) -> str:
    """A signed change, with a real minus sign so a negative number does not read as a hyphen."""
    text = fmt_num(abs(value))
    return f"+{text}" if value >= 0 else f"{MINUS_SIGN}{text}"


def fmt_state(state: Sequence[int] | None) -> str:
    """A search state as a set: ``{1, 4, 6}``, sorted, and named when it is empty.

    Set braces rather than the tuple the frame holds, because a landmark selection is a set — two
    orders of the same indices are the same state, and the drawing's tuple label is the one place
    the page still says otherwise.
    """
    if state is None:
        return "unknown"
    members = sorted(int(index) for index in state)
    if not members:
        return "the empty set"
    return "{" + ", ".join(str(index) for index in members) + "}"


@dataclass(frozen=True, slots=True)
class Moment:
    """A frame the view judged to decide the run, marked on the stepper.

    ``label`` is what fits on a marker — three words at most — and ``reason`` is the one sentence
    the page appends to that frame's narration when it is showing, which says why the frame counts
    rather than repeating what it holds.
    """

    frame_index: int
    label: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "label": self.label, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class LegendEntry:
    """One drawn thing, named and explained.

    ``key`` is the contract between this module and the view's JavaScript: the drawing tags the
    element it draws with ``data-legend="<key>"``, and a test reads both lists and compares them,
    so a legend entry for something nobody draws — or a drawn thing nothing explains — is caught
    without opening a browser.
    """

    key: str
    swatch: str
    name: str
    meaning: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "swatch": self.swatch, "name": self.name, "meaning": self.meaning}


@dataclass(frozen=True, slots=True)
class Quantity:
    """A number the page shows for every frame, under the glossary term it is called by.

    ``values`` is dense — one entry per frame, ``None`` where the frame has no such number — so the
    page indexes it with the frame index and never has to decide what a missing entry means.

    ``label`` is what the strip calls this number; ``term`` is the glossary word it belongs to.
    They differ whenever one term names two numbers — the frontier's minimum bound and its size
    are both the frontier, and a strip printing "Frontier" twice names neither. It defaults to the
    term, so a view with no such collision states each quantity once.
    """

    key: str
    term: str
    definition: str
    values: tuple[float | int | str | None, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.term)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "term": self.term,
            "label": self.label,
            "definition": self.definition,
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class Explanation:
    """Everything a walkthrough says in words, derived from the recording it says it about."""

    opening: tuple[str, ...]
    narration: tuple[str, ...]
    moments: tuple[Moment, ...]
    legend: tuple[LegendEntry, ...]
    quantities: tuple[Quantity, ...]
    ending: tuple[str, ...]
    glossary: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """The whole explanation as JSON-serialisable data, ready to ride in the page's layout."""
        return {
            "opening": list(self.opening),
            "narration": list(self.narration),
            "moments": [moment.to_dict() for moment in self.moments],
            "legend": [entry.to_dict() for entry in self.legend],
            "quantities": [quantity.to_dict() for quantity in self.quantities],
            "ending": list(self.ending),
            "glossary": dict(self.glossary),
        }


def text_of(explanation_parts: Iterable[Any]) -> tuple[str, ...]:
    """Every string a glossary scan should see, flattened out of the pieces it is scanned over."""
    blocks: list[str] = []
    for part in explanation_parts:
        if isinstance(part, str):
            blocks.append(part)
        elif isinstance(part, LegendEntry):
            blocks.extend((part.name, part.meaning))
        elif isinstance(part, Quantity):
            blocks.extend((part.term, part.label, part.definition))
        elif isinstance(part, Moment):
            blocks.append(part.reason)
        else:
            blocks.extend(text_of(part))
    return tuple(blocks)


def sort_moments(moments: Iterable[Moment]) -> tuple[Moment, ...]:
    """Moments in frame order, one per frame: the first claim on a frame is the one that keeps it.

    One marker per frame is a drawing constraint — two markers at the same slider position overlap
    into an unreadable smudge — so the view lists its candidates in the order it wants them
    resolved and this drops the later claim on a frame that is already spoken for.
    """
    kept: dict[int, Moment] = {}
    for moment in moments:
        kept.setdefault(moment.frame_index, moment)
    return tuple(sorted(kept.values(), key=lambda moment: moment.frame_index))


def pattern_for(term: str) -> re.Pattern[str]:
    """The word-boundary pattern a glossary term is looked for by, case-insensitively.

    Prose lowercases a term's first letter mid-sentence ("the frontier", not "the Frontier"), so
    the match ignores case entirely. ``"2-hot vector"`` also answers to ``"2-hot"``, which is how
    the two-hot vocabulary is actually written in a sentence.
    """
    spellings = [term, "2-hot"] if term == "2-hot vector" else [term]
    body = "|".join(re.escape(spelling) for spelling in spellings)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def used_terms(
    text_blocks: Iterable[str], glossary: dict[str, str] | None = None
) -> dict[str, str]:
    """The glossary entries whose term actually occurs in the given text, and no others.

    A page carries the words it uses. Shipping the whole glossary would put a tooltip's worth of
    vocabulary about two-hot spanning sets on a page about a search, which is how a glossary stops
    being read.

    A term inside a longer one does not count as used. "Relaxed objective" contains "objective",
    and a page that never writes the bare word would otherwise carry the search sense of it — and
    the marker, which claims the longest term first, would then hang that definition on the next
    bare "objective" the prose does write. So the spans are claimed longest term first, exactly as
    the marker claims them, and a shorter term survives only where it occurs outside all of them.
    """
    from mllib.visualization.glossary import GLOSSARY

    entries = GLOSSARY if glossary is None else glossary
    haystack = "\n".join(text_blocks)

    claimed: list[tuple[int, int]] = []
    used: dict[str, str] = {}
    for term in sorted(entries, key=lambda name: (-len(name), name)):
        matches = [
            match.span()
            for match in pattern_for(term).finditer(haystack)
            if not any(start <= match.start() and match.end() <= end for start, end in claimed)
        ]
        if matches:
            used[term] = entries[term]
            claimed.extend(matches)
    return {term: used[term] for term in sorted(used)}
