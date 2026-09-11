"""The shared explanation layer: the shape a view fills in, and the words it formats numbers with.

Two claims are worth a test here rather than in a view. That an ``Explanation`` survives the trip
into a page — it is embedded as JSON, so anything that cannot be serialised is a page that fails at
render time. And that the glossary matcher marks a term when it is used and only then, since that
rule decides which definitions a page carries and is the one place a paraphrase could silently
disappear from a tooltip.
"""

import json

import pytest

from mllib.visualization.explain import (
    Explanation,
    LegendEntry,
    Moment,
    Quantity,
    fmt_delta,
    fmt_num,
    fmt_state,
    sort_moments,
    text_of,
    used_terms,
)


def _explanation() -> Explanation:
    return Explanation(
        opening=("Instance: a kernel.", "Algorithm: A*.", "Question: which?", "How to read: step."),
        narration=("Expanded the empty set.", "Expanded {1}."),
        moments=(Moment(frame_index=1, label="proved", reason="The run ends here."),),
        legend=(
            LegendEntry(key="frontier_bar", swatch="#9aa6ba", name="Frontier", meaning="wait"),
        ),
        quantities=(
            Quantity(key="bound", term="Lower bound", definition="a bound", values=(1, None)),
        ),
        ending=("Stopped.", "Read against nothing."),
        glossary={"Frontier": "The generated states not yet expanded."},
    )


def test_an_explanation_survives_the_trip_into_the_page_as_json():
    # It rides in the layout, which is dumped with json.dumps; a tuple or a dataclass that did not
    # flatten would fail at render time on a page nobody asked to change.
    payload = _explanation().to_dict()

    assert json.loads(json.dumps(payload)) == payload


def test_the_explanations_dict_keeps_a_value_per_frame_for_every_quantity():
    payload = _explanation().to_dict()

    assert payload["quantities"][0]["values"] == [1, None]
    assert payload["moments"] == [
        {"frame_index": 1, "label": "proved", "reason": "The run ends here."}
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, "3"), (0.5345281449861901, "0.5345"), (1.0, "1.0000"), (None, "unknown")],
)
def test_a_number_reads_the_way_the_captions_read(value, expected):
    assert fmt_num(value) == expected


def test_a_change_carries_a_real_minus_sign_rather_than_a_hyphen():
    assert fmt_delta(0.0123) == "+0.0123"
    assert fmt_delta(-0.0123) == "\u22120.0123"


@pytest.mark.parametrize(
    ("state", "expected"),
    [((), "the empty set"), ((6, 1, 4), "{1, 4, 6}"), ((3,), "{3}"), (None, "unknown")],
)
def test_a_state_reads_as_the_set_it_is(state, expected):
    assert fmt_state(state) == expected


def test_only_one_moment_survives_a_frame_and_the_first_claim_is_the_one_kept():
    kept = sort_moments(
        (
            Moment(frame_index=4, label="goal priced", reason="first"),
            Moment(frame_index=4, label="peak", reason="second"),
            Moment(frame_index=1, label="incumbent", reason="third"),
        )
    )

    assert [moment.frame_index for moment in kept] == [1, 4]
    assert kept[1].label == "goal priced"


def test_the_scanned_text_is_every_string_the_reader_will_see():
    explanation = _explanation()

    blocks = text_of(
        (explanation.opening, explanation.narration, explanation.legend, explanation.quantities)
    )

    assert "Instance: a kernel." in blocks
    assert "Frontier" in blocks and "wait" in blocks
    assert "Lower bound" in blocks and "a bound" in blocks


GLOSSARY = {"Frontier": "waiting states", "Lower bound": "a floor", "2-hot vector": "a pair"}


def test_a_term_is_carried_when_prose_lowercases_its_first_letter():
    assert used_terms(("the frontier is empty",), GLOSSARY) == {"Frontier": "waiting states"}


def test_a_term_is_not_carried_when_it_only_occurs_inside_another_word():
    # "frontiersman" is not the frontier, and a page that defined it would be teaching a word it
    # never used.
    assert used_terms(("a frontiersman and a lower boundary",), GLOSSARY) == {}


def test_the_two_hot_term_answers_to_the_short_spelling_prose_actually_uses():
    assert used_terms(("three columns are 2-hot",), GLOSSARY) == {"2-hot vector": "a pair"}


def test_a_multi_word_term_is_matched_as_the_whole_phrase():
    assert used_terms(("its lower bound rose",), GLOSSARY) == {"Lower bound": "a floor"}


NESTED = {"Objective": "what a search minimises", "Relaxed objective": "E*, the continuous one"}


def test_a_term_swallowed_by_a_longer_one_is_not_counted_as_used():
    # "Relaxed objective" contains "objective". The marker claims the longer term first, so the
    # shorter one's definition would end up on whatever bare word the prose writes next.
    assert used_terms(("the relaxed objective fell",), NESTED) == {
        "Relaxed objective": "E*, the continuous one"
    }


def test_a_term_still_counts_where_it_occurs_outside_the_longer_one():
    assert used_terms(("the objective and the relaxed objective",), NESTED) == NESTED
