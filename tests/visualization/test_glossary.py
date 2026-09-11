"""The glossary against `CONTEXT.md`: a page cannot coin a word.

This is the whole enforcement of D-34's second half. The tooltips on a walkthrough are the
repository's own vocabulary, so a term that is not a `**Term**:` heading of `CONTEXT.md` fails
here — the domain model gains the word first, in the same PR, or the narration finds another one.
"""

import re
from pathlib import Path

from mllib.visualization.glossary import GLOSSARY

CONTEXT_PATH = Path(__file__).resolve().parents[2] / "CONTEXT.md"

# A heading is a line that is nothing but a bolded term and a colon.
HEADING = re.compile(r"^\*\*(.+?)\*\*:$", re.MULTILINE)

# A tooltip is one line and has to fit in one; past this it is an essay in a hover.
MAXIMUM_DEFINITION_CHARACTERS = 140


def context_terms() -> set[str]:
    return set(HEADING.findall(CONTEXT_PATH.read_text()))


def test_every_glossary_key_is_a_term_the_domain_model_defines():
    unknown = sorted(set(GLOSSARY) - context_terms())

    assert unknown == [], f"not `**Term**:` headings of CONTEXT.md: {unknown}"


def test_every_definition_is_one_line_short_enough_to_hover():
    for term, definition in sorted(GLOSSARY.items()):
        assert "\n" not in definition, term
        assert definition.strip() == definition, term
        assert len(definition) <= MAXIMUM_DEFINITION_CHARACTERS, (term, len(definition))
        assert definition.endswith("."), term
