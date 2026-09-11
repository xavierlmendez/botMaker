"""The repository's words, in one line each, for a reader who has never opened `CONTEXT.md`.

A walkthrough is read by someone who was not there when the run happened, so every term in its
prose has to be answerable on the page itself. That is all this module is: a tooltip's worth of
`CONTEXT.md`, so that "frontier" and "incumbent" can be used in a sentence rather than avoided.

The keys are exactly the `**Term**:` headings of `CONTEXT.md`, checked by a test that parses that
file. That constraint is the point. A page cannot coin a word: if the narration wants one, the
domain model gains it first (`docs/DECISIONS.md` D-34), which keeps the walkthroughs speaking the
same language as the code and the plans instead of a friendlier dialect of it. The values are
paraphrases, not copies — one line that fits a tooltip, where `CONTEXT.md` has room for the
_Avoid_ list and the citation.
"""

from __future__ import annotations

# Search vocabulary, in its standard sense; the terms every A* page uses.
SEARCH_TERMS: dict[str, str] = {
    "State": "A configuration the search can be in; here, a set of chosen landmarks.",
    "Goal state": "A state that is a complete solution, at which the search may stop.",
    "Successor": "A state reachable from another by one action; the problem defines them.",
    "Expansion": "Taking the best state off the frontier and pricing its successors.",
    "Frontier": "The generated states not yet expanded, ordered by their bound.",
    "Completion": "Any goal state reachable from a given state.",
    "Objective": "The value the search minimises, defined on goal states alone.",
    "Lower bound": (
        "A value never above the objective of any completion of a state; at a goal state it "
        "equals the objective."
    ),
    "Certificate": "The proof of optimality a search delivers when it finishes inside its budget.",
    "Bounded certificate": "A proof that a solution is within a stated factor of optimal.",
    "Optimum set": "Every solution attaining the optimal objective; ties are real.",
    "Tie collapse": "The regime where states share a bound, so the certificate becomes enumeration.",
    "Expanded set": "The states already expanded, kept so one is never expanded twice.",
    "Frontier peak": "The largest the frontier ever grew; the search's memory cost.",
    "Incumbent": "The smallest objective of any goal state seen so far; an upper bound.",
    "Goal sibling": "One of the goal states priced from the same parent in one expansion.",
    "Pruning": "Declining to push a priced child whose bound exceeds an upper bound on the optimum.",
    "Memory cap": "A run stopped because its frontier or memory exceeded a budget; no certificate.",
}

# The Nyström problem the landmark search solves.
NYSTROM_TERMS: dict[str, str] = {
    "Landmark": "A chosen data index whose kernel column represents the others.",
    "Residual kernel": "What the landmarks leave unexplained: the kernel minus its reconstruction.",
    "Residual trace": "The trace of the residual kernel; the objective every selector minimises.",
    "Column subset selection": "Choosing columns so their span loses the least Frobenius energy.",
    "Goal depth": "The depth at which a child is a complete selection, so its bound is exact.",
    "Cell": "One experimental configuration: dataset, kernel, bandwidth scale, n and k.",
}

# What a page is made of, for a reader who wonders what they are looking at.
RECORDING_TERMS: dict[str, str] = {
    "Recorder": "An injected collaborator an algorithm hands its state to; off by default.",
    "Frame": "One recorded moment of a run — an expansion, or a step — with a caption.",
    "Recording": "The frames of one run plus its metadata and result, as a JSON document.",
    "Walkthrough": "A recording rendered as a single offline HTML page with a stepper.",
}

# The relaxation side of graph partitioning: the words the two-hot span walkthrough narrates in.
TWO_HOT_TERMS: dict[str, str] = {
    "2-hot vector": "A vector with exactly two non-zero coordinates, of opposite sign: a vertex "
    "pair written as a vector.",
    "Spanning set": "The n-by-r matrix V the optimizer moves; its span, not its columns, is what the "
    "relaxed objective sees.",
    "Collision measure": "R(v_j), the per-column reward for concentrating on two coordinates; a "
    "diagnostic, never the criterion a run is judged by.",
    "Roundability": "How far a column is from its nearest 2-hot vector, and so how much the "
    "rounded cut can exceed the relaxed objective.",
    "Rounded pair": "The two coordinates a column is rounded to — its largest and its smallest "
    "entry — the vertex pair the column stands for.",
    "Rounded cut": "Ê, the ratio-cut value of the partition the rounded pairs induce; what the "
    "relaxation actually delivers.",
    "Relaxed objective": "E*, the objective at the unrounded V through the exact projector; never "
    "the ridge training loss.",
    "Spectral floor": "Σλ, the sum of the K smallest Laplacian eigenvalues: the lower bound every "
    "E* and Ê is read against.",
    "Component count": "How many connected components the rounded pairs induce, against the K "
    "clusters that were asked for.",
}

GLOSSARY: dict[str, str] = {
    **SEARCH_TERMS,
    **NYSTROM_TERMS,
    **RECORDING_TERMS,
    **TWO_HOT_TERMS,
}
