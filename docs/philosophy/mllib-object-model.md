# The object model of `mllib`

Written 2026-09-11, grilled with Xavier. Scope: the library under `src/mllib` and nothing else. A
front-end or a CRUD service would not be served by this document, and it does not try to serve them.

This is the abstract statement. It names no file, no class and no line of code, so that it stays
true while the tree moves. The concrete rules it produced are decision D-35; the contracts they
produced are in the architecture document; the vocabulary is the glossary. Pull this document into a
session when a new math object is being designed, or when a review is asking "should this be a
class". Do not pull it in to look something up.

## 1. What the library is for

Three purposes, in order: to learn a technique by implementing it by hand, to show professional
practice, and to be a reference brain that a later reader — usually the author, a year on — can open
at a concept and find it. Every rule below is judged by the third purpose. A structure that makes
the next variant cheap but hides the concept has failed; a structure that names the concept but
makes the next variant an edit to a loop has also failed.

The library has two families of algorithm and they are held to one shape. A search moves through
discrete states toward a certified goal; an optimizer moves continuous parameters down a training
loss. Both are runs: they are set up, they pay a cost, they deliver a result, and they may be
watched. Whatever is true of how one is composed is meant to be true of the other.

## 2. What a class is for

A class is a promise that there is more than one way to be this thing, or that being this thing
takes memory. A class is earned when one of three holds:

- it is **injected** into something else, so a caller needs an interface to write against;
- it has **two or more implementations**, so a reader needs one place to see what they share;
- it **carries state across calls**, so the thing has a lifetime and not just a value.

A concept that meets none of the three is a function whose name is the concept's name. The function
is still the concept's index entry; naming is not what the class was for. A class with one method,
one implementation and no state is a name pretending to be a structure, and it costs the reader a
file for nothing.

An interface is declared abstractly and fails at construction, not at first use. A concept the
library has named but never implemented is not an interface; it is intent, and intent lives in the
backlog.

## 3. Behaviour is injected, numbers are knobs

Two kinds of thing enter a run. A **math object** carries behaviour: a cost, a penalty, a projector,
a step rule, a problem, a recorder. A **knob** is one number or one name. The line between them is
the line between what is passed to a constructor and what is set on the thing that owns it.

A math object is passed in, never built inside the thing that uses it. The host is written once
against the interface and is not edited when a new implementation arrives. This is the rule that
makes a variant an addition rather than a change, and it is the rule most often broken in the name
of convenience: building the collaborator inside the loop because "there is only one".

A knob belongs to the object it parameterises, and its default sits on that object's concrete class.
Injecting a particular version of a concept is then two acts: instantiate it, pass it. A
configuration is not a place where every knob in a run is declared once; it is what a run reports
about how it was set up, assembled from the algorithm's own knobs and from those of every object
injected into it. A setting that produced a number and is absent from that number's record is the
failure this rule is written against.

A grid over knobs is a grid over constructors. Each cell builds fresh objects. Nothing is mutated
between cells, because a mutated instance is a run whose configuration has quietly changed.

## 4. One concept, one place, per arithmetic

Duplication is measured at the level of the concept, not the line. Two implementations of one
concept in two arithmetics — an exact form for reporting, a smoothed form for training — are one
concept with two members and share one interface and one name. The same arithmetic written twice,
once for each array library, is a duplicate and is removed. The array library is an implementation
detail of a concrete class, never a reason for a second concept.

Abstract bases are array-agnostic. A concrete implementation that needs an optional dependency
imports it, and such implementations live together so the boundary is one directory rather than a
scattering of guarded imports.

## 5. Composition roots

Concrete classes are named in exactly one kind of place: the script or harness that wires a run.
Such a place is a composition root. It may know everything; in exchange it consumes each algorithm
through the same run-and-result contract as every other caller, and it records the configuration on
every row it writes. A composition root is not a second implementation of the algorithm's loop, and
it is not where the algorithm's post-processing lives.

## 6. Results and observation

A result carries what a run paid, how it was set up, and what it delivered. It does not carry what
was seen along the way. Observation is a collaborator injected into the run, off by default, and its
frames stay on the collaborator. A run that stops early is still a run and delivers a result with a
stated reason; a traceback in place of a result is a hole in a grid, and a hole is not a
measurement.

## 7. Where a sentence lives

Each kind of sentence has one home, and the same sentence is not written twice.

| Sentence | Home |
|---|---|
| What a term *is* | the glossary |
| *Why* a rule holds | the decisions log |
| What a module holds and which decisions bind it | the module docstring, as one clause plus the decision's id |
| What a concept is and what its contract promises | the class docstring, which is also the description the library introspects |
| What a method requires and guarantees | the method docstring |
| A local *why* that no level above explains | an inline comment |
| Intent that is not yet code | the backlog, referenced from code by its id |

A docstring that restates a decision's reasoning will drift from the decision; it cites the id and
gives one clause. A comment that says what the next line does is deleted, because the line already
says it. A comment that contradicts a docstring is a bug in whichever of the two is wrong, and the
fix is to remove one of them, not to reconcile the wording.

## 8. The frameworks, and what wins

Two frameworks are cited here because their vocabulary is useful, not because they are obeyed.

**Cognitive load** (Zakirullin's handbook). Signed: a reader holds a few facts at once, so a file
answers one question; an abstraction earns its place by hiding something, not by naming it; a
shallow class is worse than a function. Rejected as an absolute: "prefer deep modules" when depth
would bury the concept a reader came to find. Section 2 is the reconciliation: depth is not a goal,
and neither is a class per idea; the three tests decide.

**Clean Code** (Martin). Signed: a name says what a thing is for; a function does one thing; a
comment is a failure to express in code, except when the thing to express is a why. Rejected: the
very-small-function rule as a standard, because in numeric code a formula split into five named
two-line helpers is harder to check against the paper than the formula written once.

Both are frameworks. The need of a particular implementation may differ from either, and when it
does, the representation that shows the intent of the function or class is the one chosen. That
choice is recorded as a decision when it has lasting consequences, and as a docstring clause when it
does not.

## 9. How this document is used

This document does not change on its own. A design question that it does not answer is put to a
session, grilled, and the answer becomes a decision in the log; if the answer reveals that a rule
here was wrong, the rule is corrected in the same pull request and the decision says so. Concrete
rules never enter this document; they enter the decisions log and cite it.
