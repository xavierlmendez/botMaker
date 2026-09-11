"""Observation as an injected collaborator: the recorder an algorithm hands its state to.

An algorithm that wants to be watched has three ways to offer it. It can return more — which grows
the result type until it is a scrapbook. It can log — which turns a proof into prose nobody can
replay. Or it can accept a collaborator and hand it the objects it already holds at the moments its
state advances. Only the third leaves the algorithm's own contract alone, which is why it is the one
here: a ``Recorder`` is a constructor argument, off by default, and the engine's single call site is
guarded by ``if self.recorder.enabled:`` at each of the moments that matter — for a search, every
state it advances through and the goal it stops at.

The cost when observation is off is one attribute on the instance and one attribute read per
expansion — the ``NullRecorder``'s class-level ``enabled = False``. Nothing inside the guard runs,
so the search expands the same states in the same order and both behavioural baselines stay
byte-identical. A ``NullRecorder`` that is nevertheless handed a frame means a guard is missing
somewhere, which is a bug and not a thing to swallow: it raises.

A recording is never part of a result. ``SearchResult`` carries what the search *paid*, how it was
*set up* and what it *proved* (D-28, amended 2026-09-07); frames are none of those three, so they
stay on the recorder the caller passed in. The caller who wants both keeps its own reference — the
recorder it constructed — and the search's return value is unchanged for everyone else (D-32).

``math`` never imports ``visualization``. The engine is typed against ``AbstractSearchRecorder``,
declared here; the per-problem child that knows how to extract and caption an expansion lives in
``mllib.visualization.recorders`` and depends downward only. That is also why the child does the
extracting: the engine passes the raw objects it already has — states, bounds, the heap — and never
learns which of them a reader wants to see.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Frame:
    """One recorded moment of a run: for a search one expansion, for an optimizer one step.

    ``index`` is the frame's position in the run, counted from zero, and ``caption`` is a sentence a
    reader can follow without the code beside them. Children add typed fields for their own problem
    and extend ``to_dict`` explicitly rather than through ``dataclasses.asdict``: the explicit dict
    is where numpy scalars are converted to plain Python, and a recursive helper would let one
    through unnoticed and only fail later, at ``json.dumps``, in a different module.
    """

    index: int
    caption: str

    def to_dict(self) -> dict[str, object]:
        """The frame as JSON-serialisable plain Python; children extend this dict."""
        return {"index": self.index, "caption": self.caption}


class AbstractRecorder(ABC):
    """A collaborator an algorithm hands its state to at the moments that state advances.

    ``enabled`` is a class attribute so the engine's guard is one attribute read, and so a
    ``NullRecorder`` is recognisably off without being asked. ``frames`` is dense and ordered:
    ``record`` refuses any frame whose ``index`` is not the position it lands in, which turns a
    dropped or duplicated frame into a failure at the moment it happens instead of an off-by-one in
    a walkthrough. ``metadata`` is free-form room for what the run was, filled by whoever built the
    recorder.

    ``describe_result`` is the one thing children must implement beyond recording. It converts the
    algorithm's own result to plain Python, so that a later slice can assemble a recording from
    frames plus result without the recorder ever importing the visualization package that holds the
    recording type.
    """

    enabled: bool = True

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.metadata: dict[str, object] = {}

    def record(self, frame: Frame) -> None:
        """Append a frame, insisting that the run's frames stay dense and in order.

        A raise rather than an ``assert``: the check is the contract, not a development aid, and
        ``python -O`` would strip an assertion and let a walkthrough silently mis-number itself.
        """
        if frame.index != len(self.frames):
            raise ValueError(
                f"frame index {frame.index} is not the next position {len(self.frames)}: "
                "a frame was dropped, duplicated or recorded out of order."
            )
        self.frames.append(frame)

    def frame_dicts(self) -> list[dict[str, object]]:
        """Every recorded frame as plain Python, in order."""
        return [frame.to_dict() for frame in self.frames]

    @abstractmethod
    def describe_result(self, result: object) -> dict[str, object]:
        """The algorithm's result as JSON-serialisable plain Python."""
        raise NotImplementedError


class AbstractSearchRecorder(AbstractRecorder):
    """The recorder shape a best-first search speaks to: one call per expansion.

    Declared here, in ``math``, so the engine can type-hint its collaborator without importing
    anything above it. The signature is the raw material of an expansion and nothing more — the
    state that was expanded and its bound, how many expansions have happened, the priced children
    with whether each entered the frontier, the whole frontier as (bound, state) pairs, and the size
    of the expanded set. Variants add what they alone track through ``extras``.

    A run has two kinds of moment, not one: the states it advances through, and the goal it stops
    at. A walkthrough that ended on the last expansion would stop one step before the answer, so
    ``record_goal`` is its own call — the goal is popped, never expanded, and prices no children.
    """

    @abstractmethod
    def record_expansion(
        self,
        expanded_state: object,
        bound: float,
        expansions: int,
        children: Sequence[tuple[object, float, bool]],
        frontier: Sequence[tuple[float, object]],
        expanded_set_size: int,
        **extras: object,
    ) -> None:
        """Record the expansion that has just finished pricing and pushing its children."""
        raise NotImplementedError

    @abstractmethod
    def record_goal(
        self,
        state: object,
        cost: float,
        expansions: int,
        frontier: Sequence[tuple[float, object]],
        expanded_set_size: int,
        **extras: object,
    ) -> None:
        """Record the goal pop the search is about to return on: the run's last moment."""
        raise NotImplementedError


class NullRecorder(AbstractSearchRecorder):
    """The recorder that is off: the default, and the one every engine holds when nobody watches.

    Every recording method raises. A null recorder receiving a frame can only mean the engine called
    it without checking ``enabled``, and a silent no-op would hide that guard's absence behind a
    working search — the exact bug the guard exists to prevent.
    """

    enabled = False

    def record(self, frame: Frame) -> None:
        raise RuntimeError(
            "NullRecorder.record was called: observation is off, so the engine's "
            "'if self.recorder.enabled' guard is missing."
        )

    def record_expansion(
        self,
        expanded_state: object,
        bound: float,
        expansions: int,
        children: Sequence[tuple[object, float, bool]],
        frontier: Sequence[tuple[float, object]],
        expanded_set_size: int,
        **extras: object,
    ) -> None:
        raise RuntimeError(
            "NullRecorder.record_expansion was called: observation is off, so the engine's "
            "'if self.recorder.enabled' guard is missing."
        )

    def record_goal(
        self,
        state: object,
        cost: float,
        expansions: int,
        frontier: Sequence[tuple[float, object]],
        expanded_set_size: int,
        **extras: object,
    ) -> None:
        raise RuntimeError(
            "NullRecorder.record_goal was called: observation is off, so the engine's "
            "'if self.recorder.enabled' guard is missing."
        )

    def record_visit(
        self,
        node_id: int,
        depth: int,
        queue_or_stack: Sequence[tuple[int, int]],
        visited: Sequence[int],
        traversal_order: Sequence[int],
        **extras: object,
    ) -> None:
        raise RuntimeError(
            "NullRecorder.record_visit was called: observation is off, so the engine's "
            "'if self.recorder.enabled' guard is missing."
        )

    def record_traversal_end(
        self,
        traversal_order: Sequence[int],
        found: bool,
        **extras: object,
    ) -> None:
        raise RuntimeError(
            "NullRecorder.record_traversal_end was called: observation is off, so the engine's "
            "'if self.recorder.enabled' guard is missing."
        )

    def record_step(
        self,
        step: int,
        training_loss: float,
        spanning_set: object,
        **extras: object,
    ) -> None:
        raise RuntimeError(
            "NullRecorder.record_step was called: observation is off, so the engine's "
            "'if recorder.enabled' guard is missing."
        )

    def record_end(self, run: object, **extras: object) -> None:
        raise RuntimeError(
            "NullRecorder.record_end was called: observation is off, so the engine's "
            "'if recorder.enabled' guard is missing."
        )

    def describe_result(self, result: object) -> dict[str, object]:
        return {}


class AbstractTraversalRecorder(AbstractRecorder):
    """The recorder shape a graph traversal speaks to: one call per visit, one when it stops.

    A best-first search advances by *expanding* a state and pricing its children; a traversal
    advances by *visiting* a node it had already decided to reach. The two are different moments
    with different raw material — a traversal has no bounds, no priced children and no frontier, it
    has a pending queue or stack, a visited set and the order it has walked so far — so they are
    different contracts rather than one contract with half its arguments unused
    (``AbstractSearchRecorder`` above is the other).

    ``queue_or_stack`` is deliberately one argument under one name. Breadth-first search pends its
    nodes in a queue and depth-first search in a stack, and that difference is the *only* one
    between them; a recorder that took ``queue`` from one and ``stack`` from the other would need
    two call sites to record one idea, and the walkthrough could no longer put the two runs side by
    side. What discipline the container has is visible in the recorded snapshot itself — the order
    the pairs come out in — and is named by the recording's ``algorithm``, not by an argument.

    ``record_traversal_end`` is its own moment for the same reason ``record_goal`` is: a traversal
    that stopped because it found its target and one that stopped because it ran out of nodes end
    on the same last visit, and only a frame recorded *after* that visit can say which happened.

    It is also its own *name*. ``AbstractStepRecorder`` below ends a run with ``record_end(run)``,
    and the two contracts meet on one class: ``NullRecorder`` implements every recording method in
    the module so that a missing guard raises wherever it happens (and is registered against this
    ABC, since it is declared above it). Two end-of-run methods with the same name and different
    signatures cannot both live there — one would silently shadow the other, and ``register`` binds
    nothing, so nothing would catch it. D-33's rule that each shape gets its own contract therefore
    reaches the method names too: a traversal ends at ``record_traversal_end``, a step run at
    ``record_end``, and the null recorder can refuse both.
    """

    @abstractmethod
    def record_visit(
        self,
        node_id: int,
        depth: int,
        queue_or_stack: Sequence[tuple[int, int]],
        visited: Sequence[int],
        traversal_order: Sequence[int],
        **extras: object,
    ) -> None:
        """Record the node just added to the traversal order, with what is pending behind it."""
        raise NotImplementedError

    @abstractmethod
    def record_traversal_end(
        self,
        traversal_order: Sequence[int],
        found: bool,
        **extras: object,
    ) -> None:
        """Record the moment the traversal returns, saying whether it stopped on its target."""
        raise NotImplementedError


# ``NullRecorder`` is declared above this class, so it cannot list it as a base; it nevertheless
# implements both methods above (loudly, like every other recording method it has). Registering it
# as a virtual subclass is what makes that true to ``isinstance`` as well, so an engine that type
# hints its collaborator as an ``AbstractTraversalRecorder`` is not lying when it holds the null
# one.
AbstractTraversalRecorder.register(NullRecorder)


class AbstractStepRecorder(AbstractRecorder):
    """The recorder shape an iterative optimizer speaks to: one call per step, then one at the end.

    A search advances by expanding a state; an optimizer advances by taking a step. The two moments
    carry nothing in common, so they are two contracts rather than one widened one — an engine that
    accepted either would have to be told which it was holding, and a recorder that implemented
    both would implement half of each.

    ``record_step`` is called once per completed step, after the step's own checks have passed, with
    exactly what the loop already holds at that instant: the step number, the value of the
    *training* loss it just took a gradient of, and the current iterate. The iterate is handed over
    raw and whole; the engine computes nothing for the recorder's sake, which is what keeps a
    watched run identical to an unwatched one beyond the cost of the guard. Anything derived from
    the iterate — a residual, a rounding, a diagnostic per column — is the child's work, in
    ``visualization``, where the reader's questions live.

    ``record_end`` is the run's last moment. A walkthrough that stopped at the final step would stop
    just before the numbers the run actually reports, which are recomputed once after the loop; the
    end call is where those land.
    """

    @abstractmethod
    def record_step(
        self,
        step: int,
        training_loss: float,
        spanning_set: object,
        **extras: object,
    ) -> None:
        """Record the step that has just completed and passed its checks."""
        raise NotImplementedError

    @abstractmethod
    def record_end(self, run: object, **extras: object) -> None:
        """Record the assembled run: the last moment, holding the numbers the run reports."""
        raise NotImplementedError


# ``NullRecorder`` implements both contracts but is declared above this one, so it cannot name it as
# a base. Registering it makes the off recorder an ``AbstractStepRecorder`` for ``isinstance`` too,
# which is what an optimizer that type-hints its collaborator against this class is entitled to
# assume of the default it holds. The alternative — a second null recorder, one per contract —
# would give the library two objects meaning "not watching", and a caller would have to know which
# engine it was about to pass one to.
AbstractStepRecorder.register(NullRecorder)
