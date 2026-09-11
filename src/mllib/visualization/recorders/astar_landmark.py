"""The recorder for an A* landmark search: one frame per expansion, captioned.

The engine calls ``record_expansion`` with the objects it already holds and nothing shaped for a
reader. Everything that makes those objects readable happens here: the conversion to plain Python,
the ordering of the frontier, and the sentence that says what the search just did.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mllib.math.algorithms.a_star_search import SearchResult
from mllib.math.recorder import AbstractSearchRecorder, Frame

# A state is an ascending tuple of column indices (``NystromLandmarkProblem``); a priced child is
# that state, its bound, and whether it entered the frontier.
type LandmarkState = tuple[int, ...]
type PricedChild = tuple[LandmarkState, float, bool]
type FrontierEntry = tuple[float, LandmarkState]

CAPTION_PRECISION = 4  # decimals on a bound in a caption; the raw value rides on the frame


def _state(state: object) -> LandmarkState:
    """A search state as a tuple of plain ``int``, whatever integer type it arrived as."""
    return tuple(int(index) for index in state)  # type: ignore[union-attr]


def _bound(value: object) -> float:
    """A bound as a plain ``float``: a numpy scalar would serialise nowhere."""
    return float(value)  # type: ignore[arg-type]


def _plain(value: object) -> object:
    """Any settings value as JSON-serialisable plain Python.

    An engine's ``configuration`` is read off the instance, so a caller who built it from numpy
    (a tolerance scaled by ``np.trace``, say) leaves a ``np.float64`` on it. That serialises
    nowhere, and the failure would surface in slice 2 when a recording is written rather than here,
    so every value is converted on the way out.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    return float(value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AStarFrame(Frame):
    """One expansion of an A* landmark search, in plain Python.

    ``expanded_state`` is the state the search popped and ``bound`` its admissible lower bound.
    ``children`` are the states priced from it, each with its bound and whether it entered the
    frontier — a child that did not was pruned by a variant, and ``extras`` says by which mechanism.
    ``frontier`` is the whole frontier after the push, sorted by bound and then by state so two runs
    that hold the same frontier record the same list whatever order the heap happens to be in;
    ``frontier_size`` is its length, stored so a reader of the dict need not count. ``expansions``
    is how many states have been expanded including this one, and ``expanded_set_size`` how many
    distinct states the expanded set holds.

    ``goal`` marks the run's last frame, the goal pop the search returned on. One frame type serves
    both moments so that the walkthrough's view reads one dict shape throughout: a goal frame has
    ``children = ()`` because a goal is popped and never expanded, and its ``bound`` is the goal's
    exact cost — at a goal state the bound *is* the objective (D-23), which is what lets the search
    prove anything at all.
    """

    expanded_state: LandmarkState
    bound: float
    expansions: int
    children: tuple[PricedChild, ...]
    frontier: tuple[FrontierEntry, ...]
    frontier_size: int
    expanded_set_size: int
    goal: bool = False
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """The frame as JSON-serialisable plain Python, tuples flattened to lists.

        ``Frame.to_dict`` is named rather than reached through ``super()``: ``slots=True`` rebuilds
        the class after the method's ``__class__`` cell is bound, so a zero-argument ``super()``
        here resolves against the discarded class and raises.
        """
        return {
            **Frame.to_dict(self),
            "expanded_state": list(self.expanded_state),
            "bound": self.bound,
            "expansions": self.expansions,
            "children": [[list(state), bound, pushed] for state, bound, pushed in self.children],
            "frontier": [[bound, list(state)] for bound, state in self.frontier],
            "frontier_size": self.frontier_size,
            "expanded_set_size": self.expanded_set_size,
            "goal": self.goal,
            "extras": dict(self.extras),
        }


class AStarRecorder(AbstractSearchRecorder):
    """Builds an ``AStarFrame`` per expansion and describes the search's result.

    The engine passes raw objects at each of its two moments — every expansion, then the goal it
    returns on — and this class decides which of them a reader needs and says so in a sentence.
    Both produce an ``AStarFrame``, so the frames of a run are one shape from first to last.
    Variants add their own state through ``extras`` — the pruned engine what it declined to
    store and the incumbent it holds, which the caption folds in when it is there — so the
    recorder works unchanged against exact, pruned and anytime A*.
    """

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
        priced = tuple(
            (_state(state), _bound(child_bound), bool(pushed))
            for state, child_bound, pushed in children
        )
        snapshot = self._sorted_frontier(frontier)
        plain_extras = self._plain_extras(extras)
        index = len(self.frames)
        self.record(
            AStarFrame(
                index=index,
                caption=self._caption(
                    _state(expanded_state),
                    _bound(bound),
                    int(expansions),
                    priced,
                    len(snapshot),
                    plain_extras,
                ),
                expanded_state=_state(expanded_state),
                bound=_bound(bound),
                expansions=int(expansions),
                children=priced,
                frontier=snapshot,
                frontier_size=len(snapshot),
                expanded_set_size=int(expanded_set_size),
                extras=plain_extras,
            )
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
        goal_state = _state(state)
        goal_cost = _bound(cost)
        snapshot = self._sorted_frontier(frontier)
        # ``pruned`` names what the *previous* expansion declined to store; a goal priced nothing,
        # so repeating it here would caption a moment that did not happen.
        plain_extras = {
            name: value for name, value in self._plain_extras(extras).items() if name != "pruned"
        }
        self.record(
            AStarFrame(
                index=len(self.frames),
                caption=self._goal_caption(
                    goal_state, goal_cost, int(expansions), len(snapshot), plain_extras
                ),
                expanded_state=goal_state,
                bound=goal_cost,
                expansions=int(expansions),
                children=(),
                frontier=snapshot,
                frontier_size=len(snapshot),
                expanded_set_size=int(expanded_set_size),
                goal=True,
                extras=plain_extras,
            )
        )

    def describe_result(self, result: SearchResult) -> dict[str, object]:
        """The search's result as plain Python: what it found, paid, was set up with and proved."""
        configuration = result.engine_configuration
        return {
            "state": None if result.state is None else list(_state(result.state)),
            "cost": _bound(result.cost),
            "optimal": bool(result.optimal),
            "nodes_expanded": int(result.nodes_expanded),
            "frontier_peak": (None if result.frontier_peak is None else int(result.frontier_peak)),
            "engine_configuration": (
                None
                if configuration is None
                else {str(name): _plain(value) for name, value in configuration.items()}
            ),
            "certified_gap": (
                None if result.certified_gap is None else _bound(result.certified_gap)
            ),
        }

    @staticmethod
    def _goal_caption(
        goal_state: LandmarkState,
        goal_cost: float,
        expansions: int,
        frontier_size: int,
        extras: dict[str, object],
    ) -> str:
        """The run's closing sentence: the goal popped, or the incumbent that superseded it."""
        superseded = extras.get("superseded_goal")
        if superseded is not None:
            superseded_cost = float(extras.get("superseded_cost", 0.0))  # type: ignore[arg-type]
            return (
                f"Returned the held incumbent {goal_state} at cost "
                f"{goal_cost:.{CAPTION_PRECISION}f}; the popped goal {tuple(superseded)} "  # type: ignore[arg-type]
                f"at cost {superseded_cost:.{CAPTION_PRECISION}f} was within the tie tolerance "
                f"and set aside."
            )
        return (
            f"Goal {goal_state} popped at cost {goal_cost:.{CAPTION_PRECISION}f} "
            f"after {expansions} expansions; {frontier_size} states left on the frontier."
        )

    @staticmethod
    def _sorted_frontier(
        frontier: Sequence[tuple[float, object]],
    ) -> tuple[FrontierEntry, ...]:
        """The frontier as plain Python, ordered by bound then state rather than by heap layout."""
        return tuple(
            sorted(
                ((_bound(entry_bound), _state(state)) for entry_bound, state in frontier),
                key=lambda entry: (entry[0], entry[1]),
            )
        )

    @staticmethod
    def _plain_extras(extras: dict[str, object]) -> dict[str, object]:
        """Variant state as plain Python: states become lists, the rest goes via ``_plain``."""
        plain: dict[str, object] = {}
        for name, value in extras.items():
            if name == "pruned":
                plain[name] = [
                    [list(_state(state)), _bound(child_bound), str(reason)]
                    for state, child_bound, reason in value  # type: ignore[union-attr]
                ]
            elif name in ("incumbent_state", "superseded_goal"):
                plain[name] = None if value is None else list(_state(value))
            else:
                plain[name] = _plain(value)
        return plain

    @staticmethod
    def _caption(
        expanded_state: LandmarkState,
        bound: float,
        expansions: int,
        children: tuple[PricedChild, ...],
        frontier_size: int,
        extras: dict[str, object],
    ) -> str:
        """One sentence a reader can follow: what was expanded and what it left behind."""
        pushed = sum(1 for _, _, was_pushed in children if was_pushed)
        clauses = [
            f"Expanded {expanded_state} at bound {bound:.{CAPTION_PRECISION}f}",
            f"priced {len(children)} children, pushed {pushed}",
        ]
        pruned = len(children) - pushed
        if pruned:
            clauses[-1] += f", pruned {pruned}"
        clauses.append(f"frontier now {frontier_size} states")
        incumbent = extras.get("incumbent")
        if incumbent is not None:
            incumbent_state = extras.get("incumbent_state")
            named = "" if incumbent_state is None else f" {tuple(incumbent_state)}"  # type: ignore[arg-type]
            # No engine tracks a certified gap mid-run — it is computed at the stop — so a
            # caption never claims one. See `_bounded_result` in the anytime variant.
            clauses.append(f"incumbent{named} at {float(incumbent):.{CAPTION_PRECISION}f}")
        clauses.append(f"{expansions} expansions so far")
        return "; ".join(clauses) + "."
