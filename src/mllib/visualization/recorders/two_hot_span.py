"""The recorder for a two-hot span rcut run: one frame per optimizer step, captioned.

The optimizer hands over three things per step — the step number, the training loss it just took a
gradient of, and V as it stands — and stops there. Everything a reader wants to see is derived
*here*: E\\* through the exact ``pinv`` projector, Ê through the rounded columns, R(v_j) per column,
the rounded pairs and the components they induce. That split is deliberate and is the whole reason
the engine stays fast when nobody watches: the derivations cost a pseudo-inverse each, which is a
price a training loop must never pay, and they are the reader's questions rather than the
optimizer's.

Two kinds of frame, for the same reason. A full frame carries every derived number and V itself; a
light frame carries the step and its training loss and nothing else. A run of three hundred steps
wants a curve at every step and a picture at a few, and ``frame_every`` is where a caller says how
few. Both kinds are the same dict shape — the light one's derived fields are ``null`` — so a view
reads one shape from first frame to last.

One naming rule is load-bearing throughout. ``training_loss`` is the *ridge* training loss
``E_ridge(V) - λ Σ_j R(v_j)`` (D-31): it is computed through the regularized projector and it
carries the collision reward, so it is neither E\\* nor Ê nor comparable to either. It is never
labelled "E" in a caption, a field name or a dict key, here or anywhere downstream.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from mllib.math.graph import two_hot_span_problem as problem
from mllib.math.projector import ExactProjector
from mllib.math.recorder import AbstractStepRecorder, Frame

if TYPE_CHECKING:
    # Only for the annotations below. The module that defines the run imports torch, which is an
    # optional group (D-31), and a recorder that pulled it in at import time would make every
    # reader of a recording pay for it.
    from mllib.math.algorithms.two_hot_span_optimizer import TwoHotSpanRun

CAPTION_PRECISION = 4  # decimals on a number in a caption; the exact value rides on the frame

type Pair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TwoHotSpanFrame(Frame):
    """One optimizer step of a two-hot span run, in plain Python.

    ``step`` is the step number, counted from zero, and ``training_loss`` the ridge training loss at
    that step — the value in ``TwoHotSpanRun.loss_history[step]``, and never E\\* or Ê.

    ``full`` says whether the derived fields are there. On a full frame ``spanning_set`` is V as an
    n-by-r nested list of exact doubles, ``collision_measures`` is R(v_j) per column,
    ``rounded_pairs`` are the (argmax, argmin) pairs the columns round to — the edges of the pair
    graph, shorter than r when a column rounds to zero — ``labels`` are the pair graph's components
    and ``component_count`` how many there are. ``relaxed_objective`` is E\\*, ``rounded_cut`` is Ê,
    and the two differences are the ones the harness reports: Ê - E\\* is what rounding cost, and
    Ê - Σλ is how far the answer sits above the spectral floor. On a light frame every one of them
    is ``None``.

    ``end`` marks the run's last frame, added after the loop from the run's own reported fields
    rather than recomputed — so a walkthrough ends on the numbers the run actually returns.
    """

    step: int
    training_loss: float
    full: bool
    end: bool = False
    spanning_set: tuple[tuple[float, ...], ...] | None = None
    collision_measures: tuple[float, ...] | None = None
    rounded_pairs: tuple[Pair, ...] | None = None
    labels: tuple[int, ...] | None = None
    component_count: int | None = None
    relaxed_objective: float | None = None
    rounded_cut: float | None = None
    rounded_cut_minus_relaxed: float | None = None
    rounded_cut_minus_floor: float | None = None

    def to_dict(self) -> dict[str, object]:
        """The frame as JSON-serialisable plain Python, tuples flattened to lists.

        ``Frame.to_dict`` is named rather than reached through ``super()``: ``slots=True`` rebuilds
        the class after the method's ``__class__`` cell is bound, so a zero-argument ``super()``
        here resolves against the discarded class and raises.
        """
        return {
            **Frame.to_dict(self),
            "step": self.step,
            "training_loss": self.training_loss,
            "full": self.full,
            "end": self.end,
            "spanning_set": (
                None if self.spanning_set is None else [list(row) for row in self.spanning_set]
            ),
            "collision_measures": (
                None if self.collision_measures is None else list(self.collision_measures)
            ),
            "rounded_pairs": (
                None
                if self.rounded_pairs is None
                else [[high, low] for high, low in self.rounded_pairs]
            ),
            "labels": None if self.labels is None else list(self.labels),
            "component_count": self.component_count,
            "relaxed_objective": self.relaxed_objective,
            "rounded_cut": self.rounded_cut,
            "rounded_cut_minus_relaxed": self.rounded_cut_minus_relaxed,
            "rounded_cut_minus_floor": self.rounded_cut_minus_floor,
        }


class TwoHotSpanRecorder(AbstractStepRecorder):
    """Builds a ``TwoHotSpanFrame`` per step and describes the run's result.

    Constructed with the incidence matrix the run is optimising against, the graph it came from and
    the cluster count — X for the projector residuals, the graph and K for Σλ and n. ``step_count``
    is how many steps the run will take, which is what lets the last step be a full frame whatever
    ``frame_every`` divides: a walkthrough whose final picture was four steps stale would show a V
    the run never reported on. ``spectral_floor`` may be passed when the caller has already paid for
    the eigendecomposition; otherwise it is computed once, here, and never again per frame.
    """

    def __init__(
        self,
        X: np.ndarray,
        graph: nx.Graph,
        cluster_count: int,
        step_count: int,
        *,
        frame_every: int = 1,
        spectral_floor: float | None = None,
    ) -> None:
        super().__init__()
        if frame_every < 1:
            raise ValueError(f"frame_every must be at least 1; got {frame_every}.")
        self.X = np.asarray(X, dtype=float)
        self.graph = graph
        self.cluster_count = int(cluster_count)
        self.step_count = int(step_count)
        self.frame_every = int(frame_every)
        self.node_count = problem.node_count(graph)
        self.spectral_floor = (
            problem.spectral_floor(graph, self.cluster_count)
            if spectral_floor is None
            else float(spectral_floor)
        )
        self.metadata = {
            "kind": "two_hot_span",
            "node_count": self.node_count,
            "cluster_count": self.cluster_count,
            "step_count": self.step_count,
            "frame_every": self.frame_every,
            "spectral_floor": self.spectral_floor,
        }

    # ----------------------------------------------------------------------------------------
    # Recording.
    # ----------------------------------------------------------------------------------------

    def is_full(self, step: int) -> bool:
        """Whether the step gets the derived numbers: every ``frame_every``, and the last one."""
        return step % self.frame_every == 0 or step == self.step_count - 1

    def record_step(
        self,
        step: int,
        training_loss: float,
        spanning_set: object,
        **extras: object,
    ) -> None:
        step = int(step)
        loss = float(training_loss)
        if not self.is_full(step):
            self.record(
                TwoHotSpanFrame(
                    index=len(self.frames),
                    caption=f"Step {step}: training loss {loss:.{CAPTION_PRECISION}f}.",
                    step=step,
                    training_loss=loss,
                    full=False,
                )
            )
            return

        current = np.asarray(spanning_set, dtype=float)
        measures = problem.collision_measures(current)
        pairs = problem.rounded_pairs(current)
        labels = problem.clustering_from_pairs(self.node_count, pairs)
        relaxed = ExactProjector().residual(self.X, current)
        cut = problem.rounded_cut(self.X, current)
        self.record(
            self._full_frame(
                step=step,
                training_loss=loss,
                caption=self._step_caption(step, loss, relaxed, cut, labels, measures),
                spanning_set=current,
                measures=measures,
                pairs=pairs,
                labels=labels,
                relaxed=relaxed,
                cut=cut,
                end=False,
            )
        )

    def record_end(self, run: TwoHotSpanRun, **extras: object) -> None:
        """The run's last frame: its own reported numbers, not a recomputation of them."""
        steps_taken = int(np.asarray(run.loss_history).size)
        # A run of zero steps has no training loss to restate; the field is 0.0 and the caption
        # says how many steps there were, which is the honest reading of an empty loss history.
        loss = float(run.loss_history[-1]) if steps_taken else 0.0
        labels = np.asarray(run.labels)
        measures = np.asarray(run.collision_measures, dtype=float)
        relaxed = float(run.relaxed_objective)
        cut = float(run.rounded_cut)
        spanning_set = np.asarray(run.spanning_set, dtype=float)
        self.record(
            self._full_frame(
                step=max(steps_taken - 1, 0),
                training_loss=loss,
                caption=self._end_caption(steps_taken, relaxed, cut, labels, measures),
                spanning_set=spanning_set,
                measures=measures,
                pairs=problem.rounded_pairs(spanning_set),
                labels=labels,
                relaxed=relaxed,
                cut=cut,
                end=True,
            )
        )

    def _full_frame(
        self,
        *,
        step: int,
        training_loss: float,
        caption: str,
        spanning_set: np.ndarray,
        measures: np.ndarray,
        pairs: list[Pair],
        labels: np.ndarray,
        relaxed: float,
        cut: float,
        end: bool,
    ) -> TwoHotSpanFrame:
        """One frame carrying every derived number, with numpy scalars converted on the way in."""
        return TwoHotSpanFrame(
            index=len(self.frames),
            caption=caption,
            step=step,
            training_loss=training_loss,
            full=True,
            end=end,
            spanning_set=tuple(tuple(float(value) for value in row) for row in spanning_set),
            collision_measures=tuple(float(value) for value in measures),
            rounded_pairs=tuple((int(high), int(low)) for high, low in pairs),
            labels=tuple(int(label) for label in labels),
            component_count=int(np.unique(labels).size),
            relaxed_objective=float(relaxed),
            rounded_cut=float(cut),
            rounded_cut_minus_relaxed=float(cut) - float(relaxed),
            rounded_cut_minus_floor=float(cut) - self.spectral_floor,
        )

    # ----------------------------------------------------------------------------------------
    # Captions and the result.
    # ----------------------------------------------------------------------------------------

    def _step_caption(
        self,
        step: int,
        training_loss: float,
        relaxed: float,
        cut: float,
        labels: np.ndarray,
        measures: np.ndarray,
    ) -> str:
        """One sentence: what the step cost to train, and what its V would be worth if stopped."""
        return (
            f"Step {step}: training loss {training_loss:.{CAPTION_PRECISION}f}; "
            + self._numbers(relaxed, cut, labels, measures)
        )

    def _end_caption(
        self, steps_taken: int, relaxed: float, cut: float, labels: np.ndarray, measures: np.ndarray
    ) -> str:
        """One sentence for the run's own answer, the numbers it reports rather than a step's."""
        plural = "" if steps_taken == 1 else "s"
        return f"Run ended after {steps_taken} step{plural}: " + self._numbers(
            relaxed, cut, labels, measures
        )

    def _numbers(self, relaxed: float, cut: float, labels: np.ndarray, measures: np.ndarray) -> str:
        """The clause every full frame's caption ends on: E\\*, its floor, Ê, components, R(v_j)."""
        components = int(np.unique(labels).size)
        plural = "" if components == 1 else "s"
        return (
            f"E* {relaxed:.{CAPTION_PRECISION}f} "
            f"(floor {self.spectral_floor:.{CAPTION_PRECISION}f}), "
            f"Ê {cut:.{CAPTION_PRECISION}f}, {components} component{plural}; "
            f"mean R(v_j) {float(np.mean(measures)):.{CAPTION_PRECISION}f}, "
            f"min {float(np.min(measures)):.{CAPTION_PRECISION}f}."
        )

    def describe_result(self, result: TwoHotSpanRun) -> dict[str, object]:
        """The run's reported fields as plain Python, plus Σλ and the config it ran under."""
        labels = np.asarray(result.labels)
        relaxed = float(result.relaxed_objective)
        cut = float(result.rounded_cut)
        losses = np.asarray(result.loss_history, dtype=float)
        return {
            "spanning_set": [
                [float(value) for value in row]
                for row in np.asarray(result.spanning_set, dtype=float)
            ],
            "loss_history": [float(value) for value in losses],
            "final_training_loss": float(losses[-1]) if losses.size else 0.0,
            "relaxed_objective": relaxed,
            "rounded_cut": cut,
            "rounded_cut_minus_relaxed": cut - relaxed,
            "rounded_cut_minus_floor": cut - self.spectral_floor,
            "collision_measures": [
                float(value) for value in np.asarray(result.collision_measures, dtype=float)
            ],
            "labels": [int(label) for label in labels],
            "component_count": int(np.unique(labels).size),
            "max_zero_sum_violation": float(result.max_zero_sum_violation),
            "spectral_floor": self.spectral_floor,
            "config": dataclasses.asdict(result.config),
        }
