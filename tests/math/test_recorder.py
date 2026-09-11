"""The recorder contract: what "off" means, and what a recorder promises about its frames.

`mllib.math.recorder` holds no algorithm, so it is tested against a recorder written here rather
than against a search. The A* instrumentation is tested where it lives,
`tests/visualization/recorders/test_astar_landmark.py`.
"""

from dataclasses import dataclass

import pytest

from mllib.math.recorder import AbstractRecorder, Frame, NullRecorder


@dataclass(frozen=True, slots=True)
class CountFrame(Frame):
    """A frame with one field of its own, extended into the dict explicitly like a real child."""

    count: int

    def to_dict(self) -> dict[str, object]:
        return {**Frame.to_dict(self), "count": self.count}


class CountRecorder(AbstractRecorder):
    """A concrete recorder that records what it is given and describes a result as a count."""

    def describe_result(self, result: object) -> dict[str, object]:
        return {"frames": len(self.frames)}


def test_the_null_recorder_is_off():
    assert NullRecorder.enabled is False
    assert NullRecorder().enabled is False


def test_the_null_recorder_raises_when_it_is_handed_a_frame():
    with pytest.raises(RuntimeError, match="guard is missing"):
        NullRecorder().record(Frame(index=0, caption="anything."))


def test_the_null_recorder_raises_when_an_expansion_reaches_it():
    with pytest.raises(RuntimeError, match="guard is missing"):
        NullRecorder().record_expansion((), 0.0, 1, [], [], 1)


def test_the_null_recorder_raises_when_a_goal_pop_reaches_it():
    with pytest.raises(RuntimeError, match="guard is missing"):
        NullRecorder().record_goal((1, 2), 0.5, 3, [], 3)


def test_the_null_recorder_describes_no_result():
    assert NullRecorder().describe_result(object()) == {}


def test_a_concrete_recorder_is_on_by_default():
    assert CountRecorder().enabled is True


def test_each_recorder_starts_with_its_own_empty_frames():
    first, second = CountRecorder(), CountRecorder()
    first.record(CountFrame(index=0, caption="one.", count=1))
    assert first.frames != second.frames
    assert second.frames == []


def test_recording_frames_in_order_keeps_them_dense():
    recorder = CountRecorder()
    for index in range(3):
        recorder.record(CountFrame(index=index, caption=f"frame {index}.", count=index))
    assert [frame.index for frame in recorder.frames] == [0, 1, 2]


def test_recording_a_frame_out_of_order_fails_at_the_frame_that_did_it():
    # A raise, not an assertion: the density check is the contract and must survive ``python -O``.
    recorder = CountRecorder()
    recorder.record(CountFrame(index=0, caption="one.", count=1))
    with pytest.raises(ValueError, match="not the next position"):
        recorder.record(CountFrame(index=2, caption="skipped one.", count=2))
    assert len(recorder.frames) == 1


def test_frame_dicts_round_trip_the_frames_as_plain_python():
    recorder = CountRecorder()
    recorder.record(CountFrame(index=0, caption="one.", count=1))
    recorder.record(CountFrame(index=1, caption="two.", count=2))
    assert recorder.frame_dicts() == [
        {"index": 0, "caption": "one.", "count": 1},
        {"index": 1, "caption": "two.", "count": 2},
    ]


def test_a_recorder_without_describe_result_cannot_be_built():
    class Incomplete(AbstractRecorder):
        pass

    with pytest.raises(TypeError, match="describe_result"):
        Incomplete()
