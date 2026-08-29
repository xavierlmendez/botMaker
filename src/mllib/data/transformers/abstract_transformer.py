"""The transformer contract every data transformation implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Transformer(ABC):
    """A reusable, stateful column transformation.

    ``fit`` learns anything the transformation needs from training data (means, category sets, bin
    edges) and returns ``self``; ``transform`` applies it to a frame and returns a **new** frame —
    inputs are never mutated. ``fit_transform`` is the two in sequence. Instances are declared by
    class name in a project's transformer config and assembled into a pipeline (slice 6.2).
    """

    def fit(self, frame: pd.DataFrame) -> Transformer:
        return self

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
