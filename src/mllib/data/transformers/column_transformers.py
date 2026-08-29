"""Concrete transformers, extracted from the ad-click ``temp_*`` methods in ``DataOrchestrator``.

Each reproduces the original operation exactly (the equivalence test in ``tests/data`` checks the
frames the training baseline depends on), but as a fit/transform object that can be declared in
config and reused across projects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mllib.data.transformers.abstract_transformer import Transformer


class DropColumns(Transformer):
    """Remove columns that carry no signal (ids, names)."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=self.columns)


class OneHotEncode(Transformer):
    """One-hot encode categorical columns (first level dropped, missing values as their own level).

    ``fit`` records the encoded column set so ``transform`` yields the same columns for new data,
    filling levels unseen in the new frame with 0.
    """

    def __init__(self, columns: Sequence[str], as_boolean: bool = False):
        self.columns = list(columns)
        self.dtype = bool if as_boolean else float
        self.encoded_columns_: pd.Index | None = None

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(
            frame, columns=self.columns, drop_first=True, dummy_na=True, dtype=self.dtype
        )

    def fit(self, frame: pd.DataFrame) -> OneHotEncode:
        self.encoded_columns_ = self._encode(frame).columns
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = self._encode(frame)
        if self.encoded_columns_ is None:
            return encoded
        return encoded.reindex(columns=self.encoded_columns_, fill_value=self.dtype(0))


class StandardizeNumeric(Transformer):
    """Scale numeric columns to zero mean and unit variance (population std, missing values ignored)."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)
        self.scaler_ = StandardScaler()

    def fit(self, frame: pd.DataFrame) -> StandardizeNumeric:
        self.scaler_.fit(frame[self.columns])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[self.columns] = self.scaler_.transform(frame[self.columns])
        return out


class FillNaNWithMean(Transformer):
    """Replace missing values in numeric columns with the column mean learned at fit time."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)
        self.means_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> FillNaNWithMean:
        self.means_ = {col: float(frame[col].mean()) for col in self.columns}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in self.columns:
            out[col] = out[col].replace(np.nan, self.means_[col])
        return out


class ReplaceNaNWithString(Transformer):
    """Make missing values an explicit category (``"nan"``) so tree splits can use them."""

    def __init__(self, columns: Sequence[str], token: str = "nan"):
        self.columns = list(columns)
        self.token = token

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in self.columns:
            out[col] = out[col].replace(np.nan, self.token)
        return out


class BinByStdRanges(Transformer):
    """Bin a numeric column into intervals at mean ± {0, ½, 1, 2}·std, bounded by min and max.

    Edges are learned at fit time (sample std, as pandas computes it). Interior edges that fall
    outside [min, max] are dropped and duplicates collapsed, so a tight or small column bins into
    fewer intervals instead of failing. Values equal to the minimum fall outside the right-inclusive
    first interval and become NaN — the original behaviour, kept so downstream one-hot encoding
    (``dummy_na=True``) sees the same levels.
    """

    def __init__(self, column: str):
        self.column = column
        self.edges_: list[float] = []

    def fit(self, frame: pd.DataFrame) -> BinByStdRanges:
        col = frame[self.column]
        std, mean, low, high = col.std(), col.mean(), col.min(), col.max()
        interior = [mean + k * std for k in (-2, -1, -0.5, 0, 0.5, 1, 2)]
        self.edges_ = sorted({low, high, *(e for e in interior if low < e < high)})
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[self.column] = pd.cut(out[self.column], bins=self.edges_)
        return out
