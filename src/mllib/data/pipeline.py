"""Declarative transformer pipelines (R1, BL-09).

A project declares, in JSON, which transformers run on which frame and in what order; the pipeline
resolves each step by class name against ``mllib.data.transformers`` and runs them in sequence.
Adding a transformation to a project is a config edit, not a code edit.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from mllib.data import transformers as transformer_registry
from mllib.data.transformers import Transformer


class TransformerPipeline(Transformer):
    """Run transformers in order; ``fit`` fits each on the output of the previous one."""

    def __init__(self, steps: Iterable[Transformer] = ()):
        self.steps = list(steps)
        for step in self.steps:
            if not isinstance(step, Transformer):
                raise TypeError(f"pipeline steps must be Transformers, got {type(step).__name__}")

    def fit(self, frame: pd.DataFrame) -> TransformerPipeline:
        for step in self.steps:
            frame = step.fit_transform(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            frame = step.transform(frame)
        return frame

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            frame = step.fit_transform(frame)
        return frame

    @classmethod
    def from_config(cls, spec: Iterable[Mapping[str, Any]]) -> TransformerPipeline:
        """Build from ``[{"transformer": "OneHotEncode", "args": {...}}, ...]``.

        Names resolve against ``mllib.data.transformers``; anything that is not a ``Transformer``
        subclass there is rejected, so config cannot reach arbitrary code.
        """
        steps = []
        for step in spec:
            name = step["transformer"]
            transformer_cls = getattr(transformer_registry, name, None)
            if not (
                isinstance(transformer_cls, type)
                and issubclass(transformer_cls, Transformer)
                and not inspect.isabstract(transformer_cls)
            ):
                raise ValueError(f"unknown transformer {name!r} (see mllib.data.transformers)")
            steps.append(transformer_cls(**step.get("args", {})))
        return cls(steps)


class ProjectTransformations:
    """A project's transformer config: one pipeline per named frame plus the target column.

    File shape::

        {"target": "click",
         "frames": {"logisticReg": [{"transformer": "DropColumns", "args": {"columns": [...]}}, ...],
                    ...}}
    """

    def __init__(self, target: str, frames: Mapping[str, Iterable[Mapping[str, Any]]]):
        self.target = target
        self.pipelines = {
            name: TransformerPipeline.from_config(spec) for name, spec in frames.items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> ProjectTransformations:
        config = json.loads(Path(path).read_text())
        return cls(target=config["target"], frames=config["frames"])

    def frame_names(self) -> list[str]:
        return list(self.pipelines)
