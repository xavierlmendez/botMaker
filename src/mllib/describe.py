"""Introspected component descriptors.

Every MlLib component used to carry a hand-typed ``metadata`` dict that drifted from the code
(copy-pasted names, stale descriptions). ``describe`` derives the same information from the class
itself, so it cannot drift: the name is the class name, the doc is the docstring, the parameters are
the constructor's signature. This is the descriptor the tradePlatform plugin seam will introspect
at the boundary (BL-19).
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


def describe(obj: Any) -> dict[str, Any]:
    """Return a JSON-serialisable descriptor for a class or an instance.

    Keys: ``name`` (class name), ``module`` (dotted module), ``kind`` (the MlLib domain — ``math``,
    ``ml``, ``data`` — or the module for anything outside the package), ``doc`` (first line of the
    class docstring, ``""`` if none), ``params`` (constructor parameter names, ``self`` excluded) and
    ``signature`` (the constructor signature as text) and ``task_kind`` ('regression' /
    'classification' for components that declare one, else ``None``).
    """
    cls = obj if inspect.isclass(obj) else type(obj)
    try:
        signature = inspect.signature(cls.__init__)
        params = [p.name for p in signature.parameters.values() if p.name != "self"]
        signature_text = str(signature)
    except (TypeError, ValueError):  # builtins / C-implemented constructors
        params, signature_text = [], "()"
    module = cls.__module__
    parts = module.split(".")
    kind = parts[1] if parts[0] == "mllib" and len(parts) > 2 else module
    doc = inspect.getdoc(cls) or ""
    task_kind = getattr(cls, "task_kind", None)
    return {
        "name": cls.__name__,
        "module": module,
        "kind": kind,
        "doc": doc.splitlines()[0] if doc else "",
        "params": params,
        "signature": signature_text,
        "task_kind": task_kind.value if task_kind is not None else None,
    }


def configuration_of(obj: Any) -> dict[str, Any]:
    """The knobs of a math object as plain Python, its injected objects nested (D-35 (4)).

    Each constructor parameter of ``obj`` is read back off the instance by name. A number, a
    string, a boolean or ``None`` is recorded as it is; an enum by its value; another MlLib object
    by recursion; anything else — an array, a tensor, a graph — is data rather than a setting and
    is left out. The class name comes first so the record says which implementation ran.
    """
    record: dict[str, Any] = {"name": type(obj).__name__}
    for name in describe(obj)["params"]:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if isinstance(value, Enum):
            record[name] = value.value
        elif value is None or isinstance(value, bool | int | float | str):
            record[name] = value
        elif type(value).__module__.startswith("mllib."):
            record[name] = configuration_of(value)
    return record
