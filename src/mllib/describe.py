"""Introspected component descriptors.

Every MlLib component used to carry a hand-typed ``metadata`` dict that drifted from the code
(copy-pasted names, stale descriptions). ``describe`` derives the same information from the class
itself, so it cannot drift: the name is the class name, the doc is the docstring, the parameters are
the constructor's signature. This is the descriptor the tradePlatform plugin seam will introspect
at the boundary (BL-19).
"""

from __future__ import annotations

import inspect
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
