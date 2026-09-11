"""Turning a run into something a reader can step through, one frame at a time.

This package sits above ``ml`` and ``math`` and nothing below it imports it. The dependency runs one
way for the same reason it does everywhere else in the library: an algorithm that knew how its
frontier should be drawn would be an algorithm you could not change without changing a picture. So
``math`` declares only the shape of a recorder (``mllib.math.recorder``) and the per-problem child
that knows what an expansion *means* — which fields a reader wants, how to caption them — lives
here, in ``recorders/``.

The layering also decides where conversion happens. Everything crossing out of ``math`` is raw:
states, bounds, the heap as it stands. Everything this package stores is plain Python — ints,
floats, lists, strings — so that a frame is JSON by construction and no numpy scalar reaches a
document that has to survive being written to disk and read back somewhere else.
"""
