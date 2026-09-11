"""The two-hot span optimizer stack: the torch implementations of its math objects (D-35 (6)).

The torch implementations of the relaxation's math objects live here, so the optional dependency
group (D-31) has one boundary; the loop itself joins in slice 4 of the BL-48 plan. The abstract
contracts they implement stay array-agnostic in `mllib.math`.
"""
