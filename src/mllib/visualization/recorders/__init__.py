"""Per-problem recorders: the children that know what one recorded moment of a run contains.

One module per problem. Each holds a ``Frame`` subclass naming that problem's fields and a recorder
that converts the raw objects an engine hands it into plain Python and writes the caption.
"""
