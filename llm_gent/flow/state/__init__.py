# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""State — package split into runtime, serialization, and CAS layers.

Submodule layout:

- :mod:`.base` — :class:`State`, :class:`StateData`, :class:`StateDataclass`,
  :class:`StateFactory`, :class:`TypeStateFactory`.
- :mod:`.serialization` — module-level :data:`state_converter` and its
  build hooks.
- :mod:`.cas` — content-addressed :class:`Blob`, :class:`Tree`,
  :class:`Commit` object model. Not re-exported here — CAS-facing
  consumers import from ``llm_gent.flow.state.cas`` explicitly.

Public re-exports below preserve the pre-split import surface:
``from llm_gent.flow.state import <name>`` resolves for every ``<name>``
that previously lived at the module top.
"""

from __future__ import annotations

from .base import (
    State,
    StateData,
    StateDataclass,
    StateFactory,
    T,
    T_co,
    TypeStateFactory,
)
from .serialization import state_converter


def serialize_state_data(data: object) -> object:
    """Return a JSON-compatible view of ``data`` for checkpointing.

    Plain dicts pass through as-is (the framework does not deep-copy
    — the store implementation owns durability). Objects satisfying
    :class:`StateData` are converted via ``to_dict()``. ``None`` also
    passes through (a payload that never carried structured data).
    Anything else raises :class:`TypeError` at the save site with a
    pointer to the contract.
    """
    if data is None or isinstance(data, dict):
        return data
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(
        f"cannot checkpoint state.data of type {type(data).__name__} — "
        f"payload must be a plain dict or satisfy StateData (to_dict/from_dict)"
    )


__all__ = [
    "State",
    "StateData",
    "StateDataclass",
    "StateFactory",
    "T",
    "T_co",
    "TypeStateFactory",
    "serialize_state_data",
    "state_converter",
]
