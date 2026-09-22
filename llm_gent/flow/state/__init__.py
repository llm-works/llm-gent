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


__all__ = [
    "State",
    "StateData",
    "StateDataclass",
    "StateFactory",
    "T",
    "T_co",
    "TypeStateFactory",
    "state_converter",
]
