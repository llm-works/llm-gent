# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resources — run-scoped objects a flow carries, checkpointed with the run.

- :class:`Resource` — the protocol (``snapshot()`` / ``restore(data)``,
  optional ``child(...)``); :class:`ResourceKey` — its typed handle;
  :func:`resource_method` — a fluent name of the app's own for one
  (:mod:`.base`).
- :data:`COST` — the cost tracker's key: the cost API is sugar over it
  (:mod:`.cost`).

The runtime — scoping, checkpoints, restore — is :mod:`._runtime`; it is
imported by the flow, not from here (it depends on the flow's node
modules, which depend on this package).
"""

from .base import NO_RESOURCES, R, Resource, ResourceKey, check_resource, resource_method
from .cost import COST


__all__ = [
    "COST",
    "NO_RESOURCES",
    "R",
    "Resource",
    "ResourceKey",
    "check_resource",
    "resource_method",
]
