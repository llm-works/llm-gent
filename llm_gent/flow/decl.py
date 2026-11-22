# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""What one flow declares for its subtree: a scope with every field optional.

A run applies it as the walk enters the flow (``Scope.child`` in
:mod:`llm_gent.flow._scope`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._shortcut import Shortcut
from .checkpoint import CheckpointPolicy
from .nodes import Checkpointer
from .resource import NO_RESOURCES, ResourceKey


_NO_CHILDREN: Mapping[ResourceKey[Any], Mapping[str, Any]] = MappingProxyType({})


@dataclass(frozen=True)
class Decl:
    """What one flow declares for its subtree: a scope with every field optional.

    Read off a flow by :attr:`Flow.decl <llm_gent.flow.Flow.decl>`.
    """

    checkpointer: Checkpointer | None = None
    policy: CheckpointPolicy | None = None
    resources: Mapping[ResourceKey[Any], Any] = NO_RESOURCES
    """The resources the flow declares (:meth:`Flow.with_resource`)."""
    resource_children: Mapping[ResourceKey[Any], Mapping[str, Any]] = _NO_CHILDREN
    """The per-run children the flow asks for: each key maps to the keyword arguments of
    the resource's ``child()`` (``with_resource(key, **args)``)."""
    hooks: tuple[object, ...] = ()
    hooks_replace: bool = False
    """The flow's hooks replace the inherited ones instead of extending them."""
    shortcut: Shortcut | None = None
