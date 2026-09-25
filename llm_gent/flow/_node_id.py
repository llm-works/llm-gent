# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Content-addressed node IDs for the composition graph.

Every ``_Node`` at execution time gets a stable, globally unique ID
derived from the enclosing Flow's ``chain_context``, the node's kind
(call / branch / iterate / map), its position in the parent chain,
and a qualname string identifying its target. Subflow / branch-arm /
iterate-body descents extend ``chain_context`` via
:func:`_descend_context` so a shared subflow used at two call sites
produces two distinct IDs for the same underlying ``_Node``.

The :class:`Flow` isinstance check in :func:`_target_qualname`
resolves through a localized late import to break the circular
dependency between this module and :mod:`.flow`.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .nodes import _Branch, _Iterate, _Map, _Node


_NODE_ID_DIGEST_SIZE = 8
"""Byte length of the blake2b digest for node IDs — 64 bits, 16 hex chars.

Sized for headroom: a composition tree of a few thousand nodes has
essentially zero birthday-collision risk at 2^32, which is the design
guarantee behind treating node IDs as globally unique in the graph.
Bumping to 16 (128 bits) would leave zero doubt but doubles envelope
size; 8 is the deliberate default.
"""


def _target_qualname(target: Any) -> str:
    """Stable identity string for a node target — feeds :func:`_compute_node_id`.

    Verb / plain callable → ``verb:<__module__>.<__qualname__>`` (module
    prefix prevents cross-module collisions between two functions with
    the same qualname). :class:`Flow` subflow → ``flow:<name>`` (or
    ``flow:<anonymous>`` when unnamed). Composition primitives
    (:class:`_Branch`, :class:`_Iterate`, :class:`_Map`) → the
    primitive's kind string; the primitive's identity flows from the
    outer :class:`_Node`'s chain position and its enclosing
    ``chain_context``, not from any label on the primitive itself.
    """
    from .flow import Flow

    if isinstance(target, Flow):
        return f"flow:{target.name or '<anonymous>'}"
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Iterate):
        return "iterate"
    if isinstance(target, _Map):
        return "map"
    module = getattr(target, "__module__", "?")
    qualname = getattr(target, "__qualname__", type(target).__name__)
    return f"verb:{module}.{qualname}"


def _node_kind(node: _Node) -> str:
    """Chain-step kind for the composition tree hash: ``call`` / ``branch`` / ``iterate`` / ``map``."""
    target = node.target
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Iterate):
        return "iterate"
    if isinstance(target, _Map):
        return "map"
    return "call"


def _compute_node_id(chain_context: str, node: _Node, position: int) -> str:
    """Runtime content-addressed node ID for the chain step at ``position``.

    Composes the enclosing Flow's ``chain_context`` with this node's
    local key ``(kind, position, target_qualname)``. The chain_context
    is itself a hash chain from the run's root down through every
    subflow / branch-arm / iterate-body descent above this Flow (see
    :func:`_descend_context`), so the resulting node ID is globally
    unique across the entire composition tree — a shared subflow used
    at two call sites produces two distinct IDs for the same underlying
    ``_Node`` because their ``chain_context`` values differ.

    Deterministic: identical composition graphs produce identical IDs
    across processes / Python versions (blake2b is stable and every
    input is a Unicode-canonical string). Collision-free at the design
    level for any well-formed graph — see :data:`_NODE_ID_DIGEST_SIZE`
    for the birthday-collision margin.
    """
    payload = f"{chain_context}|{_node_kind(node)}|{position}|{_target_qualname(node.target)}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_NODE_ID_DIGEST_SIZE).hexdigest()


def _descend_context(parent_node_id: str, boundary: str) -> str:
    """Chain context for a Flow entered from ``parent_node_id`` via ``boundary``.

    ``boundary`` names the slot of the parent node this Flow fills:
    ``"body"`` (an :class:`_Iterate` body), ``"then"`` / ``"else"``
    (arms of a :class:`_Branch`), or ``"call"`` (a subflow reached from
    :meth:`Flow.call`). Baking the boundary into the descended context
    keeps the two arms of a branch and the body of an iterate at
    identity-distinct positions even when they share a target.
    """
    payload = f"{parent_node_id}|{boundary}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_NODE_ID_DIGEST_SIZE).hexdigest()
