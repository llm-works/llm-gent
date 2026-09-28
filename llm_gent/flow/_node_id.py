# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Content-addressed node IDs for the composition graph.

Every ``_Node`` at execution time gets a stable, globally unique ID
derived from the enclosing Flow's ``chain_context``, the node's kind
(call / branch / iterate / map), a qualname string identifying its
target, and its occurrence among same-target steps of the parent
chain (not its chain position). Subflow / branch-arm /
iterate-body descents extend ``chain_context`` via
:func:`_descend_context` so a shared subflow used at two call sites
produces two distinct IDs for the same underlying ``_Node``.

:func:`flow_root_hash` hashes every chain step's kind and target in
chain order over the whole static composition tree — a superset of the
id inputs, so equal hashes mean equal ids: one hash per flow
definition, recorded on every commit a history writes.

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
    """Stable identity string for a node target — feeds :func:`_compute_node_ids`.

    Verb / plain callable → ``verb:<__module__>.<__qualname__>`` (module
    prefix prevents cross-module collisions between two functions with
    the same qualname). :class:`Flow` subflow → ``flow:<name>`` (or
    ``flow:<anonymous>`` when unnamed). Composition primitives
    (:class:`_Branch`, :class:`_Iterate`, :class:`_Map`) → the
    primitive's kind string; the primitive's identity flows from its
    occurrence among same-kind steps of the chain and the enclosing
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


def _compute_node_ids(chain_context: str, nodes: list[_Node]) -> tuple[str, ...]:
    """Runtime content-addressed node IDs for a Flow's chain steps, in chain order.

    Each step's local key is ``(kind, target_qualname, occurrence)``,
    where ``occurrence`` counts the earlier steps in the same chain with
    the same kind and target. Chain position does not enter the key:
    inserting, removing or reordering steps with other targets leaves a
    step's id unchanged, so its record and paused turns are still found
    after a deploy that edits the chain around it. Steps sharing a
    target (the same verb twice, two anonymous subflows, two maps) are
    told apart by their order among themselves.

    The key is composed with the enclosing Flow's ``chain_context`` — a
    hash chain from the run's root down through every subflow /
    branch-arm / iterate-body / map-body descent above this Flow (see
    :func:`_descend_context`) — so ids are globally unique across the
    composition tree: a shared subflow used at two call sites produces
    two distinct ids for the same underlying ``_Node``.

    Deterministic: identical composition graphs produce identical ids
    across processes / Python versions (blake2b is stable and every
    input is a Unicode-canonical string). See
    :data:`_NODE_ID_DIGEST_SIZE` for the birthday-collision margin.
    """
    seen: dict[tuple[str, str], int] = {}
    ids: list[str] = []
    for node in nodes:
        local = (_node_kind(node), _target_qualname(node.target))
        occurrence = seen.get(local, 0)
        seen[local] = occurrence + 1
        payload = f"{chain_context}|{local[0]}|{occurrence}|{local[1]}"
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_NODE_ID_DIGEST_SIZE)
        ids.append(digest.hexdigest())
    return tuple(ids)


def flow_root_hash(flow: Any) -> str:
    """Structure hash of ``flow``'s composition tree — the commit meta's ``flow_root_hash``.

    Hashes (CAS :func:`content_hash` over canonical JSON) every chain
    step's kind and target qualname in chain order, and every Flow it
    descends into under its boundary name — a superset of what
    :func:`_compute_node_ids` and :func:`_descend_context` consume. Two
    flows with equal hashes therefore assign identical node ids to every
    step; flows that differ only in step order hash differently but can
    still share ids.

    Node parameters (``max_iters``, predicates, projections) do not
    enter node ids and do not enter this hash. A Flow that re-enters
    itself (recursion through a branch arm) is recorded as a back
    reference to its depth on the descent path.
    """
    from .state.cas import canonical_json, content_hash

    return content_hash(canonical_json(_flow_structure(flow, ())))


def _flow_structure(flow: Any, ancestors: tuple[int, ...]) -> Any:
    """Canonical-JSON-ready description of ``flow``'s chain, one entry per step.

    ``ancestors`` holds the ``id()`` of every Flow on the descent path
    from the root; only its positions reach the output, never the ids.
    """
    if id(flow) in ancestors:
        return {"cycle": ancestors.index(id(flow))}
    inner = ancestors + (id(flow),)
    return [
        {
            "kind": _node_kind(node),
            "target": _target_qualname(node.target),
            "children": {
                boundary: _flow_structure(child, inner) for boundary, child in _child_flows(node)
            },
        }
        for node in flow._nodes
    ]


def iter_flows(root: Any) -> list[Any]:
    """Every distinct Flow in ``root``'s composition tree, ``root`` first."""
    seen: dict[int, Any] = {}
    stack = [root]
    while stack:
        flow = stack.pop()
        if id(flow) in seen:
            continue
        seen[id(flow)] = flow
        for node in flow._nodes:
            stack.extend(child for _, child in _child_flows(node))
    return list(seen.values())


def _child_flows(node: _Node) -> list[tuple[str, Any]]:
    """``(boundary, Flow)`` pairs a chain step descends into, as named by the executor."""
    from .flow import Flow

    target = node.target
    if isinstance(target, _Branch):
        arms = [("then", target.then_flow)]
        return arms + ([("else", target.else_flow)] if target.else_flow is not None else [])
    if isinstance(target, _Iterate):
        return [("body", target.body)]
    if isinstance(target, _Map):
        return [("map", target.body)]
    if isinstance(target, Flow):
        return [("call", target)]
    return []


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
