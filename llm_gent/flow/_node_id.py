# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Content-addressed node IDs for the composition graph.

Every ``_Node`` at execution time gets a stable, globally unique ID
derived from the enclosing Flow's ``chain_context``, the node's kind
(call / branch / iterate / map), a string identifying its target (a
named ``.call`` step's name alone, else the target's qualname with its
label, when it has one), and its occurrence among same-target steps of
the parent chain (not its chain position). Subflow / branch-arm /
iterate-body descents extend ``chain_context`` via
:func:`_descend_context` so a shared subflow used at two call sites
produces two distinct IDs for the same underlying ``_Node``.

:class:`~llm_gent.flow.structure.FlowStructure` describes the same
inputs over the whole static composition tree; its hash is recorded on
every commit a history writes.

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
    the same qualname), with ``[<node_label>]`` appended when the target
    exposes a non-empty ``node_label`` (a :class:`Loop` labels itself
    with its ``name=`` or its role's name). :class:`Flow` subflow →
    ``flow:<name>`` (or ``flow:<anonymous>`` when unnamed). Composition
    primitives (:class:`_Branch`, :class:`_Iterate`, :class:`_Map`) →
    the primitive's kind string, plus ``:<name>`` when built with
    ``name=``.

    Targets whose strings are equal — unnamed primitives, two Loops with
    the same label — are told apart only by their order among
    themselves, so removing an earlier one hands its id to the next.
    Labels exist to prevent that; a ``.call`` step's ``name=`` replaces
    this string altogether (:func:`_step_target`).
    """
    from .flow import Flow

    if isinstance(target, Flow):
        return f"flow:{target.name or '<anonymous>'}"
    if isinstance(target, _Branch | _Iterate | _Map):
        kind = _PRIMITIVE_KINDS[type(target)]
        return kind if target.name is None else f"{kind}:{target.name}"
    module = getattr(target, "__module__", "?")
    qualname = getattr(target, "__qualname__", type(target).__name__)
    label = getattr(target, "node_label", None)
    base = f"verb:{module}.{qualname}"
    return f"{base}[{label}]" if isinstance(label, str) and label else base


_PRIMITIVE_KINDS: dict[type, str] = {_Branch: "branch", _Iterate: "iterate", _Map: "map"}


def _node_kind(node: _Node) -> str:
    """Chain-step kind for the composition tree hash: ``call`` / ``branch`` / ``iterate`` / ``map``."""
    return _PRIMITIVE_KINDS.get(type(node.target), "call")


def _step_target(node: _Node) -> str:
    """The step's target qualname; ``#<name>`` alone when a ``.call`` step was named.

    A named step is identified by its name, not by what it calls: moving
    or renaming its verb keeps its node id, and so its checkpoints and
    paused turns.
    """
    return _target_qualname(node.target) if node.name is None else f"#{node.name}"


def _compute_node_ids(chain_context: str, nodes: list[_Node]) -> tuple[str, ...]:
    """Runtime content-addressed node IDs for a Flow's chain steps, in chain order.

    Each step's local key is ``(kind, target_qualname, occurrence)``,
    where ``occurrence`` counts the earlier steps in the same chain with
    the same kind and target. Chain position does not enter the key:
    inserting, removing or reordering steps with other targets leaves a
    step's id unchanged, so its checkpoints and paused turns are still found
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
        local = (_node_kind(node), _step_target(node))
        occurrence = seen.get(local, 0)
        seen[local] = occurrence + 1
        payload = f"{chain_context}|{local[0]}|{occurrence}|{local[1]}"
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_NODE_ID_DIGEST_SIZE)
        ids.append(digest.hexdigest())
    return tuple(ids)


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
