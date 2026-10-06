# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The structure of a flow: its chains of steps, by what their node ids are hashed from.

A :class:`Structure` is the static composition tree of a flow: every
chain's steps in order, each identified by its :class:`StepKey` — kind,
target and occurrence, the inputs of its node id — with the chains it
descends into by boundary (``call``, ``then``, ``else``, ``body``,
``map``). Node parameters (``max_iters``, predicates, projections) are
not part of it.

Its hash is the ``flow_root_hash`` every commit records, and every
commit's tree holds it, so the structure that wrote a commit can be read
back (:meth:`~llm_gent.flow.History.structure`) and compared with a flow
today (:meth:`Structure.diff`). Equal structures assign identical
node ids to every step.

A step is addressed by its :data:`StepPath`: the keys of the steps above
it and its own, each with the boundary its chain was entered by. A path
is position-independent, like a node id: inserting or removing steps
with other targets leaves it unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from ._node_id import _child_flows, _node_kind, _step_target
from .state.cas import Blob, canonical_json


@dataclass(frozen=True)
class StepKey:
    """A step's identity in its chain: what its node id is hashed from.

    ``kind`` is ``call``, ``branch``, ``iterate`` or ``map``. ``target`` is
    ``#<name>`` for a named ``.call`` step, else its target (a verb's
    module and qualname, ``flow:<name>``, a primitive's kind and name).
    ``occurrence`` counts the earlier steps of the chain with the same
    kind and target.
    """

    kind: str
    target: str
    occurrence: int

    @property
    def label(self) -> str:
        """Readable form: the step's name or target, with ``[n]`` from its second occurrence on."""
        base = self.target.removeprefix("#")
        return base if self.occurrence == 0 else f"{base}[{self.occurrence}]"


Segment = tuple[str, StepKey]
"""One level of a :data:`StepPath`: the boundary its chain was entered by, and the step's key."""

StepPath = tuple[Segment, ...]
"""A step's address from the top-level chain (whose boundary is ``""``) down to the step."""


def path_label(path: StepPath) -> str:
    """Readable form of ``path``, e.g. ``"research / map: explore"``."""
    return " / ".join(key.label if not b else f"{b}: {key.label}" for b, key in path)


@dataclass(frozen=True)
class Cycle:
    """A flow re-entering a flow above it (recursion through a branch arm), by its depth."""

    depth: int


@dataclass(frozen=True)
class Step:
    """One step of a chain: its key and the chains it descends into, by boundary."""

    key: StepKey
    children: Mapping[str, Structure | Cycle]


@dataclass(frozen=True)
class Structure:
    """A flow's static composition tree: its chain's steps, in order."""

    steps: tuple[Step, ...]

    @classmethod
    def of(cls, flow: Any) -> Structure:
        """The structure of ``flow`` as it is now."""
        return _of(flow, ())

    @classmethod
    def from_json(cls, raw: list[dict[str, Any]]) -> Structure:
        """The structure :meth:`to_json` wrote.

        Raises:
            KeyError: An entry lacks ``kind``, ``target`` or ``children``.
        """
        seen: dict[tuple[str, str], int] = {}
        steps: list[Step] = []
        for entry in raw:
            local = (entry["kind"], entry["target"])
            occurrence = seen.get(local, 0)
            seen[local] = occurrence + 1
            children = {b: _child_from_json(c) for b, c in entry["children"].items()}
            steps.append(Step(StepKey(local[0], local[1], occurrence), children))
        return cls(tuple(steps))

    def to_json(self) -> list[dict[str, Any]]:
        """Canonical-JSON-ready form: one entry per step (occurrences follow from the order)."""
        return [
            {
                "kind": step.key.kind,
                "target": step.key.target,
                "children": {b: _child_to_json(c) for b, c in step.children.items()},
            }
            for step in self.steps
        ]

    def blob(self) -> Blob:
        """The structure as the blob a commit's tree holds; its hash is :attr:`hash`."""
        return Blob.from_bytes(canonical_json(self.to_json()))

    @cached_property
    def hash(self) -> str:
        """The structure hash, recorded on every commit as ``flow_root_hash``."""
        return self.blob().content_hash

    def steps_by_path(self) -> dict[StepPath, Step]:
        """Every step by its path, parents before children (not into cycles)."""
        out: dict[StepPath, Step] = {}
        _collect(self, (), "", out)
        return out

    def chain(self, parent: StepPath, boundary: str) -> Structure | None:
        """The chain step ``parent`` descends into by ``boundary`` (the top level: ``(), ""``)."""
        if not parent:
            return self if boundary == "" else None
        step = self.steps_by_path().get(parent)
        child = None if step is None else step.children.get(boundary)
        return child if isinstance(child, Structure) else None

    def diff(self, new: Structure) -> StructureDiff:
        """How ``new`` differs from this structure."""
        return StructureDiff(self, new)


@dataclass(frozen=True)
class StructureDiff:
    """The steps two structures share, and those only one of them has, by path.

    ``old`` is the structure a commit was written by, ``new`` a flow's
    today. A step whose key and every key above it are in both is kept;
    a step under a removed step is removed with it.
    """

    old: Structure
    new: Structure

    @property
    def changed(self) -> bool:
        """Whether the structures differ at all, step order included (their hashes differ)."""
        return self.old.hash != self.new.hash

    @cached_property
    def _old_paths(self) -> dict[StepPath, Step]:
        return self.old.steps_by_path()

    @cached_property
    def _new_paths(self) -> dict[StepPath, Step]:
        return self.new.steps_by_path()

    @property
    def kept(self) -> tuple[StepPath, ...]:
        """Steps in both structures, in the new structure's order."""
        return tuple(p for p in self._new_paths if p in self._old_paths)

    @property
    def removed(self) -> tuple[StepPath, ...]:
        """Steps only the old structure has, in its order."""
        return tuple(p for p in self._old_paths if p not in self._new_paths)

    @property
    def added(self) -> tuple[StepPath, ...]:
        """Steps only the new structure has, in its order."""
        return tuple(p for p in self._new_paths if p not in self._old_paths)

    def added_before(self, path: StepPath) -> tuple[StepPath, ...]:
        """Steps added to ``path``'s chain ahead of it in the new structure.

        Empty when ``path`` is not in the new structure.
        """
        if not path or path not in self._new_paths:
            return ()
        parent, (boundary, key) = path[:-1], path[-1]
        chain = self.new.chain(parent, boundary)
        assert chain is not None  # path is in the new structure
        ahead = [s.key for s in chain.steps[: [s.key for s in chain.steps].index(key)]]
        return tuple(p for k in ahead if (p := (*parent, (boundary, k))) not in self._old_paths)


def _of(flow: Any, ancestors: tuple[int, ...]) -> Structure:
    """``flow``'s structure; ``ancestors`` are the ``id()`` of the flows on the descent path."""
    inner = ancestors + (id(flow),)
    seen: dict[tuple[str, str], int] = {}
    steps: list[Step] = []
    for node in flow._nodes:
        local = (_node_kind(node), _step_target(node))
        occurrence = seen.get(local, 0)
        seen[local] = occurrence + 1
        children: dict[str, Structure | Cycle] = {
            boundary: Cycle(inner.index(id(child))) if id(child) in inner else _of(child, inner)
            for boundary, child in _child_flows(node)
        }
        steps.append(Step(StepKey(local[0], local[1], occurrence), children))
    return Structure(tuple(steps))


def _child_to_json(child: Structure | Cycle) -> Any:
    """A child chain's JSON: its steps, or ``{"cycle": depth}``."""
    return {"cycle": child.depth} if isinstance(child, Cycle) else child.to_json()


def _child_from_json(raw: Any) -> Structure | Cycle:
    """The child chain :func:`_child_to_json` wrote."""
    return Cycle(raw["cycle"]) if isinstance(raw, dict) else Structure.from_json(raw)


def _collect(
    structure: Structure, prefix: StepPath, boundary: str, out: dict[StepPath, Step]
) -> None:
    """Add ``structure``'s steps under ``prefix`` (entered by ``boundary``) to ``out``."""
    for step in structure.steps:
        path = (*prefix, (boundary, step.key))
        out[path] = step
        for child_boundary, child in step.children.items():
            if isinstance(child, Structure):
                _collect(child, path, child_boundary, out)
