# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Per-run execution-record writer and live-scope registry.

A :class:`RunRecorder` exists for every top-level run of a flow with a
checkpointer (``Flow._recorder``). The executor reports each node
instance that completes, and each control decision it makes, and the
recorder writes the entry into its :class:`ExecutionRecord`. Every commit
the run writes carries the record and the registered scopes (see
:meth:`CheckpointContext.put_state_tree`).

An instance is identified by its node id and the dynamic coordinates of
the iterate passes and map items enclosing it. It is recorded only when
it finishes complete:

- no halt event has been seen set — the run's, or a subflow's own; once
  one is, nothing is recorded, control decisions included, since a verb
  may have returned early because of it;
- no Loop under it has paused — a paused Loop is found under an instance
  when the instance's node is the Loop's step or one of its structural
  ancestors, and the instance's coordinates are a prefix of the Loop's.
  A pause makes its enclosing instances incomplete, not its siblings.

An incomplete instance is not recorded and runs again on resume. The
executor keeps each chain's (and each iterate's) recorded instances a
prefix: after one goes unrecorded, the later ones in the same walk are
not recorded either, since their inputs or state came from it.

Entry keys are ``"<kind>|<address>"`` with kinds ``s`` (chain step),
``p`` (one iterate pass), ``i`` (a map's item list), ``m`` (one map
item) and ``b`` (a branch verdict).

Scopes: a child scope created by ``state=`` on ``.call`` / ``.iterate`` /
``.map`` is registered under its owner's address while it is live. When
an instance returns, its scope and every scope under it are dropped,
unless the halt or a pause under it left its work unfinished. Scopes
still registered when a commit is written — those of owners the halt, a
pause or a raised exception left incomplete — are saved with it, so the
work done inside them survives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ._halt_observer import is_halt_signaled
from .state import serialize_state_data
from .state.cas import canonical_json
from .state.record import (
    BRANCH,
    ITEM,
    ITEMS,
    PASS,
    STEP,
    ExecutionRecord,
    RecordError,
    instance_address,
    iteration_coord,
    record_key,
    storable,
)


if TYPE_CHECKING:
    from .nodes import _RunEnv
    from .state import State


BranchArm = Literal["then", "else", "none"]


@dataclass(frozen=True)
class _Site:
    """Where an instance sits: its node id, structural ancestors and coordinates."""

    node_id: str
    ancestors: tuple[str, ...]
    coords: tuple[str, ...]

    def under(self, node_id: str, coords: tuple[str, ...]) -> bool:
        """True when this site is the instance ``(node_id, coords)`` or lies inside it."""
        in_node = node_id == self.node_id or node_id in self.ancestors
        return in_node and self.coords[: len(coords)] == coords


@dataclass(frozen=True)
class _Scope:
    """A registered child scope and the site of the instance that owns it."""

    owner: _Site
    state: State[Any]


class RunRecorder:
    """Writes one run's execution record and tracks its live child scopes."""

    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record
        self._scopes: dict[str, _Scope] = {}
        self._pauses: list[_Site] = []
        self._unlocated_pause = False
        self._halted = False

    def mark_paused(self, env: _RunEnv, node_id: str | None) -> None:
        """A Loop at chain step ``node_id`` (under ``env``) returned a paused result.

        A Loop dispatched without a node id cannot be located, so its
        pause stops recording for the rest of the run, like the halt.
        """
        if node_id is None:
            self._unlocated_pause = True
            return
        self._pauses.append(_Site(node_id, env.ancestor_chain, env.coords))

    def halted(self, env: _RunEnv) -> bool:
        """True once nothing more may be recorded anywhere in the run.

        A halt seen under any env latches: a subflow's own ``.with_halt()``
        event is not visible from the enclosing env, yet the step holding
        that subflow returned early because of it.
        """
        if not self._halted and is_halt_signaled(env):
            self._halted = True
        return self._halted or self._unlocated_pause

    def closed(self, env: _RunEnv) -> bool:
        """True when nothing may be recorded under ``env``.

        The run is halted, or ``env`` belongs to a walk that stopped
        recording because an earlier instance of an enclosing walk went
        unrecorded (``env.recording``).
        """
        return not env.recording or self.halted(env)

    def incomplete(self, env: _RunEnv, node_id: str, coords: tuple[str, ...]) -> bool:
        """True when the instance ``(node_id, coords)`` must not be recorded."""
        return self.closed(env) or self._held(env, node_id, coords)

    def _held(self, env: _RunEnv, node_id: str, coords: tuple[str, ...]) -> bool:
        """True when the instance returned with its work unfinished (halted, or a Loop paused).

        Unlike :meth:`incomplete`, a walk that merely stopped recording
        does not hold its instances: they finished, and their merges are
        in the parent's state.
        """
        return self.halted(env) or any(p.under(node_id, coords) for p in self._pauses)

    # --- scopes --------------------------------------------------------------

    def open_scope(
        self, env: _RunEnv, node_id: str, coords: tuple[str, ...], state: State[Any]
    ) -> None:
        """Register ``state`` as the live child scope of the instance ``(node_id, coords)``.

        Every commit the run writes carries the open scopes, so a scope
        that cannot be serialized is refused here, with its owner named,
        rather than failing each later commit.
        """
        address = instance_address(node_id, coords)
        _serialized_scope(address, state)
        self._scopes[address] = _Scope(_Site(node_id, env.ancestor_chain, coords), state)

    def close_scope(self, env: _RunEnv, node_id: str, coords: tuple[str, ...]) -> None:
        """Drop the scopes of the instance ``(node_id, coords)`` and of everything under it.

        Called once the instance returned. Kept when the instance is held
        (halted, or a Loop under it paused): it runs again on resume and
        needs the work done in them. Otherwise every scope under it is
        dead — including those a raised exception left behind that a
        rescue, or a non-strict map, turned into a result.
        """
        if self._held(env, node_id, coords):
            return
        for address in [a for a, s in self._scopes.items() if s.owner.under(node_id, coords)]:
            del self._scopes[address]

    def serialized_scopes(self) -> dict[str, bytes]:
        """Canonical JSON of every open scope by owner address; :class:`RecordError` when one fails."""
        return {a: _serialized_scope(a, s.state) for a, s in self._scopes.items()}

    # --- entries -------------------------------------------------------------

    def record_step(
        self,
        env: _RunEnv,
        node_id: str,
        label: str,
        input_hash: str,
        output: Any,
        needed: bool,
    ) -> bool:
        """Record a completed chain step; return whether it was recorded.

        Stores its input hash and, when storable, its output. ``needed`` —
        a later node receives ``output`` — makes an output the record
        cannot store exactly a :class:`RecordError`; otherwise the step is
        recorded without it.
        """
        if self.incomplete(env, node_id, env.coords):
            return False
        address = instance_address(node_id, env.coords)
        entry: dict[str, Any] = {"in": input_hash}
        try:
            entry["out"] = storable(output)
        except RecordError as e:
            if needed:
                raise RecordError(
                    f"output of {label} (node {address}) cannot be recorded, and a later "
                    f"node receives it: {e}"
                ) from e
        self.record.put(record_key(STEP, address), entry)
        return True

    def record_pass(
        self, env: _RunEnv, iterate_id: str, iteration: int, output: Any, cont: bool
    ) -> bool:
        """Record pass ``iteration`` of an iterate body and whether ``until`` let it continue."""
        coords = env.coords + (iteration_coord(iteration),)
        if self.incomplete(env, iterate_id, coords):
            return False
        address = instance_address(iterate_id, coords)
        entry = {"out": _needed_output(output, address), "cont": cont}
        self.record.put(record_key(PASS, address), entry)
        return True

    def record_items(self, env: _RunEnv, map_id: str, items: list[Any]) -> None:
        """Record the item list a map fanned out over."""
        if self.closed(env):
            return
        address = instance_address(map_id, env.coords)
        try:
            encoded = [storable(item) for item in items]
        except RecordError as e:
            raise RecordError(f"items of map {address} cannot be recorded: {e}") from e
        self.record.put(record_key(ITEMS, address), {"items": encoded})

    def record_item(
        self, env: _RunEnv, map_id: str, item_coords: tuple[str, ...], output: Any
    ) -> bool:
        """Record a map item whose body completed and whose merge was applied; return whether."""
        if self.incomplete(env, map_id, item_coords):
            return False
        address = instance_address(map_id, item_coords)
        self.record.put(record_key(ITEM, address), {"out": _needed_output(output, address)})
        return True

    def unrecord_item(self, map_id: str, item_coords: tuple[str, ...]) -> None:
        """Undo :meth:`record_item` (if it recorded) after the item's merge was rolled back."""
        self.record.remove(record_key(ITEM, instance_address(map_id, item_coords)))

    def record_item_skipped(self, env: _RunEnv, map_id: str, item_coords: tuple[str, ...]) -> None:
        """Record a map item its guard skipped."""
        if self.closed(env):
            return
        self.record.put(record_key(ITEM, instance_address(map_id, item_coords)), {"skip": True})

    def record_branch(self, env: _RunEnv, branch_id: str, arm: BranchArm) -> None:
        """Record which arm a branch's predicate chose (``none``: falsy with no ``else_``)."""
        if self.closed(env):
            return
        self.record.put(record_key(BRANCH, instance_address(branch_id, env.coords)), {"arm": arm})


def _serialized_scope(owner: str, state: State[Any]) -> bytes:
    """Canonical JSON of a child scope; :class:`RecordError` naming its owner when it has none."""
    try:
        return canonical_json(serialize_state_data(state.data))
    except (TypeError, ValueError) as e:
        raise RecordError(f"scope of {owner} cannot be serialized for a commit: {e}") from e


def _needed_output(output: Any, address: str) -> Any:
    """Stored form of an output a later node receives; :class:`RecordError` when it has none."""
    try:
        return storable(output)
    except RecordError as e:
        raise RecordError(f"output at {address} cannot be recorded: {e}") from e
