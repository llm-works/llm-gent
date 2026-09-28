# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Per-run execution-record writer and live-scope registry.

A :class:`RunRecorder` exists for every top-level run of a flow with a
checkpointer (``Flow._recorder``). The executor reports each node
instance that completes, and each control decision it makes, and the
recorder writes the entry into its :class:`ExecutionRecord`. Every commit
the run writes carries the record and the registered scopes (see
:meth:`CheckpointContext.put_state_tree`).

Completion rule — an instance is recorded only when it finishes while
the run is not interrupted: the halt event is unset and no Loop has
paused during the run. A verb that returns early because the halt fired,
or a step whose Loop returned a paused result, is therefore never
recorded; it runs again on resume. Nothing is recorded once the halt is
set, control decisions included.

Entry keys are ``"<kind>|<address>"`` with kinds ``s`` (chain step),
``p`` (one iterate pass), ``i`` (a map's item list), ``m`` (one map
item) and ``b`` (a branch verdict).

Scopes: a child scope created by ``state=`` on ``.call`` / ``.iterate`` /
``.map`` is registered under its owner's address while it is live and
dropped once the owner merges it back. Scopes still registered when a
commit is written — those of instances the halt or an exception
interrupted — are saved with it, so the work done inside them survives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ._halt_observer import is_halt_signaled
from .state.record import (
    ExecutionRecord,
    RecordError,
    instance_address,
    iteration_coord,
    storable,
)


if TYPE_CHECKING:
    from .nodes import _RunEnv
    from .state import State


BranchArm = Literal["then", "else", "none"]


class RunRecorder:
    """Writes one run's execution record and tracks its live child scopes."""

    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record
        self.scopes: dict[str, State[Any]] = {}
        self._paused = False

    def mark_paused(self) -> None:
        """A Loop returned a paused result: nothing completes from here on."""
        self._paused = True

    def interrupted(self, env: _RunEnv) -> bool:
        """True once the halt is set or a Loop has paused."""
        return self._paused or is_halt_signaled(env)

    # --- scopes --------------------------------------------------------------

    def open_scope(self, owner: str, state: State[Any]) -> None:
        """Register ``state`` as the live child scope of the instance at ``owner``."""
        self.scopes[owner] = state

    def close_scope(self, env: _RunEnv, owner: str) -> None:
        """Drop ``owner``'s scope once its owner completed; keep it when interrupted.

        An owner that finishes while the run is interrupted is not
        recorded, so it runs again on resume and needs its scope back.
        """
        if not self.interrupted(env):
            self.scopes.pop(owner, None)

    # --- entries -------------------------------------------------------------

    def record_step(
        self,
        env: _RunEnv,
        node_id: str,
        label: str,
        input_hash: str,
        output: Any,
        needed: bool,
    ) -> None:
        """Record a completed chain step: its input hash and, when storable, its output.

        ``needed`` — a later node receives ``output`` — makes an output the
        record cannot store exactly a :class:`RecordError`; otherwise it is
        recorded without the output.
        """
        if self.interrupted(env):
            return
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
        self.record.put(f"s|{address}", entry)

    def record_pass(
        self, env: _RunEnv, iterate_id: str, iteration: int, output: Any, cont: bool
    ) -> None:
        """Record the ``iteration``-th pass of an iterate body and whether ``until`` let it continue."""
        if self.interrupted(env):
            return
        address = instance_address(iterate_id, env.coords + (iteration_coord(iteration),))
        self.record.put(f"p|{address}", {"out": _needed_output(output, address), "cont": cont})

    def record_items(self, env: _RunEnv, map_id: str, items: list[Any]) -> None:
        """Record the item list a map fanned out over."""
        if self.interrupted(env):
            return
        address = instance_address(map_id, env.coords)
        try:
            encoded = [storable(item) for item in items]
        except RecordError as e:
            raise RecordError(f"items of map {address} cannot be recorded: {e}") from e
        self.record.put(f"i|{address}", {"items": encoded})

    def record_item(self, env: _RunEnv, item_address: str, output: Any) -> None:
        """Record a map item whose body completed and whose merge was applied."""
        if self.interrupted(env):
            return
        self.record.put(f"m|{item_address}", {"out": _needed_output(output, item_address)})

    def record_item_skipped(self, env: _RunEnv, item_address: str) -> None:
        """Record a map item its guard skipped."""
        if self.interrupted(env):
            return
        self.record.put(f"m|{item_address}", {"skip": True})

    def record_branch(self, env: _RunEnv, branch_id: str, arm: BranchArm) -> None:
        """Record which arm a branch's predicate chose (``none``: falsy with no ``else_``)."""
        if self.interrupted(env):
            return
        self.record.put(f"b|{instance_address(branch_id, env.coords)}", {"arm": arm})


def _needed_output(output: Any, address: str) -> Any:
    """Stored form of an output a later node receives; :class:`RecordError` when it has none."""
    try:
        return storable(output)
    except RecordError as e:
        raise RecordError(f"output at {address} cannot be recorded: {e}") from e
