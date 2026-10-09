# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Map-node dispatcher: fan-out + per-item run with guards, hooks, and merges.

Extracted from :mod:`._executor`. Two classes:

- :class:`MapRunner` resolves the input items (via ``items_fn`` or
  the incoming ``prev_result``), spawns one :class:`MapItemRunner` per
  item under an optional concurrency cap, then aggregates results
  through ``mp.aggregate`` when provided. Strict mode re-raises the
  first per-item exception; non-strict swaps failing items for a
  :class:`Failure` sentinel so the result list keeps every position.

- :class:`MapItemRunner` runs a single item through the halt-skip
  check, state projection, guard predicate, body dispatch, and the
  strict/non-strict exception fan-out, then merges the child state
  back into the parent under a shared lock and fires the
  ``on_item_complete`` hook. Per-item merges serialize on
  ``merge_lock`` so concurrent items don't race on ``env.state``.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._executor import (
    _enter_scope,
    _merge_state,
    _restore_scope_state,
    _running,
    _save_scope_commit,
)
from ._halt_observer import is_halt_signaled
from ._node_id import _descend_context
from ._shortcut import is_fast_forward, leave_drains
from ._validation import check_concurrency
from .context import Context
from .nodes import Failure, Interrupted, ItemsFn, RestoredError, Skipped
from .state import serialize_state_data
from .state.snapshot import DONE, ITEMS, ScopePath


if TYPE_CHECKING:
    from .nodes import _Map, _RunEnv
    from .state import State


_INTERRUPTED = object()
"""Result of a map item the halt stopped before it completed."""


@dataclass
class _Done:
    """A completed item: its result, and whether its scope merged into the parent yet.

    A failed (``strict=False``) or skipped item is done too: its result is
    its :class:`Failure` or :class:`Skipped`, with nothing to merge.
    """

    result: Any
    merged: bool

    def stored(self) -> dict[str, Any]:
        """The record as the map's cursor stores it.

        A failure keeps its exception's type and message (an exception is
        not storable); a resumed map gets them back as a :class:`RestoredError`.
        """
        if isinstance(self.result, Failure):
            exc = self.result.exception
            if isinstance(exc, RestoredError):
                return {"failure": {"type": exc.type_name, "message": exc.message}}
            return {"failure": {"type": type(exc).__name__, "message": str(exc)}}
        if isinstance(self.result, Skipped):
            return {"skipped": True}
        return {"result": self.result, "merged": self.merged}

    @classmethod
    def from_stored(cls, raw: dict[str, Any], item: Any) -> _Done:
        """The record :meth:`stored` wrote for ``item``."""
        if "failure" in raw:
            failure = raw["failure"]
            error = RestoredError(failure["type"], failure["message"])
            return cls(Failure(exception=error, item=item), merged=True)
        if raw.get("skipped"):
            return cls(Skipped(item=item), merged=True)
        return cls(raw["result"], raw["merged"])


class MapRunner:
    """Fan out one ``.map`` node's body over its items.

    Constructed with (mp, env, node_id). Public method :meth:`run` takes
    the incoming ``node_args`` and returns the aggregated result (or the
    raw list when no aggregator is attached).

    Its cursor — the items it runs over and the completed ones with their
    results — is state on the runner, in every snapshot taken while it
    runs. Each item has a key: its index in a map over items, the member's
    key in a map over members. Items that are running keep their own
    positions under ``<map>/i/<key>``. On ``resume="latest"`` a map over
    items runs over the saved items; a map over members over its current
    members, matched to their records by key. Either way a completed item
    does not run again — nor does one that failed (``strict=False``) or was
    skipped — a running one continues where it was, and the rest run.
    """

    def __init__(self, mp: _Map, env: _RunEnv, node_id: str) -> None:
        """Run ``mp`` as the step ``node_id`` of ``env``'s flow."""
        self.mp = mp
        self.env = env
        self.node_id = node_id
        self.path: ScopePath = env.owner_path(node_id)
        # The cursor. Kept here, not in run's locals, so a checkpoint reads it.
        self.items: list[Any] = []
        self.keys: list[str] = []
        self.done: dict[str, _Done] = {}

    def cursor(self) -> dict[str, Any]:
        """The items (a map over items) and the done ones: key to its record (:meth:`_Done.stored`).

        A map over members saves no items: its members come from the flow,
        and each runs on the step's input, which the chain's cursor holds.
        """
        done = {DONE: {k: d.stored() for k, d in self.done.items()}}
        return done if self.mp.over_members else {ITEMS: self.items, **done}

    async def run(self, node_args: tuple[Any, ...]) -> Any:
        """Resolve items, spawn per-item runners, aggregate.

        ``strict=True`` re-raises the first non-cancellation
        exception; sibling tasks continue but their results are
        discarded. ``strict=False`` swaps each failing item for a
        :class:`Failure` sentinel so the caller sees every position.
        Guard skips replace items with :class:`Skipped` before the
        body runs. ``on_error`` fires for each per-item exception in
        both modes without altering control flow. Cancellation
        propagates unconditionally regardless.

        ``max_concurrency=N`` caps in-flight per-item runners via an
        :class:`asyncio.Semaphore`; items above the cap wait. When
        the ambient halt event fires, any per-item runner that has
        not yet passed its halt check stops without running its body,
        so the remaining queue drains; already-in-flight items complete
        or stop at their own halt. When any item stopped that way, the
        map is interrupted: it raises :class:`Interrupted` instead of
        aggregating partial results, and stays registered with its
        completed items for the run's halt checkpoint.

        Raises:
            Interrupted: The halt stopped at least one item.
        """
        await self._take_saved_or_resolve(node_args)
        with _running(self.env, self.path, self):
            results = await self._gather_items(self.items)
            if any(r is _INTERRUPTED for r in results):
                raise Interrupted()
        return await self._aggregate(results)

    async def _take_saved_or_resolve(self, node_args: tuple[Any, ...]) -> None:
        """Continue with the items and completed items a checkout saved; else resolve the items.

        Saved items are used as stored — ``items`` is not evaluated again,
        since what it reads may have changed since the map started. A map
        over members takes its members from the flow and keeps the saved
        records of the members it still has.
        """
        prev_result = node_args[0] if node_args else None
        scopes = self.env.scopes
        if self.mp.over_members:
            self.keys = list(self.mp.member_keys)
            self.items = [prev_result] * len(self.keys)
            _, done = scopes.take_cursor(self.path, DONE)
            saved = {k: d for k, d in (done or {}).items() if k in self.keys}
            self.done = {k: _Done.from_stored(d, prev_result) for k, d in saved.items()}
            return
        found, items = scopes.take_cursor(self.path, ITEMS)
        if found:
            self.items, self.keys = items, [str(i) for i in range(len(items))]
            _, done = scopes.take_cursor(self.path, DONE)
            self.done = {k: _Done.from_stored(d, items[int(k)]) for k, d in (done or {}).items()}
            return
        self.items = await _resolve_items(self.mp.items, prev_result, self._build_ctx())
        self.keys = [str(i) for i in range(len(self.items))]

    def _build_ctx(self) -> Context[Any]:
        """Build the :class:`Context` passed to ``items_fn`` and a computed ``max_concurrency``.

        Uses the parent state at map entry. Role is ``None`` since both
        are data-producing callbacks, not role actions.
        """
        env = self.env
        return Context(
            role=None,
            state=env.state,
            flow=env.runtime,
            traits=env.runtime._traits,
            halt=env.halt,
            extra=env.extra,
            resources=env.resources,
            _env=env,
            _node_id=self.node_id,
        )

    async def _gather_items(self, items: list[Any]) -> list[Any]:
        """Spawn one runner per item under the concurrency cap; return per-item results.

        A completed item returns its saved result without running; one
        whose merge had not happened yet merges now. Strict mode re-raises
        the first non-cancellation exception; non-strict returns every
        item's result (or :class:`Failure` sentinel).
        """
        merge_lock = asyncio.Lock()
        cap = await self._concurrency(items)
        sem = asyncio.Semaphore(cap) if cap is not None else None

        async def _gated(index: int, item: Any) -> Any:
            runner = MapItemRunner(self, item, index, merge_lock)
            done = self.done.get(runner.key)
            if done is not None:
                return done.result if done.merged else await runner.merge_saved(done.result)
            if sem is None:
                return await runner.run()
            async with sem:
                return await runner.run()

        coros = [_gated(i, item) for i, item in enumerate(items)]
        if not self.mp.strict:
            return list(await asyncio.gather(*coros))
        gathered = list(await asyncio.gather(*coros, return_exceptions=True))
        for r in gathered:
            if isinstance(r, BaseException):
                raise r
        return gathered

    async def _concurrency(self, items: list[Any]) -> int | None:
        """The cap on in-flight items: ``max_concurrency``, computed now when it is a callable.

        A computed cap is not saved: a resumed map computes it again. It is
        not computed when no item is left to run.

        Raises:
            ValueError: The callable returned anything but an ``int >= 1``.
        """
        cap = self.mp.max_concurrency
        if cap is None or isinstance(cap, int):
            return cap
        if all(k in self.done for k in self.keys):
            return None
        value = cap(items, self._build_ctx())
        if inspect.isawaitable(value):
            value = await value
        return check_concurrency(value, f"map {self.mp.name or self.node_id!r}: max_concurrency")

    async def _aggregate(self, results: list[Any]) -> Any:
        """Apply ``mp.aggregate`` (if attached); await when it returns a coroutine."""
        if self.mp.aggregate is None:
            return results
        result = self.mp.aggregate(results)
        if inspect.isawaitable(result):
            result = await result
        return result


class MapItemRunner:
    """Run one map item through halt-check → project → guard → body → merge.

    Terminal states:

    - :class:`Skipped` — the guard predicate returned falsy; recorded as
      done in the map's cursor, so a resumed map does not ask again.
    - :class:`Failure` — the body (or projection / guard) raised;
      non-strict returns the sentinel and records it as done (a resumed
      map does not run the item again), strict re-raises after firing
      hooks and records nothing: the item runs again from the last save.
    - Body's return value — the successful path; recorded as completed
      in the map's cursor, merges the child state into the parent (under
      the shared lock) and saves a per-item boundary commit when policy
      asks.
    - :data:`_INTERRUPTED` — the halt stopped the item before it
      completed (before its body ran, or inside it).

    ``on_item_complete`` fires at every terminal state but the last: an
    interrupted item has not completed, and continues on resume.
    Cancellation is unconditional and never fires the hook.
    """

    def __init__(
        self, owner: MapRunner, item: Any, item_index: int, merge_lock: asyncio.Lock
    ) -> None:
        self.owner = owner
        self.mp = owner.mp
        self.env = owner.env
        self.node_id = owner.node_id
        self.item = item
        self.item_index = item_index
        self.key = owner.keys[item_index]
        self.body = self.mp.bodies[item_index] if self.mp.over_members else self.mp.bodies[0]
        self.merge_lock = merge_lock
        self.path: ScopePath = (*owner.path, "i", self.key)

    async def run(self) -> Any:
        """Drive this item through the run pipeline.

        Halt is checked first (before projection). Projection,
        guard, and body exceptions are wrapped per the parent map's
        strict/non-strict contract. An item the halt stops — before it
        starts, or inside its body — returns :data:`_INTERRUPTED` without
        firing ``on_item_complete``: it did not complete. One stopped
        inside its body — by the halt, or raising out of a strict map —
        stays registered where it stopped (its scope and its body's
        cursors), for the run's halt checkpoint and any checkpoint a
        sibling takes meanwhile. In a map that fast-forwards an item that
        had not started is :class:`Skipped` (:meth:`_shortcut_skips`).
        """
        if is_halt_signaled(self.env):
            return _INTERRUPTED
        if self._shortcut_skips():
            skipped = Skipped(item=self.item)
            self.owner.done[self.key] = _Done(skipped, merged=True)
            await self._fire_on_item_complete(skipped, self._ctx(self.env.state))
            return skipped
        try:
            outcome = await self._run_item()
        except Interrupted:
            return _INTERRUPTED
        # Merged back, skipped or failed: the item leaves the snapshot. A
        # skipped or failed one is recorded as done in the same step — no
        # await between — so a checkpoint holds its positions or its record,
        # never both. (A completed one was recorded when its body returned.)
        self.env.scopes.close_under(self.path)
        if isinstance(outcome, Failure | Skipped):
            self.owner.done[self.key] = _Done(outcome, merged=True)
        return outcome

    def _shortcut_skips(self) -> bool:
        """True when this map fast-forwards and the item had not started.

        It fast-forwards in a region whose signal is set
        (:func:`~._shortcut.is_fast_forward`). An item a checkout saved as
        running (its positions are saved) continues from them.
        """
        if not is_fast_forward(self.env):
            return False
        return not self.env.scopes.has_saved_under(self.path)

    async def merge_saved(self, result: Any) -> Any:
        """Finish an item whose body completed before the checkpoint but whose merge had not.

        Merges the item's saved scope and fires ``on_item_complete``; the
        body does not run again. Without the saved scope the merge cannot
        happen, and the item runs from the start.
        """
        found, raw = self.env.scopes.take_saved(self.path)
        if not found:
            self.owner.done.pop(self.key, None)
            return await self.run()
        factory = self.mp.state_factory or self.env.state._factory
        child_state = _restore_scope_state(self.env.state, raw, factory)
        # Registered until it merges, like the scope of an item that runs.
        self.env.scopes.open(self.path, child_state)
        try:
            return await self._on_success(result, child_state, self._ctx(child_state))
        finally:
            self.env.scopes.close(self.path)

    async def _run_item(self) -> Any:
        """Project the item's scope, then guard, body and merge per the map's contract."""
        item_ctx = self._ctx(self.env.state)
        try:
            child_state = await _enter_scope(
                self.env, self.path, self.mp.state_fn, self.mp.state_factory
            )
            if self.mp.state_fn is not None:
                self.env.scopes.open(self.path, child_state)
            item_ctx = self._ctx(child_state)
            if self.mp.guard is not None and not await _run_guard(
                self.mp.guard, self.item, item_ctx
            ):
                skipped = Skipped(item=self.item)
                await self._fire_on_item_complete(skipped, item_ctx)
                return skipped
            result = await self._dispatch_body(child_state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.mp.on_error is not None:
                await self._run_on_error(exc, item_ctx)
            failure = Failure(exception=exc, item=self.item)
            await self._fire_on_item_complete(failure, item_ctx)
            if self.mp.strict:
                raise
            return failure
        return await self._on_success(result, child_state, item_ctx)

    async def _dispatch_body(self, child_state: State[Any]) -> Any:
        """Run the body subflow with per-item composition-tree identity.

        Each item descends with a distinct ``chain_context`` keyed
        by its key, so nested iterates produce unique node IDs per item,
        and under its own path (``self.path``), so its scopes and cursors
        are distinct in every snapshot.
        """
        env = self.env
        return await self.body._run_as_subflow(
            self.item,
            state=child_state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_resources=env.resources,
            parent_checkpoint_ctx=env.checkpoint_ctx,
            parent_checkpointer=env.checkpointer,
            parent_chain_context=_descend_context(self.node_id, f"map:{self.key}"),
            parent_ancestor_chain=env.ancestor_chain + (self.node_id,),
            parent_extra=env.extra,
            parent_policy=env.policy,
            parent_path=self.path,
            # A started item runs to its end however its drain regions are cut.
            parent_shortcuts=leave_drains(env.shortcuts),
        )

    async def _on_success(
        self, result: Any, child_state: State[Any], item_ctx: Context[Any]
    ) -> Any:
        """Merge, save-per-policy, fire on_item_complete; return the body result.

        Both strict and non-strict converge here on the successful
        body path. Merge-time and checkpoint-save failures are handled
        per-mode: strict re-raises the underlying exception; non-strict
        returns the :class:`Failure` sentinel. The hook fires exactly
        once with the final outcome — either the body result or a
        :class:`Failure` wrapping the merge/save exception.

        When ``on_map_item`` checkpointing is enabled, the merge is
        atomic with the checkpoint write: a snapshot is taken before
        merge, and on checkpoint failure the parent state is rolled
        back so concurrent items don't serialize an uncommitted merge.
        Without ``state=`` there is no merge (``merge`` requires it) and
        items write the shared state directly, so no snapshot is taken:
        restoring one would erase what siblings wrote while the commit
        was in flight.

        The map's cursor records the item as completed as soon as its body
        returns, and as merged in the same step as the merge — no await
        between — so every checkpoint holds a merge together with its
        record: a completed item never runs again, and its merge is
        applied exactly once.
        """
        scoped = self.mp.state_fn is not None
        self.owner.done[self.key] = _Done(result, merged=not scoped)
        rollback: list[Any] = []
        try:
            await self._merge_and_save(child_state, scoped, rollback)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.owner.done.pop(self.key, None)
            if rollback:
                _restore_state_data(self.env.state.data, rollback[0])
            if self.mp.on_error is not None:
                await self._run_on_error(exc, item_ctx)
            failure = Failure(exception=exc, item=self.item)
            await self._fire_on_item_complete(failure, item_ctx)
            if self.mp.strict:
                raise
            return failure
        await self._fire_on_item_complete(result, item_ctx)
        return result

    async def _merge_and_save(
        self, child_state: State[Any], scoped: bool, rollback: list[Any]
    ) -> None:
        """Under the merge lock: merge, record it in the map's cursor, save per policy.

        With a policy save of a scoped item, the parent's data before the
        merge is appended to ``rollback`` for the caller to restore if the
        merge or the save fails.
        """
        async with self.merge_lock:
            if self.env.policy.on_map_item and scoped:
                # serialize_state_data passes a dict through as-is: copy it,
                # or the merge mutates the snapshot the rollback restores from.
                rollback.append(copy.deepcopy(serialize_state_data(self.env.state.data)))
            await _merge_state(self.mp.merge_fn, self.env.state, child_state)
            self.owner.done[self.key].merged = True
            self.env.scopes.close(self.path)
            if self.env.policy.on_map_item:
                await _save_scope_commit(
                    self.env, self.item_index, self.node_id, self.env.state, "ok"
                )

    async def _run_on_error(self, exc: BaseException, ctx: Context[Any]) -> None:
        """Invoke on_error and swallow any exception it raises.

        An error hook that raises must never mask the original
        exception the map body threw.
        """
        assert self.mp.on_error is not None
        try:
            result = self.mp.on_error(exc, self.item, ctx)
            if inspect.isawaitable(result):
                await result
        except Exception as hook_exc:
            self.env.lg.warning(
                "map on_error hook raised — original exception preserved",
                extra={"exception": hook_exc, "original": exc},
            )

    async def _fire_on_item_complete(self, outcome: Any, ctx: Context[Any]) -> None:
        """Invoke on_item_complete (if attached) and swallow hook exceptions.

        An observer that raises must never mask the outcome that
        lands in the map's result list. Matches
        :meth:`_run_on_error` semantics.
        """
        hook = self.mp.on_item_complete
        if hook is None:
            return
        try:
            result = hook(self.item, outcome, ctx)
            if inspect.isawaitable(result):
                await result
        except Exception as hook_exc:
            self.env.lg.warning(
                "map on_item_complete hook raised — outcome preserved",
                extra={"exception": hook_exc},
            )

    def _ctx(self, child_state: Any) -> Context[Any]:
        """Build the per-item :class:`Context` fed to guard / on_error / on_item_complete.

        These hooks run without a :class:`Role`, so ``ctx.saia`` is
        ``None``. The map node's own ``node_id`` is threaded so a
        ``ctx.checkpoint()`` from a map hook addresses the map's
        position.
        """
        env = self.env
        return Context(
            role=None,
            state=child_state,
            flow=env.runtime,
            traits=env.runtime._traits,
            halt=env.halt,
            extra=env.extra,
            resources=env.resources,
            _env=env,
            _node_id=self.node_id,
        )


async def _resolve_items(
    items_fn: ItemsFn | None, prev_result: Any, ctx: Context[Any]
) -> list[Any]:
    """Materialize the map input list from ``items_fn`` (or ``prev_result``)."""
    source = prev_result if items_fn is None else items_fn(prev_result, ctx)
    if inspect.isawaitable(source):
        source = await source
    try:
        return list(source)
    except TypeError as exc:
        raise TypeError(f".map items must be iterable; got {type(source).__name__}") from exc


def _restore_state_data(data: Any, snapshot: Any) -> None:
    """Restore state data in-place from a serialized snapshot.

    Used to roll back a failed checkpoint: after ``_merge_state`` has
    mutated the parent's data, a checkpoint-write failure should leave
    the parent unchanged so concurrent items don't serialize a merge
    that was never committed.

    Dicts are restored via clear+update. Objects with a ``from_dict``
    classmethod (the :class:`StateData` contract) are restored by
    reconstructing from the snapshot and copying attributes. Payloads
    that satisfy neither are unsupported under checkpoint atomicity.
    """
    if isinstance(data, dict):
        data.clear()
        data.update(snapshot)
        return
    from_dict = getattr(type(data), "from_dict", None)
    if callable(from_dict):
        restored = from_dict(snapshot)
        for key, value in vars(restored).items():
            object.__setattr__(data, key, value)
        return
    raise TypeError(
        f"cannot restore state.data of type {type(data).__name__} — "
        f"payload must be a plain dict or satisfy StateData"
    )


async def _run_guard(guard_fn: Any, item: Any, ctx: Context[Any]) -> bool:
    """Evaluate the guard predicate, awaiting when async, coercing to bool."""
    verdict = guard_fn(item, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)
