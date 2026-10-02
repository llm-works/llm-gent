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
from .context import Context
from .nodes import Failure, Interrupted, ItemsFn, Skipped
from .state import serialize_state_data
from .state.snapshot import DONE, ITEMS, ScopePath


if TYPE_CHECKING:
    from .nodes import _Map, _RunEnv
    from .state import State


_INTERRUPTED = object()
"""Result of a map item the halt stopped before it completed."""


@dataclass
class _Done:
    """A completed item: its result, and whether its scope merged into the parent yet."""

    result: Any
    merged: bool


class MapRunner:
    """Fan out one ``.map`` node's body over its items.

    Constructed with (mp, env, node_id). Public method :meth:`run` takes
    the incoming ``node_args`` and returns the aggregated result (or the
    raw list when no aggregator is attached).

    Its cursor — the items it runs over and the completed ones with their
    results — is state on the runner, in every snapshot taken while it
    runs. Items that are running keep their own positions under
    ``<map>/i/<index>``. On ``resume="latest"`` the map runs over the saved
    items: a completed item does not run again, a running one continues
    where it was, and the rest run.
    """

    def __init__(self, mp: _Map, env: _RunEnv, node_id: str, path: ScopePath | None = None) -> None:
        """Run ``mp`` as the step ``node_id`` of ``env``'s flow, at ``path`` (default: the step's).

        A map that is not a step of its own (a :class:`~llm_gent.flow.Panel`
        inside a verb) passes the path it runs at.
        """
        self.mp = mp
        self.env = env
        self.node_id = node_id
        self.path: ScopePath = env.owner_path(node_id) if path is None else path
        self.is_step = path is None
        # The cursor. Kept here, not in run's locals, so a checkpoint reads it.
        self.items: list[Any] = []
        self.done: dict[int, _Done] = {}

    def cursor(self) -> dict[str, Any]:
        """The items and the completed ones: index to ``{"result", "merged"}``."""
        done = {str(i): {"result": d.result, "merged": d.merged} for i, d in self.done.items()}
        return {ITEMS: self.items, DONE: done}

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
        since what it reads may have changed since the map started.
        """
        scopes = self.env.scopes
        found, items = scopes.take_cursor(self.path, ITEMS)
        if found:
            self.items = items
            _, done = scopes.take_cursor(self.path, DONE)
            self.done = {int(i): _Done(d["result"], d["merged"]) for i, d in (done or {}).items()}
            return
        prev_result = node_args[0] if node_args else None
        self.items = await _resolve_items(self.mp.items, prev_result, self._build_ctx())

    def _build_ctx(self) -> Context[Any]:
        """Build the :class:`Context` passed to ``items_fn``.

        Uses the parent state at map entry. Role is ``None`` since
        ``items_fn`` is a data-producing callback, not a role action.
        """
        env = self.env
        return Context(
            role=None,
            state=env.state,
            flow=env.runtime,
            traits=env.runtime._traits,
            halt=env.halt,
            cost=env.cost,
            extra=env.extra,
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
        sem = (
            asyncio.Semaphore(self.mp.max_concurrency)
            if self.mp.max_concurrency is not None
            else None
        )

        async def _gated(index: int, item: Any) -> Any:
            runner = self._item_runner(item, index, merge_lock)
            done = self.done.get(index)
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

    def _item_runner(self, item: Any, index: int, merge_lock: asyncio.Lock) -> MapItemRunner:
        """The runner for item ``index``; a subclass runs a different body per item."""
        return MapItemRunner(self, item, index, merge_lock)

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

    - :class:`Skipped` — the guard predicate returned falsy.
    - :class:`Failure` — the body (or projection / guard) raised;
      non-strict returns the sentinel, strict re-raises after
      firing hooks.
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
        self.merge_lock = merge_lock
        self.path: ScopePath = (*owner.path, "i", str(item_index))

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
        sibling takes meanwhile. In shortcut mode an item that had not
        started is :class:`Skipped` (:meth:`_shortcut_skips`).
        """
        if is_halt_signaled(self.env):
            return _INTERRUPTED
        if self._shortcut_skips():
            skipped = Skipped(item=self.item)
            await self._fire_on_item_complete(skipped, self._ctx(self.env.state))
            return skipped
        try:
            outcome = await self._run_item()
        except Interrupted:
            return _INTERRUPTED
        # Merged back, skipped or failed: the item leaves the snapshot.
        self.env.scopes.close_under(self.path)
        return outcome

    def _shortcut_skips(self) -> bool:
        """True when this map is a step of a flow in shortcut mode and the item had not started.

        An item that was running when the flow stopped (its positions are
        saved) continues from them. A map run by a Panel inside a step is
        not a step of the flow: its items run.
        """
        shortcut = self.env.shortcut
        if shortcut is None or not shortcut.active or not self.owner.is_step:
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
            self.owner.done.pop(self.item_index, None)
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
        by ``item_index``, so nested iterates produce unique node
        IDs per item, and under its own path (``self.path``), so its
        scopes and cursors are distinct in every snapshot.
        """
        env = self.env
        return await self.mp.body._run_as_subflow(
            self.item,
            state=child_state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_cost=env.cost,
            parent_checkpoint_ctx=env.checkpoint_ctx,
            parent_checkpointer=env.checkpointer,
            parent_chain_context=_descend_context(self.node_id, f"map:{self.item_index}"),
            parent_ancestor_chain=env.ancestor_chain + (self.node_id,),
            parent_extra=env.extra,
            parent_policy=env.policy,
            parent_path=self.path,
            parent_shortcuts=env.shortcuts,
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
        self.owner.done[self.item_index] = _Done(result, merged=not scoped)
        rollback: list[Any] = []
        try:
            await self._merge_and_save(child_state, scoped, rollback)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.owner.done.pop(self.item_index, None)
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
            self.owner.done[self.item_index].merged = True
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
            cost=env.cost,
            extra=env.extra,
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
