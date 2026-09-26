# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Map-node dispatcher: fan-out + per-item run with guards, hooks, and merges.

Extracted from :mod:`._executor`. Two classes:

- :class:`MapRunner` resolves the input items (via ``items_fn`` or
  the incoming ``prev_result``), pre-resolves which item (if any)
  inherits the resume replay, spawns one :class:`MapItemRunner` per
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
import inspect
from typing import TYPE_CHECKING, Any

from ._executor import (
    _merge_state,
    _pop_replay_for,
    _project_state,
    _save_scope_commit,
    _serialize_state_data,
)
from ._halt_observer import is_halt_signaled
from ._node_id import _compute_node_id, _descend_context
from .context import Context
from .nodes import Failure, ItemsFn, Skipped


if TYPE_CHECKING:
    from .nodes import _Map, _ResumeReplay, _RunEnv
    from .state import State


class MapRunner:
    """Fan out one ``.map`` node's body over its items.

    Constructed with (mp, env, node_id). Public method
    :meth:`run` takes the map node's :class:`Context` (needed to
    thread into ``items_fn`` when the user supplied one) plus the
    incoming ``node_args``, returns the aggregated result (or the
    raw list when no aggregator is attached).
    """

    def __init__(self, mp: _Map, env: _RunEnv, node_id: str) -> None:
        self.mp = mp
        self.env = env
        self.node_id = node_id

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
        not yet passed its halt check short-circuits to
        :class:`Skipped`, so the remaining queue drains without
        running any more bodies. Already-in-flight items complete.
        """
        prev_result = node_args[0] if node_args else None
        ctx = self._build_ctx()
        items = await _resolve_items(self.mp.items, prev_result, ctx)
        results = await self._gather_items(items)
        return await self._aggregate(results)

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
            budget=env.budget,
            extra=env.extra,
            _env=env,
            _node_id=self.node_id,
        )

    async def _gather_items(self, items: list[Any]) -> list[Any]:
        """Spawn one runner per item under the concurrency cap; return per-item results.

        Strict mode re-raises the first non-cancellation exception;
        non-strict returns every item's result (or :class:`Failure`
        sentinel).
        """
        merge_lock = asyncio.Lock()
        sem = (
            asyncio.Semaphore(self.mp.max_concurrency)
            if self.mp.max_concurrency is not None
            else None
        )
        item_replays = self._resolve_item_replays(len(items))

        async def _gated(index: int, item: Any) -> Any:
            replay = item_replays.get(index)
            runner = MapItemRunner(self.mp, self.env, self.node_id, item, index, replay, merge_lock)
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

    async def _aggregate(self, results: list[Any]) -> Any:
        """Apply ``mp.aggregate`` (if attached); await when it returns a coroutine."""
        if self.mp.aggregate is None:
            return results
        result = self.mp.aggregate(results)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _resolve_item_replays(self, item_count: int) -> dict[int, _ResumeReplay | None]:
        """Pre-resolve which item (if any) inherits the replay.

        Pop replay once at the map boundary. Then, for each item,
        check if the popped path's head matches any node_id in that
        item's body. Only the matching item gets the replay; all
        others get ``None``. This prevents scheduling-dependent
        replay failures when concurrent map items race to validate
        the path.
        """
        result: dict[int, _ResumeReplay | None] = {}
        popped = _pop_replay_for(self.env, self.node_id)
        if popped is None or not popped.remaining_path:
            return result
        head = popped.remaining_path[0]
        for i in range(item_count):
            item_ctx = _descend_context(self.node_id, f"map:{i}")
            item_ids = tuple(
                _compute_node_id(item_ctx, n, j) for j, n in enumerate(self.mp.body._nodes)
            )
            if head in item_ids:
                result[i] = popped
                break
        return result


class MapItemRunner:
    """Run one map item through halt-check → project → guard → body → merge.

    Terminal states:

    - :class:`Skipped` — halt was signaled before the body ran, or
      the guard predicate returned falsy.
    - :class:`Failure` — the body (or projection / guard) raised;
      non-strict returns the sentinel, strict re-raises after
      firing hooks.
    - Body's return value — the successful path; merges the child
      state into the parent (under the shared lock) and saves a
      per-item boundary commit when policy asks.

    ``on_item_complete`` fires at every terminal state.
    Cancellation is unconditional and never fires the hook.
    """

    def __init__(
        self,
        mp: _Map,
        env: _RunEnv,
        node_id: str,
        item: Any,
        item_index: int,
        replay: _ResumeReplay | None,
        merge_lock: asyncio.Lock,
    ) -> None:
        self.mp = mp
        self.env = env
        self.node_id = node_id
        self.item = item
        self.item_index = item_index
        self.replay = replay
        self.merge_lock = merge_lock

    async def run(self) -> Any:
        """Drive this item through the run pipeline.

        Halt is checked first (before projection). Projection,
        guard, and body exceptions are wrapped per the parent map's
        strict/non-strict contract.
        """
        if is_halt_signaled(self.env):
            skipped = Skipped(item=self.item)
            await self._fire_on_item_complete(skipped, self._ctx(self.env.state))
            return skipped
        item_ctx = self._ctx(self.env.state)
        try:
            child_state = await _project_state(
                self.mp.state_fn, self.env.state, self.mp.state_factory
            )
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
        IDs per item — no checkpoint overwrites and no replay race
        conditions when the map body contains checkpointed
        primitives. ``replay`` was pre-resolved at the map boundary
        (see :meth:`MapRunner._resolve_item_replays`) — only the
        item whose body contains the replay's path head receives a
        non-None value.
        """
        env = self.env
        return await self.mp.body._run_as_subflow(
            self.item,
            state=child_state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_budget=env.budget,
            parent_checkpointer=env.checkpointer,
            parent_client_flow_id=env.client_flow_id,
            parent_chain_context=_descend_context(self.node_id, f"map:{self.item_index}"),
            parent_ancestor_chain=env.ancestor_chain + (self.node_id,),
            parent_replay=self.replay,
            parent_extra=env.extra,
            parent_policy=env.policy,
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
        """
        snapshot = None
        try:
            async with self.merge_lock:
                if self.env.policy.on_map_item:
                    snapshot = _serialize_state_data(self.env.state.data)
                await _merge_state(self.mp.merge_fn, self.env.state, child_state)
                if self.env.policy.on_map_item:
                    await _save_scope_commit(
                        self.env, self.item_index, self.node_id, self.env.state, "ok"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if snapshot is not None:
                _restore_state_data(self.env.state.data, snapshot)
            if self.mp.on_error is not None:
                await self._run_on_error(exc, item_ctx)
            failure = Failure(exception=exc, item=self.item)
            await self._fire_on_item_complete(failure, item_ctx)
            if self.mp.strict:
                raise
            return failure
        await self._fire_on_item_complete(result, item_ctx)
        return result

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
            budget=env.budget,
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
