# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow execution helpers — walk the composition graph, dispatch each node.

Internal to :mod:`llm_gent.flow`: the coroutines that :meth:`Flow.run` calls
into to execute each node in order (verbs, subflows, and the
branch/iterate/map control-flow primitives), together with the scoped-state
projection/merge plumbing.

Depends on :mod:`.flow` for the :class:`Flow` class itself, which appears in
three ``isinstance`` checks used to distinguish subflow nodes from verb
nodes and control-flow primitives (:func:`_build_ctx`,
:func:`_invoke_target`, :func:`_target_label`). Each uses a localized late
import (``from .flow import Flow`` inside the function body) to break the
circular dependency between the executor and the class it operates on.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any

from .context import Context
from .nodes import (
    UNSET,
    Failure,
    ItemsFn,
    OnErrorFn,
    Skipped,
    StateMerge,
    StateProject,
    UntilFn,
    _Branch,
    _Iterate,
    _Map,
    _Node,
    _RunEnv,
)
from .state import State


if TYPE_CHECKING:
    from .flow import Flow


def _build_ctx(target: Any, env: _RunEnv) -> Context:
    """Build the Context passed to the node's verb (and to its hooks).

    Verb nodes get a role-bound ctx; ``ctx.saia`` resolves lazily on first
    read via the flow's SAIAFactory. Subflow nodes and control-flow nodes
    (branch/iterate/map) get an ambient ctx with ``role=None`` — those nodes
    have no single role, so ``ctx.saia`` returns ``None`` (each inner verb
    builds its own role-bound ctx as it runs).
    """
    from .flow import Flow

    traits = env.runtime._traits
    if isinstance(target, Flow | _Branch | _Iterate | _Map):
        return Context(role=None, state=env.state, flow=env.runtime, traits=traits, halt=env.halt)
    return Context(
        role=target.role, state=env.state, flow=env.runtime, traits=traits, halt=env.halt
    )


async def _execute_node(
    node: _Node,
    ctx: Context,
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
) -> Any:
    """Invoke a node's target with cancellation-safe rescue + optional after hook.

    ``asyncio.CancelledError`` propagates unchanged past any rescue policy.
    Other exceptions surface unless a rescue is attached; the rescue's
    return value (awaited if awaitable) becomes the node's result.
    """
    target_name = _target_label(node.target)
    env.lg.debug("executing node", extra={"target": target_name})
    try:
        result = await _invoke_target(node, ctx, env, node_args, node_kwargs)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if node.rescue is None:
            raise
        env.lg.warning("rescue policy invoked", extra={"target": target_name, "exception": exc})
        pending_input = node_args[0] if node_args else UNSET
        fallback = node.rescue(exc, pending_input, ctx)
        if inspect.isawaitable(fallback):
            fallback = await fallback
        result = fallback
    if node.after is not None:
        env.lg.debug("running after hook", extra={"target": target_name})
        hook_result = node.after(result, ctx)
        if inspect.isawaitable(hook_result):
            await hook_result
    return result


async def _invoke_target(
    node: _Node,
    ctx: Context,
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
) -> Any:
    """Dispatch a node's target: verb, subflow, or control-flow primitive."""
    from .flow import Flow

    target = node.target
    if isinstance(target, Flow):
        return await _run_subflow(target, env, node.state_fn, node.merge_fn, node_args, node_kwargs)
    if isinstance(target, _Branch):
        return await _run_branch(target, ctx, env, node_args)
    if isinstance(target, _Iterate):
        return await _run_iterate(target, ctx, env, node_args)
    if isinstance(target, _Map):
        return await _run_map(target, ctx, env, node_args)
    return await target(ctx, *node_args, **node_kwargs)


async def _run_subflow(
    body: Flow,
    env: _RunEnv,
    state_fn: StateProject | None,
    merge_fn: StateMerge | None,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
) -> Any:
    """Run a subflow node, honoring optional scoped-state projection/merge."""
    child_state = await _project_state(state_fn, env.state)
    result = await body._run_as_subflow(
        *node_args,
        state=child_state,
        runtime=env.runtime,
        parent_halt=env.halt,
        **node_kwargs,
    )
    await _merge_state(merge_fn, env.state, child_state)
    return result


async def _project_state(state_fn: StateProject | None, parent: State) -> State:
    """Build the child :class:`State` for a scoped block; pass-through when unset.

    With no projection, the subflow sees the parent's :class:`State` object
    directly — same reference, shared payload, ``is_root`` echoes the parent.
    With a projection, ``state_fn(parent.data)`` produces the child payload,
    which the framework wraps as ``State(data=child_payload, _parent=parent)``
    so the child's :meth:`State.root` still walks back to the outermost scope.
    """
    if state_fn is None:
        return parent
    child_payload = state_fn(parent.data)
    if inspect.isawaitable(child_payload):
        child_payload = await child_payload
    return State(data=child_payload, _parent=parent)


async def _merge_state(merge_fn: StateMerge | None, parent: State, child: State) -> None:
    """Fold the child payload back into the parent's; no-op when unset.

    The merge callback receives the two payloads (``parent.data``,
    ``child.data``); the :class:`State` wrappers are unwrapped for the user
    so signatures match the pre-unification shape.
    """
    if merge_fn is None:
        return
    result = merge_fn(parent.data, child.data)
    if inspect.isawaitable(result):
        await result


async def _check_until(until_fn: UntilFn | None, iterate_state: State, env: _RunEnv) -> bool:
    """Evaluate the iterate node's until predicate with a ctx bound to its scoped state."""
    if until_fn is None:
        return False
    ctx = Context(
        role=None,
        state=iterate_state,
        flow=env.runtime,
        traits=env.runtime._traits,
        halt=env.halt,
    )
    verdict = until_fn(ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_branch(
    br: _Branch,
    ctx: Context,
    env: _RunEnv,
    node_args: tuple[Any, ...],
) -> Any:
    """Evaluate the predicate and dispatch the chosen subflow.

    Passes the branch input (``node_args[0]``, or ``None`` when the branch
    is the chain's head with no positional) as the sole positional to the
    chosen subflow. Falsy predicate with no ``else_`` returns the input
    unchanged. Both bodies share the parent's active state.
    """
    prev_result = node_args[0] if node_args else None
    verdict = br.when(prev_result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    chosen = br.then_flow if verdict else br.else_flow
    if chosen is None:
        return prev_result
    return await chosen._run_as_subflow(
        prev_result, state=env.state, runtime=env.runtime, parent_halt=env.halt
    )


async def _run_iterate(
    it: _Iterate,
    ctx: Context,
    env: _RunEnv,
    node_args: tuple[Any, ...],
) -> Any:
    """Iterate the body under bounds, threading each result to the next.

    Post-check semantics: the body runs at least once, then ``until`` (if
    set) is evaluated. ``max_iters`` and ``deadline`` bound the total
    iteration count and elapsed wall clock respectively; an ambient halt
    event (via :meth:`Flow.with_halt`) is checked between iterations, so a
    running body is not interrupted mid-request. Scoped state is projected
    once before the first iteration; every iteration sees the same child
    state, and the merge fires once after the block exits successfully.
    """
    child_state = await _project_state(it.state_fn, env.state)
    result: Any = node_args[0] if node_args else None
    started = time.monotonic()
    iteration = 0
    while True:
        if it.max_iters is not None and iteration >= it.max_iters:
            break
        if it.deadline is not None and time.monotonic() - started >= it.deadline:
            break
        if env.halt is not None and env.halt.is_set():
            break
        result = await it.body._run_as_subflow(
            result, state=child_state, runtime=env.runtime, parent_halt=env.halt
        )
        iteration += 1
        if await _check_until(it.until, child_state, env):
            break
    await _merge_state(it.merge_fn, env.state, child_state)
    return result


async def _run_map(
    mp: _Map,
    ctx: Context,
    env: _RunEnv,
    node_args: tuple[Any, ...],
) -> Any:
    """Fan out the body over items concurrently, then (optionally) aggregate.

    ``strict=True`` re-raises the first non-cancellation exception; sibling
    tasks continue but their results are discarded. ``strict=False`` swaps
    each failing item for a :class:`Failure` sentinel so the caller sees
    every position. A :meth:`Flow.guard` predicate (if attached) skips
    items before the body runs and replaces them with a :class:`Skipped`
    sentinel. An :meth:`Flow.on_error` hook (if attached) fires for each
    per-item exception under both strict modes without altering control
    flow. Cancellation propagates unconditionally regardless.

    ``max_concurrency=N`` caps in-flight per-item runners via an
    ``asyncio.Semaphore``; items above the cap wait for a slot. An ambient
    :meth:`Flow.with_halt` event, if set, short-circuits any per-item runner
    that has not yet passed its halt check to :class:`Skipped`, so once the
    event fires the remaining queue drains without running any more bodies.
    Already-in-flight items complete.

    Scoped state is projected per item; per-item merges run only for items
    that completed successfully (skipped and errored items do not merge).
    """
    prev_result = node_args[0] if node_args else None
    items = await _resolve_items(mp.items, prev_result, ctx)
    merge_lock = asyncio.Lock()
    runner = _run_map_item_strict if mp.strict else _run_map_item
    sem = asyncio.Semaphore(mp.max_concurrency) if mp.max_concurrency is not None else None

    async def _gated(item: Any) -> Any:
        if sem is None:
            return await runner(mp, item, env, merge_lock)
        async with sem:
            return await runner(mp, item, env, merge_lock)

    coros = [_gated(item) for item in items]
    if mp.strict:
        results = list(await asyncio.gather(*coros, return_exceptions=True))
        for r in results:
            if isinstance(r, BaseException):
                raise r
    else:
        results = list(await asyncio.gather(*coros))
    result = mp.aggregate(results) if mp.aggregate is not None else results
    if inspect.isawaitable(result):
        result = await result
    return result


async def _run_map_item_strict(
    mp: _Map,
    item: Any,
    env: _RunEnv,
    merge_lock: asyncio.Lock,
) -> Any:
    """Run one strict-mode map item; merge fires only when the body succeeds.

    Halt is checked first (before projection). Projection, guard, and body
    exceptions all propagate; ``on_error`` fires before the exception
    escapes.
    """
    if env.halt is not None and env.halt.is_set():
        return Skipped(item=item)
    item_ctx = _map_item_ctx(env, env.state)
    try:
        child_state = await _project_state(mp.state_fn, env.state)
        item_ctx = _map_item_ctx(env, child_state)
        if mp.guard is not None and not await _run_guard(mp.guard, item, item_ctx):
            return Skipped(item=item)
        result = await mp.body._run_as_subflow(
            item, state=child_state, runtime=env.runtime, parent_halt=env.halt
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if mp.on_error is not None:
            await _run_on_error(mp.on_error, exc, item, item_ctx, env)
        raise
    async with merge_lock:
        await _merge_state(mp.merge_fn, env.state, child_state)
    return result


async def _run_map_item(
    mp: _Map,
    item: Any,
    env: _RunEnv,
    merge_lock: asyncio.Lock,
) -> Any:
    """Run one non-strict map item; wrap non-cancellation exceptions as :class:`Failure`.

    Halt is checked first (before projection). Projection, guard, and body
    exceptions are all wrapped as :class:`Failure` and passed to
    ``on_error``. Merge fires only for items that complete successfully —
    a failed or skipped item's partially-mutated child state is discarded.
    """
    if env.halt is not None and env.halt.is_set():
        return Skipped(item=item)
    item_ctx = _map_item_ctx(env, env.state)
    try:
        child_state = await _project_state(mp.state_fn, env.state)
        item_ctx = _map_item_ctx(env, child_state)
        if mp.guard is not None and not await _run_guard(mp.guard, item, item_ctx):
            return Skipped(item=item)
        result = await mp.body._run_as_subflow(
            item, state=child_state, runtime=env.runtime, parent_halt=env.halt
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if mp.on_error is not None:
            await _run_on_error(mp.on_error, exc, item, item_ctx, env)
        return Failure(exception=exc, item=item)
    async with merge_lock:
        await _merge_state(mp.merge_fn, env.state, child_state)
    return result


def _map_item_ctx(env: _RunEnv, child_state: Any) -> Context:
    """Build the per-item :class:`Context` fed to guard and on_error hooks.

    These hooks run without a :class:`Role`, so ``ctx.saia`` is ``None``.
    """
    return Context(
        role=None,
        state=child_state,
        flow=env.runtime,
        traits=env.runtime._traits,
        halt=env.halt,
    )


async def _run_guard(guard_fn: Any, item: Any, ctx: Context) -> bool:
    """Evaluate the guard predicate, awaiting when async, coercing to bool."""
    verdict = guard_fn(item, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_on_error(
    on_error_fn: OnErrorFn,
    exc: BaseException,
    item: Any,
    ctx: Context,
    env: _RunEnv,
) -> None:
    """Invoke on_error and swallow any exception it raises (never mask the original)."""
    try:
        result = on_error_fn(exc, item, ctx)
        if inspect.isawaitable(result):
            await result
    except Exception as hook_exc:
        env.lg.warning(
            "map on_error hook raised — original exception preserved",
            extra={"exception": hook_exc, "original": exc},
        )


async def _resolve_items(items_fn: ItemsFn | None, prev_result: Any, ctx: Context) -> list[Any]:
    """Materialize the map input list from ``items_fn`` (or ``prev_result``)."""
    source = prev_result if items_fn is None else items_fn(prev_result, ctx)
    if inspect.isawaitable(source):
        source = await source
    try:
        return list(source)
    except TypeError as exc:
        raise TypeError(f".map items must be iterable; got {type(source).__name__}") from exc


def _target_label(target: Any) -> str:
    """Return a human-readable label for a node target."""
    from .flow import Flow

    if isinstance(target, Flow):
        return f"Flow({target.name!r})" if target.name else "Flow(<anonymous>)"
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Iterate):
        return "iterate"
    if isinstance(target, _Map):
        return "map"
    return getattr(target, "__name__", type(target).__name__)


def _step_inputs(
    index: int,
    node: _Node,
    prev_result: Any,
    run_args: tuple[Any, ...],
    run_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return the (args, kwargs) to feed the node's target.

    First node: forward ``run()`` inputs verbatim. Later nodes: pass the
    previous result as the single positional (through ``project`` if set).
    Later nodes never inherit ``run()`` kwargs — those are input to the
    chain's head only.
    """
    if index == 0:
        return run_args, run_kwargs
    projected = node.project(prev_result) if node.project is not None else prev_result
    return (projected,), {}
