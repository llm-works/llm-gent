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
import dataclasses
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
    _ResumeReplay,
    _RunEnv,
)
from .state import State


if TYPE_CHECKING:
    from .flow import Flow


def _build_ctx(target: Any, env: _RunEnv) -> Context[Any]:
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
        return Context(
            role=None,
            state=env.state,
            flow=env.runtime,
            traits=traits,
            halt=env.halt,
            budget=env.budget,
        )
    return Context(
        role=target.role,
        state=env.state,
        flow=env.runtime,
        traits=traits,
        halt=env.halt,
        budget=env.budget,
    )


async def _execute_node(
    node: _Node,
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Invoke a node's target with cancellation-safe rescue + optional after hook.

    ``asyncio.CancelledError`` propagates unchanged past any rescue policy.
    Other exceptions surface unless a rescue is attached; the rescue's
    return value (awaited if awaitable) becomes the node's result.
    """
    target_name = _target_label(node.target)
    env.lg.debug("executing node", extra={"target": target_name})
    try:
        result = await _invoke_target(node, ctx, env, node_args, node_kwargs, node_id)
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
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Dispatch a node's target: verb, subflow, or control-flow primitive."""
    from .flow import Flow

    target = node.target
    if isinstance(target, Flow):
        return await _run_subflow(
            target, env, node.state_fn, node.merge_fn, node_args, node_kwargs, node_id
        )
    if isinstance(target, _Branch):
        return await _run_branch(target, ctx, env, node_args, node_id)
    if isinstance(target, _Iterate):
        return await _run_iterate(target, ctx, env, node_args, node_id)
    if isinstance(target, _Map):
        return await _run_map(target, ctx, env, node_args, node_id)
    return await target(ctx, *node_args, **node_kwargs)


async def _run_subflow(
    body: Flow,
    env: _RunEnv,
    state_fn: StateProject | None,
    merge_fn: StateMerge | None,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Run a subflow node, honoring optional scoped-state projection/merge.

    Threads the composition-tree identity into the child Flow:
    ``ancestor_chain`` gains ``node_id`` (this call's ``_Node`` is now
    an ancestor of everything inside), and ``chain_context`` becomes
    ``_descend_context(node_id, "call")`` — the hash the child's
    chain-step IDs are computed against.
    """
    from .flow import _descend_context

    child_state = await _project_state(state_fn, env.state)
    result = await body._run_as_subflow(
        *node_args,
        state=child_state,
        runtime=env.runtime,
        parent_halt=env.halt,
        parent_budget=env.budget,
        parent_checkpointer=env.checkpointer,
        parent_client_flow_id=env.client_flow_id,
        parent_chain_context=_descend_context(node_id, "call"),
        parent_ancestor_chain=env.ancestor_chain + (node_id,),
        parent_replay=_pop_replay_for(env, node_id),
        **node_kwargs,
    )
    await _merge_state(merge_fn, env.state, child_state)
    return result


async def _project_state(state_fn: StateProject | None, parent: State[Any]) -> State[Any]:
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


async def _merge_state(merge_fn: StateMerge | None, parent: State[Any], child: State[Any]) -> None:
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


async def _check_until(
    until_fn: UntilFn | None,
    result: Any,
    iterate_state: State[Any],
    env: _RunEnv,
) -> bool:
    """Evaluate the iterate node's until predicate with the last body result and scoped state."""
    if until_fn is None:
        return False
    ctx = Context(
        role=None,
        state=iterate_state,
        flow=env.runtime,
        traits=env.runtime._traits,
        halt=env.halt,
        budget=env.budget,
    )
    verdict = until_fn(result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_branch(
    br: _Branch,
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_id: str,
) -> Any:
    """Evaluate the predicate and dispatch the chosen subflow.

    Passes the branch input (``node_args[0]``, or ``None`` when the branch
    is the chain's head with no positional) as the sole positional to the
    chosen subflow. Falsy predicate with no ``else_`` returns the input
    unchanged. Both bodies share the parent's active state.

    The chosen arm's ``chain_context`` is
    ``_descend_context(node_id, "then"|"else")`` so the two arms are at
    identity-distinct positions in the composition tree even when they
    share a target Flow.
    """
    from .flow import _descend_context

    prev_result = node_args[0] if node_args else None
    verdict = br.when(prev_result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    chosen = br.then_flow if verdict else br.else_flow
    if chosen is None:
        _assert_replay_allows_skip(
            env,
            node_id,
            "branch predicate returned falsy with no else_ arm, but resume "
            "path expected descent through this branch — predicate behavior "
            "has changed since checkpoint.",
        )
        return prev_result
    return await chosen._run_as_subflow(
        prev_result,
        state=env.state,
        runtime=env.runtime,
        parent_halt=env.halt,
        parent_budget=env.budget,
        parent_checkpointer=env.checkpointer,
        parent_client_flow_id=env.client_flow_id,
        parent_chain_context=_descend_context(node_id, "then" if verdict else "else"),
        parent_ancestor_chain=env.ancestor_chain + (node_id,),
        parent_replay=_pop_replay_for(env, node_id),
    )


async def _run_iterate(
    it: _Iterate,
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_id: str,
) -> Any:
    """Iterate the body under bounds, threading each result to the next.

    Post-check semantics: the body runs at least once, then ``until`` (if
    set) is evaluated. ``max_iters`` and ``deadline`` bound the total
    iteration count and elapsed wall clock respectively; an ambient halt
    event (via :meth:`Flow.with_halt`) is checked between iterations, so a
    running body is not interrupted mid-request. Scoped state is projected
    once before the first iteration; every iteration sees the same child
    state, and the merge fires once after the block exits successfully.

    Save-at-iterate-boundary: when the runtime carries a checkpointer +
    ``client_flow_id`` (attached via :meth:`Flow.with_checkpointer`), the
    framework calls :meth:`CheckpointStore.save_checkpoint` after each
    successful iteration with the parent-scope payload (``env.state``,
    which is the outer scope's :class:`State` that persists across
    iterations of this block). Note: when ``state=`` projects a child
    scope, only the parent state is checkpointed — progress in the child
    state is lost on resume. To preserve iteration progress, accumulate
    results in the parent state or use ``until=`` with state-driven
    termination.

    Resume: when a ``_ResumeReplay`` is threaded via :attr:`_RunEnv.replay`
    and its ``remaining_path`` has been head-popped down to a single
    entry equal to this iterate's runtime ``node_id`` (i.e., this
    iterate IS the save-point leaf), the counter starts at the saved
    iteration instead of 0 — the folded fix for the note-423 gap
    (``max_iters`` becomes a cumulative bound across resumes, not
    per-run). At most one iterate per run consumes the replay;
    :attr:`Flow._replay_consumed` flips the first time a match fires
    so re-entrant dispatches of the same node (e.g., an inner iterate
    spun up by an outer loop) do not re-apply the fast-forward.
    ``deadline`` is not restored — the wall clock resets each run.

    When resuming, if ``child_state_data`` is present in the replay, it
    replaces the projected child state — restoring mutations that
    occurred before the checkpoint was saved.
    """
    iteration, restored_child = _resume_iteration_for(env, node_id)
    if restored_child is not None:
        child_state = State(data=restored_child, _parent=env.state)
    else:
        child_state = await _project_state(it.state_fn, env.state)
    result: Any = node_args[0] if node_args else None
    started = time.monotonic()
    while True:
        if it.max_iters is not None and iteration >= it.max_iters:
            break
        if it.deadline is not None and time.monotonic() - started >= it.deadline:
            break
        if env.halt is not None and env.halt.is_set():
            break
        result = await _dispatch_iterate_body(it, env, child_state, result, node_id)
        iteration += 1
        _save_iterate_checkpoint(env, iteration, node_id, child_state)
        if await _check_until(it.until, result, child_state, env):
            break
    await _merge_state(it.merge_fn, env.state, child_state)
    return result


def _resume_iteration_for(env: _RunEnv, node_id: str) -> tuple[int, Any]:
    """Return the starting iteration count and restored child state for ``_run_iterate``.

    ``(0, None)`` for a fresh run. On resume, when ``env.replay`` is set
    and ``remaining_path == (node_id,)`` — the head-pop path has shrunk
    to a single entry equal to this iterate's ``node_id`` — this iterate
    IS the save-point leaf: returns the saved iteration and
    ``child_state_data`` (if present) and flips
    :attr:`Flow._replay_consumed` on the top-level runtime so the
    fast-forward fires exactly once. Any longer remaining path means
    this iterate is an ancestor of the leaf (its body descent will
    head-pop and thread the tail); a non-matching head, empty path, or
    already-consumed replay all yield ``(0, None)`` and the iterate
    runs from scratch.
    """
    replay = env.replay
    if replay is None or not replay.remaining_path:
        return 0, None
    if env.runtime._replay_consumed:
        return 0, None
    if replay.remaining_path[0] != node_id or len(replay.remaining_path) != 1:
        return 0, None
    env.runtime._replay_consumed = True
    return replay.iteration, replay.child_state_data


def _pop_replay_for(env: _RunEnv, node_id: str) -> _ResumeReplay | None:
    """Return the replay to thread through a descent under ``node_id``.

    Head-pop at descent site: when ``env.replay.remaining_path[0]``
    equals ``node_id``, the descent is on the saved ancestor chain —
    pop the head and thread the tail to the child Flow. Otherwise the
    descent is off-path (its subtree cannot contain the leaf) or the
    replay has already been consumed at the leaf, and no replay is
    threaded. Called by every descent helper (``_run_subflow``,
    ``_run_branch``, ``_dispatch_iterate_body``, ``_dispatch_map_body``).
    ``full_path`` is preserved verbatim across the pop so downstream
    triage messages can show the whole saved ancestor chain.
    """
    replay = env.replay
    if replay is None or not replay.remaining_path:
        return None
    if env.runtime._replay_consumed:
        return None
    if replay.remaining_path[0] != node_id:
        return None
    return dataclasses.replace(replay, remaining_path=replay.remaining_path[1:])


def _resolve_map_item_replay(
    mp: _Map, env: _RunEnv, node_id: str, item_count: int
) -> dict[int, _ResumeReplay | None]:
    """Determine which map item (if any) gets the replay.

    Pop replay once at the map boundary. Then, for each item, check if
    the popped path's head matches any node_id in that item's body.
    Only the matching item gets the replay; all others get ``None``.
    This prevents scheduling-dependent replay failures when concurrent
    map items race to validate the path.
    """
    from .flow import _compute_node_id, _descend_context

    result: dict[int, _ResumeReplay | None] = {}
    popped = _pop_replay_for(env, node_id)
    if popped is None or not popped.remaining_path:
        return result
    head = popped.remaining_path[0]
    for i in range(item_count):
        item_ctx = _descend_context(node_id, f"map:{i}")
        item_ids = tuple(_compute_node_id(item_ctx, n, j) for j, n in enumerate(mp.body._nodes))
        if head in item_ids:
            result[i] = popped
            break
    return result


def _assert_replay_allows_skip(env: _RunEnv, node_id: str, reason: str) -> None:
    """Fail-fast when a skipped descent was on the replay's saved path.

    Called when a descent site decides not to descend (e.g., branch with
    falsy predicate and no ``else_`` arm). If the replay's
    ``remaining_path[0]`` equals ``node_id``, the checkpoint expected
    descent through this node — skipping it means the composition
    graph's runtime behavior has changed since the checkpoint was
    written. Raising here preserves the head-pop invariant: no chain
    step after a doomed skip runs before the error.
    """
    replay = env.replay
    if replay is None or not replay.remaining_path:
        return
    if env.runtime._replay_consumed:
        return
    if replay.remaining_path[0] != node_id:
        return
    path_repr = " → ".join(replay.full_path) if replay.full_path else "<empty>"
    raise RuntimeError(f"{reason} Saved path: {path_repr}")


async def _dispatch_iterate_body(
    it: _Iterate,
    env: _RunEnv,
    child_state: State[Any],
    prev_result: Any,
    node_id: str,
) -> Any:
    """Run one pass of an ``.iterate`` body under the current env.

    Descends into the body with ``chain_context =
    _descend_context(node_id, "body")`` and ``ancestor_chain`` extended
    by ``node_id`` — the same context every pass, so the body's chain
    steps have iteration-invariant IDs (the runtime pass counter is
    stored alongside the path, not baked into node identity).
    """
    from .flow import _descend_context

    return await it.body._run_as_subflow(
        prev_result,
        state=child_state,
        runtime=env.runtime,
        parent_halt=env.halt,
        parent_budget=env.budget,
        parent_checkpointer=env.checkpointer,
        parent_client_flow_id=env.client_flow_id,
        parent_chain_context=_descend_context(node_id, "body"),
        parent_ancestor_chain=env.ancestor_chain + (node_id,),
        parent_replay=_pop_replay_for(env, node_id),
    )


def _save_iterate_checkpoint(
    env: _RunEnv,
    iteration: int,
    node_id: str,
    current_state: State[Any],
) -> None:
    """Persist a recursive snapshot at an iterate boundary.

    ``state_json`` carries the state stack from root to ``current_state``
    as a nested ``{data, children}`` tree — ``data`` at the top is the
    outermost run-level payload, and each layer of ``children`` is one
    step deeper into scoped composition. Old-shape readers looking at
    ``state_json["data"]`` still see the outermost payload; the tree
    extension is additive.

    ``metadata_json`` carries the schema version, the composition-tree
    path (a flat list of content-addressed node IDs from root to and
    including this iterate — the ancestor chain in ``env`` plus this
    iterate's own ID), and the completed iteration count. Resume walks
    the graph, matching each id at the corresponding chain step to
    relocate the same iterate; a mismatch is a hard error at that
    depth.

    No-op when the runtime has no checkpointer / client_flow_id bound.
    """
    if env.checkpointer is None or env.client_flow_id is None:
        return
    state_json = _serialize_state_tree(current_state)
    path = list(env.ancestor_chain + (node_id,))
    metadata_json = {"path": path, "iteration": iteration}
    env.checkpointer.save_checkpoint(env.client_flow_id, iteration, state_json, metadata_json)


def _serialize_state_tree(current: State[Any]) -> dict[str, Any]:
    """Serialize the state stack from root to ``current`` as a nested tree.

    Returns ``{data, children}`` where the outermost dict is the run-level
    (root) scope and each ``children`` slot descends one scoped layer;
    ``children`` at the innermost scope is ``[]``. Every ``data`` is
    normalized via :func:`_serialize_state_data` (plain-dict passthrough
    or :class:`StateData.to_dict`).

    The current stack is linear (each :class:`State` has one
    ``_parent``) — a list would suffice today, but the nested shape
    leaves room for a future scoped-composition site that fans out into
    sibling children without a schema break.
    """
    scopes: list[State[Any]] = []
    node: State[Any] | None = current
    while node is not None:
        scopes.append(node)
        node = node._parent
    scopes.reverse()
    tree: dict[str, Any] = {"data": _serialize_state_data(scopes[-1].data), "children": []}
    for scope in reversed(scopes[:-1]):
        tree = {"data": _serialize_state_data(scope.data), "children": [tree]}
    return tree


def _serialize_state_data(data: Any) -> Any:
    """Return a JSON-compatible view of ``data`` for the checkpoint.

    Plain dicts pass through as-is (the framework does not deep-copy — the
    store implementation owns durability). Objects satisfying
    :class:`StateData` are converted via ``to_dict()``. ``None`` also
    passes through (a payload that never carried structured data).
    Anything else raises :class:`TypeError` at the save site with a
    pointer to the contract.
    """
    if data is None or isinstance(data, dict):
        return data
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(
        f"cannot checkpoint state.data of type {type(data).__name__} — "
        f"payload must be a plain dict or satisfy StateData (to_dict/from_dict)"
    )


async def _run_map(
    mp: _Map,
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_id: str,
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
    item_replays = _resolve_map_item_replay(mp, env, node_id, len(items))

    async def _gated(index: int, item: Any) -> Any:
        replay = item_replays.get(index)
        if sem is None:
            return await runner(mp, item, env, merge_lock, node_id, index, replay)
        async with sem:
            return await runner(mp, item, env, merge_lock, node_id, index, replay)

    coros = [_gated(i, item) for i, item in enumerate(items)]
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


async def _dispatch_map_body(
    mp: _Map,
    item: Any,
    child_state: State[Any],
    env: _RunEnv,
    node_id: str,
    item_index: int,
    replay: _ResumeReplay | None,
) -> Any:
    """Run a map body's subflow with per-item composition-tree identity.

    Each item descends with a distinct ``chain_context`` keyed by
    ``item_index``, so nested iterates produce unique node IDs per item.
    This prevents checkpoint overwrites and replay race conditions when
    the map body contains checkpointed primitives.

    ``replay`` is pre-resolved at the map boundary by
    :func:`_resolve_map_item_replay` — only the item whose body
    contains the replay's path head receives a non-None value; all
    others receive ``None`` and skip replay validation.
    """
    from .flow import _descend_context

    return await mp.body._run_as_subflow(
        item,
        state=child_state,
        runtime=env.runtime,
        parent_halt=env.halt,
        parent_budget=env.budget,
        parent_checkpointer=env.checkpointer,
        parent_client_flow_id=env.client_flow_id,
        parent_chain_context=_descend_context(node_id, f"map:{item_index}"),
        parent_ancestor_chain=env.ancestor_chain + (node_id,),
        parent_replay=replay,
    )


async def _run_map_item_strict(
    mp: _Map,
    item: Any,
    env: _RunEnv,
    merge_lock: asyncio.Lock,
    node_id: str,
    item_index: int,
    replay: _ResumeReplay | None,
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
        result = await _dispatch_map_body(mp, item, child_state, env, node_id, item_index, replay)
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
    node_id: str,
    item_index: int,
    replay: _ResumeReplay | None,
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
        result = await _dispatch_map_body(mp, item, child_state, env, node_id, item_index, replay)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if mp.on_error is not None:
            await _run_on_error(mp.on_error, exc, item, item_ctx, env)
        return Failure(exception=exc, item=item)
    async with merge_lock:
        await _merge_state(mp.merge_fn, env.state, child_state)
    return result


def _map_item_ctx(env: _RunEnv, child_state: Any) -> Context[Any]:
    """Build the per-item :class:`Context` fed to guard and on_error hooks.

    These hooks run without a :class:`Role`, so ``ctx.saia`` is ``None``.
    """
    return Context(
        role=None,
        state=child_state,
        flow=env.runtime,
        traits=env.runtime._traits,
        halt=env.halt,
        budget=env.budget,
    )


async def _run_guard(guard_fn: Any, item: Any, ctx: Context[Any]) -> bool:
    """Evaluate the guard predicate, awaiting when async, coercing to bool."""
    verdict = guard_fn(item, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_on_error(
    on_error_fn: OnErrorFn,
    exc: BaseException,
    item: Any,
    ctx: Context[Any],
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
