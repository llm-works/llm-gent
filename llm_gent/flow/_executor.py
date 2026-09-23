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
import contextlib
import dataclasses
import inspect
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .checkpoint import maybe_await
from .context import Context
from .nodes import (
    UNSET,
    Failure,
    ItemsFn,
    OnErrorFn,
    OnItemCompleteFn,
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
from .state import State, StateFactory
from .state.cas import (
    Blob,
    Commit,
    CommitMeta,
    CommitOutcome,
    ProducedBy,
    Tree,
    TreeEntry,
    canonical_json,
)


if TYPE_CHECKING:
    from .flow import Flow


def _build_ctx(target: Any, env: _RunEnv, node_id: str | None = None) -> Context[Any]:
    """Build the Context passed to the node's verb (and to its hooks).

    Verb nodes get a role-bound ctx; ``ctx.saia`` resolves lazily on first
    read via the flow's SAIAFactory. Subflow nodes and control-flow nodes
    (branch/iterate/map) get an ambient ctx with ``role=None`` — those nodes
    have no single role, so ``ctx.saia`` returns ``None`` (each inner verb
    builds its own role-bound ctx as it runs).

    ``node_id`` is the composition-graph id of the node the ctx is being
    built for; threaded onto ``ctx._node_id`` so :meth:`Context.checkpoint`
    can address the currently-executing position. ``None`` for hook ctxs
    (until predicates, on_error, on_item_complete) where no verb is
    running under a single node id.
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
            extra=env.extra,
            _env=env,
            _node_id=node_id,
        )
    return Context(
        role=target.role,
        state=env.state,
        flow=env.runtime,
        traits=traits,
        halt=env.halt,
        budget=env.budget,
        extra=env.extra,
        _env=env,
        _node_id=node_id,
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
            target,
            env,
            node.state_fn,
            node.merge_fn,
            node.state_factory,
            node_args,
            node_kwargs,
            node_id,
        )
    if isinstance(target, _Branch):
        return await _run_branch(target, ctx, env, node_args, node_id)
    if isinstance(target, _Iterate):
        return await _run_iterate(target, ctx, env, node_args, node_id)
    if isinstance(target, _Map):
        return await _run_map(target, ctx, env, node_args, node_id)
    passed_args, passed_kwargs = _filter_verb_args(target, node_args, node_kwargs)
    return await target(ctx, *passed_args, **passed_kwargs)


@dataclasses.dataclass(frozen=True)
class _VerbArity:
    """Cached signature shape of a verb callable, minus the ``ctx`` slot.

    ``var_positional`` — verb declares ``*args``. ``var_keyword`` — verb
    declares ``**kwargs``. ``positional_slots`` — count of fixed positional
    parameters after ``ctx`` (POSITIONAL_ONLY + POSITIONAL_OR_KEYWORD).
    ``keyword_names`` — names of parameters reachable by keyword
    (POSITIONAL_OR_KEYWORD + KEYWORD_ONLY). ``introspection_failed`` —
    the callable's signature could not be inspected (C callables, some
    partials); the caller then forwards args verbatim so behavior matches
    pre-filter dispatch.
    """

    var_positional: bool
    var_keyword: bool
    positional_slots: int
    keyword_names: frozenset[str]
    introspection_failed: bool = False


_VERB_ARITY_ATTR = "_llm_gent_verb_arity"
"""Attribute name used to memoize :class:`_VerbArity` on the verb itself.

Attaching to the callable ties the cache lifetime to the target — a
locally-defined verb GC'd at the end of a test cannot leak into a later
test that happens to allocate a different signature at the same id
slot. Falls back to a fresh analysis when ``setattr`` is rejected
(``__slots__``, some C-level callables, class instances forbidding
attribute assignment).
"""


def _verb_arity(target: Any) -> _VerbArity:
    """Return the arity of ``target`` (skipping ``ctx``); memoize on the target itself."""
    cached = getattr(target, _VERB_ARITY_ATTR, None)
    if isinstance(cached, _VerbArity):
        return cached
    computed = _compute_verb_arity(target)
    with contextlib.suppress(AttributeError, TypeError):
        setattr(target, _VERB_ARITY_ATTR, computed)
    return computed


def _compute_verb_arity(target: Any) -> _VerbArity:
    """Inspect ``target``'s signature and derive its :class:`_VerbArity`."""
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return _VerbArity(
            var_positional=True,
            var_keyword=True,
            positional_slots=0,
            keyword_names=frozenset(),
            introspection_failed=True,
        )
    params = list(sig.parameters.values())
    # Skip ctx if it's a fixed positional (not *args).
    fixed_pos = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if params and params[0].kind in fixed_pos:
        params = params[1:]
    return _VerbArity(
        var_positional=any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params),
        var_keyword=any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params),
        positional_slots=sum(1 for p in params if p.kind in fixed_pos),
        keyword_names=frozenset(
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ),
    )


def _filter_verb_args(
    target: Any,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Trim ``node_args``/``node_kwargs`` to what ``target``'s signature accepts.

    Verbs may declare ``(ctx)`` only and still sit at any chain position —
    the previous node's result is dropped rather than raising, so
    pure-Python verbs don't need a placeholder ``_prev`` parameter. Verbs
    declaring ``*args`` / ``**kwargs`` see the full inputs unchanged.
    Introspection failures fall through with all args forwarded so C-level
    callables and exotic partials keep working.
    """
    arity = _verb_arity(target)
    if arity.introspection_failed:
        return node_args, node_kwargs
    passed_args = node_args if arity.var_positional else node_args[: arity.positional_slots]
    passed_kwargs = (
        node_kwargs
        if arity.var_keyword
        else {k: v for k, v in node_kwargs.items() if k in arity.keyword_names}
    )
    return passed_args, passed_kwargs


async def _run_subflow(
    body: Flow,
    env: _RunEnv,
    state_fn: StateProject | None,
    merge_fn: StateMerge | None,
    state_factory: StateFactory[Any] | None,
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

    child_replay = _pop_replay_for(env, node_id)
    raw, child_replay = _consume_scope_data(child_replay, state_fn)
    if raw is UNSET:
        child_state = await _project_state(state_fn, env.state, state_factory)
    else:
        effective_factory = state_factory if state_factory is not None else env.state._factory
        child_state = _restore_scope_state(env.state, raw, effective_factory)
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
        parent_replay=child_replay,
        parent_extra=env.extra,
        parent_policy=env.policy,
        **node_kwargs,
    )
    await _merge_state(merge_fn, env.state, child_state)
    return result


def _consume_scope_data(
    replay: _ResumeReplay | None,
    state_fn: StateProject | None,
) -> tuple[Any, _ResumeReplay | None]:
    """Head-pop the next intermediate-scope payload from the replay when applicable.

    Returns ``(raw_scope_data, updated_replay)``. Returns
    ``(UNSET, replay)`` — signaling "no restored data, project fresh" —
    when any of:

    - ``replay`` is ``None`` (off the replay path).
    - ``state_fn`` is ``None`` (no scope is being created at this
      descent, so nothing to consume).
    - ``replay.intermediate_scope_data`` is empty (all middle scopes
      already consumed, or the checkpointed stack had no middle scopes).

    On a hit, the returned replay carries the tail so the child scope's
    intermediate list stays aligned with its own remaining descents.
    """
    if replay is None or state_fn is None or not replay.intermediate_scope_data:
        return UNSET, replay
    raw = replay.intermediate_scope_data[0]
    updated = dataclasses.replace(
        replay, intermediate_scope_data=replay.intermediate_scope_data[1:]
    )
    return raw, updated


def _restore_scope_state(
    parent: State[Any],
    raw: Any,
    factory: StateFactory[Any] | None,
) -> State[Any]:
    """Wrap a restored raw scope payload as a child :class:`State`.

    Companion to :func:`_project_state` — same shape as the fresh
    projection but uses ``factory.restore(raw)`` (or a passthrough when
    ``factory is None``) instead of running ``state_fn(parent.data)``.
    """
    child_payload = factory.restore(raw) if factory is not None else raw
    return State(data=child_payload, _parent=parent, _factory=factory)


async def _project_state(
    state_fn: StateProject | None,
    parent: State[Any],
    factory: StateFactory[Any] | None = None,
) -> State[Any]:
    """Build the child :class:`State` for a scoped block; pass-through when unset.

    With no projection, the subflow sees the parent's :class:`State` object
    directly — same reference, shared payload, ``is_root`` echoes the parent,
    and ``_factory`` is inherited.

    With a projection, ``state_fn(parent.data)`` produces the child payload,
    which the framework wraps as ``State(data=child_payload, _parent=parent,
    _factory=...)`` so the child's :meth:`State.root` still walks back to
    the outermost scope. The factory attached is ``factory`` if provided,
    else inherited from ``parent._factory``.
    """
    if state_fn is None:
        return parent
    child_payload = state_fn(parent.data)
    if inspect.isawaitable(child_payload):
        child_payload = await child_payload
    effective_factory = factory if factory is not None else parent._factory
    return State(data=child_payload, _parent=parent, _factory=effective_factory)


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
    node_id: str,
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
        extra=env.extra,
        _env=env,
        _node_id=node_id,
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
        parent_extra=env.extra,
        parent_policy=env.policy,
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
    framework builds a content-addressed commit (Blob→Tree→Commit) after each
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
    env, child_state = await _iterate_child_scope(it, env, node_id, restored_child)
    result: Any = node_args[0] if node_args else None
    started = time.monotonic()
    while True:
        if it.max_iters is not None and iteration >= it.max_iters:
            break
        if it.deadline is not None and time.monotonic() - started >= it.deadline:
            break
        if env.halt is not None and env.halt.is_set():
            await _save_halt_checkpoint(env, iteration, node_id, child_state)
            break
        result = await _dispatch_iterate_body(it, env, child_state, result, node_id)
        iteration += 1
        if env.policy.on_iterate:
            await _save_iterate_checkpoint(env, iteration, node_id, child_state)
        if await _check_until(it.until, result, child_state, env, node_id):
            break
    await _merge_state(it.merge_fn, env.state, child_state)
    return result


async def _iterate_child_scope(
    it: _Iterate,
    env: _RunEnv,
    node_id: str,
    restored_child: Any,
) -> tuple[_RunEnv, State[Any]]:
    """Build the iterate body's child :class:`State` for this run.

    Three paths, tried in order:

    1. ``restored_child`` is not ``None`` — this iterate is the leaf, use
       the checkpointed leaf-scope data (via ``_ResumeReplay.child_state_data``).
    2. This iterate is a middle scope on the replay path with a scoping
       ``state_fn`` — consume the next intermediate scope payload from
       the replay and return an ``env`` with the popped replay so the
       body descent sees the aligned tail.
    3. Fresh projection via ``state_fn`` (or passthrough when
       ``state_fn`` is ``None``).
    """
    effective_factory = it.state_factory if it.state_factory is not None else env.state._factory
    if restored_child is not None:
        restored_data = (
            effective_factory.restore(restored_child)
            if effective_factory is not None
            else restored_child
        )
        return env, State(data=restored_data, _parent=env.state, _factory=effective_factory)
    on_path = (
        env.replay is not None
        and env.replay.remaining_path
        and env.replay.remaining_path[0] == node_id
        and not env.runtime._replay_consumed
    )
    if on_path:
        raw, updated_replay = _consume_scope_data(env.replay, it.state_fn)
        if raw is not UNSET:
            env = dataclasses.replace(env, replay=updated_replay)
            return env, _restore_scope_state(env.state, raw, effective_factory)
    return env, await _project_state(it.state_fn, env.state, it.state_factory)


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
        parent_extra=env.extra,
        parent_policy=env.policy,
    )


async def _save_iterate_checkpoint(
    env: _RunEnv,
    iteration: int,
    node_id: str,
    current_state: State[Any],
) -> None:
    """Persist an ``outcome="ok"`` commit at an iterate boundary."""
    await _save_scope_commit(env, iteration, node_id, current_state, "ok")


async def _save_map_item_checkpoint(
    env: _RunEnv,
    item_index: int,
    node_id: str,
    current_state: State[Any],
) -> None:
    """Persist an ``outcome="ok"`` commit at a map item boundary.

    Uses ``item_index`` as the iteration slot so each item lands in
    a distinct ref under the map's node_path. Items complete in
    parallel; save order is not guaranteed to match item order and
    saves may interleave with concurrent items' merges.
    """
    await _save_scope_commit(env, item_index, node_id, current_state, "ok")


async def _save_halt_checkpoint(
    env: _RunEnv,
    iteration: int,
    node_id: str,
    current_state: State[Any],
) -> None:
    """Persist an ``outcome="halted"`` commit at a halt observation point.

    Called from the executor's halt-observation sites — the between-
    iterations check in :func:`_run_iterate` and the between-chain-
    steps check in :meth:`Flow._walk_chain` — so a ``run(resume=True)``
    after a halted process restart resolves to this commit and re-
    enters at the halted position.

    At most one halt-save fires per run: :attr:`Flow._halt_saved` on
    the top-level runtime latches after the first save so a halt fires
    through an inner iterate's boundary check + the outer chain-walk's
    between-steps check does not double-save (the inner save is the
    finer-grained resume anchor).

    ``iteration`` is the iterate's current counter at halt time (``0``
    for a chain-only halt with no enclosing iterate). ``node_id`` is
    the node the halt was observed against — the iterate's node id for
    the iterate case; the not-yet-run chain step's id for the chain
    case. State is captured verbatim; verbs are expected to be
    idempotent-in-effects to survive re-run on resume, the same
    contract that already governs iterate re-run-iteration-N.
    """
    if env.runtime._halt_saved:
        return
    env.runtime._halt_saved = True
    await _save_scope_commit(env, iteration, node_id, current_state, "halted")


async def _save_scope_commit(
    env: _RunEnv,
    iteration: int,
    node_id: str,
    current_state: State[Any],
    outcome: CommitOutcome,
) -> None:
    """Persist a content-addressed commit at ``node_id`` under ``env``.

    Walks the scope stack from root to ``current_state``. For each scope:
    serialize its ``data`` via the state-data contract (dict passthrough
    or ``StateData.to_dict``) to canonical JSON bytes and
    :meth:`put_object` a Blob keyed by content hash. Bundle every scope's
    blob hash into a Tree (one :class:`TreeEntry` per scope, ordered by
    depth via a two-digit ``scope_id``). Wrap the Tree in a Commit whose
    :class:`CommitMeta` pins ``(client_flow_id, node_path, iteration)``
    and the provenance triple (``produced_by``, ``trace_ref``,
    ``outcome``). Finally put_ref points this boundary at the commit
    hash.

    ``node_path`` is the ``"/"``-joined ancestor chain (from run root to
    this node, inclusive). blake2b hex has no ``"/"``, so split
    round-trips on resume.

    ``outcome`` records why the commit fired — ``"ok"`` for a successful
    iterate boundary, ``"halted"`` when the halt-observation site
    triggered the save.

    No-op when the runtime has no checkpointer / client_flow_id bound.
    """
    if env.checkpointer is None or env.client_flow_id is None:
        return
    scopes = _collect_scope_stack(current_state)
    entries = await _put_scope_blobs(env, scopes)
    tree = Tree.from_entries(entries)
    await maybe_await(
        env.checkpointer.put_object(env.client_flow_id, "tree", tree.content_hash, tree.to_bytes())
    )
    node_path = "/".join(env.ancestor_chain + (node_id,))
    meta = _build_commit_meta(env, node_path, iteration, node_id, outcome)
    commit = Commit.build(root_tree_hash=tree.content_hash, parent_hashes=(), meta=meta)
    await maybe_await(
        env.checkpointer.put_object(
            env.client_flow_id, "commit", commit.content_hash, commit.to_bytes()
        )
    )
    await maybe_await(
        env.checkpointer.put_ref(env.client_flow_id, node_path, iteration, commit.content_hash)
    )


def _collect_scope_stack(current: State[Any]) -> list[State[Any]]:
    """Return the ``State`` chain from run-root down to ``current``."""
    scopes: list[State[Any]] = []
    node: State[Any] | None = current
    while node is not None:
        scopes.append(node)
        node = node._parent
    scopes.reverse()
    return scopes


async def _put_scope_blobs(env: _RunEnv, scopes: list[State[Any]]) -> list[TreeEntry]:
    """Serialize each scope's data to a blob, put_object it, return tree entries.

    Uses a zero-padded two-digit index as :attr:`TreeEntry.scope_id` so
    canonical sort ordering matches root→leaf depth ordering.
    """
    entries: list[TreeEntry] = []
    for depth, scope in enumerate(scopes):
        blob_bytes = canonical_json(_serialize_state_data(scope.data))
        blob = Blob.from_bytes(blob_bytes)
        assert env.checkpointer is not None
        assert env.client_flow_id is not None
        await maybe_await(
            env.checkpointer.put_object(env.client_flow_id, "blob", blob.content_hash, blob.payload)
        )
        entries.append(
            TreeEntry(scope_id=f"{depth:02d}", kind="blob", child_hash=blob.content_hash)
        )
    return entries


def _build_commit_meta(
    env: _RunEnv,
    node_path: str,
    iteration: int,
    node_id: str,
    outcome: CommitOutcome,
) -> CommitMeta:
    """Assemble :class:`CommitMeta` for one scope-commit save.

    ``produced_by`` records the node's ``node_id`` — verb-level
    attribution (``verb_name`` / ``role`` / ``result_hash``) lands with
    the SAIA-verb-wrapper wiring. ``trace_ref`` is empty until the same
    wiring stamps SAIA turn ids.

    ``outcome`` is set by the caller: ``"ok"`` at an iterate boundary,
    ``"halted"`` at a halt-observation save.

    ``flow_root_id`` is the run's ``client_flow_id`` — a stable
    per-run identifier — until the framework computes a proper
    composition-tree root hash.
    """
    from llm_gent import __version__

    return CommitMeta(
        client_flow_id=env.client_flow_id or "",
        node_path=node_path,
        iteration=iteration,
        produced_by=ProducedBy(node_id=node_id, verb_name=None, role=None, result_hash=None),
        trace_ref=(),
        outcome=outcome,
        flow_root_id=env.client_flow_id or "",
        timestamp_iso=datetime.now(UTC).isoformat(),
        framework_version=__version__,
    )


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
        parent_extra=env.extra,
        parent_policy=env.policy,
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
    escapes. ``on_item_complete`` fires at every terminal state (success,
    halt-Skipped, guard-Skipped, or a synthesized :class:`Failure` before
    re-raise) — cancellation is unconditional and never fires the hook.
    """
    if env.halt is not None and env.halt.is_set():
        skipped = Skipped(item=item)
        await _fire_on_item_complete(
            mp.on_item_complete, item, skipped, _map_item_ctx(env, env.state), env
        )
        return skipped
    item_ctx = _map_item_ctx(env, env.state)
    try:
        child_state = await _project_state(mp.state_fn, env.state, mp.state_factory)
        item_ctx = _map_item_ctx(env, child_state)
        if mp.guard is not None and not await _run_guard(mp.guard, item, item_ctx):
            skipped = Skipped(item=item)
            await _fire_on_item_complete(mp.on_item_complete, item, skipped, item_ctx, env)
            return skipped
        result = await _dispatch_map_body(mp, item, child_state, env, node_id, item_index, replay)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if mp.on_error is not None:
            await _run_on_error(mp.on_error, exc, item, item_ctx, env)
        await _fire_on_item_complete(
            mp.on_item_complete, item, Failure(exception=exc, item=item), item_ctx, env
        )
        raise
    return await _map_item_success(
        mp, item, result, child_state, item_ctx, env, merge_lock, node_id, item_index, strict=True
    )


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
    ``on_item_complete`` fires at every terminal state (success,
    :class:`Failure`, halt-Skipped, guard-Skipped); cancellation is
    unconditional and never fires the hook.
    """
    if env.halt is not None and env.halt.is_set():
        skipped = Skipped(item=item)
        await _fire_on_item_complete(
            mp.on_item_complete, item, skipped, _map_item_ctx(env, env.state), env
        )
        return skipped
    item_ctx = _map_item_ctx(env, env.state)
    try:
        child_state = await _project_state(mp.state_fn, env.state, mp.state_factory)
        item_ctx = _map_item_ctx(env, child_state)
        if mp.guard is not None and not await _run_guard(mp.guard, item, item_ctx):
            skipped = Skipped(item=item)
            await _fire_on_item_complete(mp.on_item_complete, item, skipped, item_ctx, env)
            return skipped
        result = await _dispatch_map_body(mp, item, child_state, env, node_id, item_index, replay)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if mp.on_error is not None:
            await _run_on_error(mp.on_error, exc, item, item_ctx, env)
        failure = Failure(exception=exc, item=item)
        await _fire_on_item_complete(mp.on_item_complete, item, failure, item_ctx, env)
        return failure
    return await _map_item_success(
        mp, item, result, child_state, item_ctx, env, merge_lock, node_id, item_index, strict=False
    )


def _map_item_ctx(env: _RunEnv, child_state: Any, node_id: str | None = None) -> Context[Any]:
    """Build the per-item :class:`Context` fed to guard, on_error, and
    on_item_complete hooks.

    These hooks run without a :class:`Role`, so ``ctx.saia`` is ``None``.
    ``node_id`` is the :class:`_Map` node's own id — passing it lets a
    ``ctx.checkpoint()`` from a map hook address the map's position.
    """
    return Context(
        role=None,
        state=child_state,
        flow=env.runtime,
        traits=env.runtime._traits,
        halt=env.halt,
        budget=env.budget,
        extra=env.extra,
        _env=env,
        _node_id=node_id,
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


async def _fire_on_item_complete(
    hook: OnItemCompleteFn | None,
    item: Any,
    outcome: Any,
    ctx: Context[Any],
    env: _RunEnv,
) -> None:
    """Invoke on_item_complete (if attached) and swallow any exception it raises.

    An observer that raises must never mask the outcome that lands in
    the map's result list. Matches :func:`_run_on_error` semantics.
    """
    if hook is None:
        return
    try:
        result = hook(item, outcome, ctx)
        if inspect.isawaitable(result):
            await result
    except Exception as hook_exc:
        env.lg.warning(
            "map on_item_complete hook raised — outcome preserved",
            extra={"exception": hook_exc},
        )


async def _merge_and_notify(
    mp: _Map,
    item: Any,
    result: Any,
    child_state: Any,
    item_ctx: Context[Any],
    env: _RunEnv,
    merge_lock: asyncio.Lock,
) -> Failure | None:
    """Merge child state into parent, then fire on_item_complete; return Failure if merge raised."""
    try:
        async with merge_lock:
            await _merge_state(mp.merge_fn, env.state, child_state)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if mp.on_error is not None:
            await _run_on_error(mp.on_error, exc, item, item_ctx, env)
        failure = Failure(exception=exc, item=item)
        await _fire_on_item_complete(mp.on_item_complete, item, failure, item_ctx, env)
        return failure
    await _fire_on_item_complete(mp.on_item_complete, item, result, item_ctx, env)
    return None


async def _map_item_success(
    mp: _Map,
    item: Any,
    result: Any,
    child_state: State[Any],
    item_ctx: Context[Any],
    env: _RunEnv,
    merge_lock: asyncio.Lock,
    node_id: str,
    item_index: int,
    *,
    strict: bool,
) -> Any:
    """Merge, optionally save (per policy), and return the map item's result.

    Both ``strict`` and non-strict paths converge here after the body
    returned successfully. Merge-time failures are handled per-mode:
    strict re-raises the underlying exception; non-strict returns the
    Failure sentinel. Successful merges save a scope commit when
    ``env.policy.on_map_item`` is set (see :attr:`CheckpointPolicy`).
    """
    failure = await _merge_and_notify(mp, item, result, child_state, item_ctx, env, merge_lock)
    if failure is not None:
        if strict:
            raise failure.exception
        return failure
    if env.policy.on_map_item:
        await _save_map_item_checkpoint(env, item_index, node_id, env.state)
    return result


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
