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
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .checkpoint import checkpoint_tag
from .context import Context
from .nodes import (
    UNSET,
    StateMerge,
    StateProject,
    UntilFn,
    _Branch,
    _Iterate,
    _Map,
    _Node,
    _RunEnv,
)
from .state import State, StateFactory
from .state.cas import (
    Commit,
    CommitOutcome,
    TraceRef,
)
from .state.snapshot import ARM, Cursor, Live, ScopePath


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
            cost=env.cost,
            extra=env.extra,
            resources=env.resources,
            _env=env,
            _node_id=node_id,
        )
    return Context(
        role=target.role,
        state=env.state,
        flow=env.runtime,
        traits=traits,
        halt=env.halt,
        cost=env.cost,
        extra=env.extra,
        resources=env.resources,
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
        from ._iterate import IterateRunner

        return await IterateRunner(target, env, node_id).run(node_args)
    if isinstance(target, _Map):
        from ._map import MapRunner

        return await MapRunner(target, env, node_id).run(node_args)
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
    from ._node_id import _descend_context

    path = env.owner_path(node_id)
    child_state = await _enter_scope(env, path, state_fn, state_factory)
    with _live_scope(env, path, state_fn, child_state):
        result = await body._run_as_subflow(
            *node_args,
            state=child_state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_cost=env.cost,
            parent_resources=env.resources,
            parent_checkpoint_ctx=env.checkpoint_ctx,
            parent_checkpointer=env.checkpointer,
            parent_chain_context=_descend_context(node_id, "call"),
            parent_ancestor_chain=env.ancestor_chain + (node_id,),
            parent_extra=env.extra,
            parent_policy=env.policy,
            parent_path=path,
            parent_shortcuts=env.shortcuts,
            **node_kwargs,
        )
        await _merge_state(merge_fn, env.state, child_state)
    return result


@contextlib.contextmanager
def _live_scope(
    env: _RunEnv, path: ScopePath, state_fn: StateProject | None, scope: State[Any]
) -> Iterator[None]:
    """Keep ``scope`` registered at ``path`` for the block when a ``state=`` projection made it.

    Without a projection the block shares its parent's scope, which is
    already registered (or is the root). The scope is dropped once the
    block completes — after its merge, so a snapshot taken meanwhile
    never misses its data. A block that stops early keeps it (see
    :func:`_running`).
    """
    if state_fn is None:
        yield
        return
    env.scopes.open(path, scope)
    yield
    env.scopes.close(path)


@contextlib.contextmanager
def _running(env: _RunEnv, path: ScopePath, runner: Cursor) -> Iterator[None]:
    """Keep ``runner``'s cursor in the run's snapshots at ``path`` while the block runs.

    Dropped once the block completes. A block that stops early — the halt
    interrupted it, or it raised — keeps it registered where it stopped:
    the run's halt checkpoint holds it there, and so does any checkpoint a
    sibling (another map item) takes meanwhile, so resume continues at the
    step that stopped. What carries on past the block drops it — the
    enclosing chain moving past the step (e.g. after a rescue policy), a
    map item finishing, the next run's start.
    """
    env.scopes.open_cursor(path, runner)
    yield
    env.scopes.close_cursor(path, runner)


async def _enter_scope(
    env: _RunEnv,
    path: ScopePath,
    state_fn: StateProject | None,
    state_factory: StateFactory[Any] | None,
) -> State[Any]:
    """The scope a block at ``path`` runs under: restored from a checkout, else projected.

    A block with a ``state=`` projection gets back the scope the checked-out
    snapshot saved at ``path``, the first time the run reaches that path;
    otherwise the projection runs. Without a projection the block shares
    its parent's scope.
    """
    if state_fn is not None:
        found, raw = env.scopes.take_saved(path)
        if found:
            factory = state_factory if state_factory is not None else env.state._factory
            return _restore_scope_state(env.state, raw, factory)
    return await _project_state(state_fn, env.state, state_factory)


def _restore_scope_state(
    parent: State[Any],
    raw: Any,
    factory: StateFactory[Any] | None,
) -> State[Any]:
    """Wrap a restored raw scope payload as a child :class:`State`.

    Companion to :func:`_project_state` — same shape as the fresh
    projection but uses ``factory.restore(raw)`` (or a passthrough when
    ``factory is None``) instead of running ``state_fn(parent.data)``. A
    restaged scope (:class:`~llm_gent.flow.state.snapshot.Live`) is the
    child :class:`State` itself.
    """
    if isinstance(raw, Live):
        return raw.value  # type: ignore[no-any-return]
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
        cost=env.cost,
        extra=env.extra,
        resources=env.resources,
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
    prev_result = node_args[0] if node_args else None
    verdict = await _branch_verdict(br, ctx, env, node_id, prev_result)
    chosen = br.then_flow if verdict else br.else_flow
    if chosen is None:
        return prev_result
    arm = _BranchArm("then" if verdict else "else")
    return await _run_arm(chosen, arm, env, node_id, prev_result)


async def _branch_verdict(
    br: _Branch, ctx: Context[Any], env: _RunEnv, node_id: str, prev_result: Any
) -> bool:
    """True for the ``then`` arm: the arm a checkout saved at this branch, else ``when``'s verdict.

    A saved arm is taken as is: ``when`` is not evaluated again, since the
    state it reads may have changed since the branch chose.
    """
    found, arm = env.scopes.take_cursor(env.owner_path(node_id), ARM)
    if found:
        return bool(arm == "then")
    verdict = br.when(prev_result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_arm(
    chosen: Flow, arm: _BranchArm, env: _RunEnv, node_id: str, prev_result: Any
) -> Any:
    """Run the arm a branch took, with the branch's cursor in every snapshot meanwhile."""
    from ._node_id import _descend_context

    path = env.owner_path(node_id)
    with _running(env, path, arm):
        return await chosen._run_as_subflow(
            prev_result,
            state=env.state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_cost=env.cost,
            parent_resources=env.resources,
            parent_checkpoint_ctx=env.checkpoint_ctx,
            parent_checkpointer=env.checkpointer,
            parent_chain_context=_descend_context(node_id, arm.arm),
            parent_ancestor_chain=env.ancestor_chain + (node_id,),
            parent_extra=env.extra,
            parent_policy=env.policy,
            parent_path=path,
            parent_shortcuts=env.shortcuts,
        )


class _BranchArm:
    """Cursor of a running ``.branch``: the arm it took, so resume never re-evaluates ``when``."""

    def __init__(self, arm: str) -> None:
        self.arm = arm

    def cursor(self) -> dict[str, Any]:
        """The arm taken, ``"then"`` or ``"else"``."""
        return {ARM: self.arm}


async def _save_scope_commit(
    env: _RunEnv,
    iteration: int,
    node_id: str,
    current_state: State[Any],
    outcome: CommitOutcome,
    trace_ref: tuple[TraceRef, ...] = (),
) -> Commit | None:
    """Save the whole run as a commit in its repo and return it; ``None`` when nothing saves here.

    The save point of :class:`IterateRunner`, :class:`MapItemRunner` and
    :meth:`Context.checkpoint`. It writes only with a repo
    (``env.checkpoint_ctx``) and a checkpointer on the saving flow or
    above it (``env.checkpointer``); the save belongs to that innermost
    checkpointer, and a named one's tag moves to the commit. Delegates the
    persistence to :meth:`CheckpointContext.save_scope_commit`.
    """
    ctx, checkpointer = env.checkpoint_ctx, env.checkpointer
    if ctx is None or checkpointer is None:
        return None
    commit = await ctx.save_scope_commit(
        env.ancestor_chain, iteration, node_id, env.scopes, current_state, outcome, trace_ref
    )
    if checkpointer.name is not None:
        await ctx.put_tag(checkpoint_tag(checkpointer.name), commit.content_hash)
    return commit


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
