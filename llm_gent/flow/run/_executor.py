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
from typing import Any

from ..checkpoint import checkpoint_tag
from ..context import Context
from ..flow import Flow
from ..hooks import Where
from ..nodes import (
    UNSET,
    StateMerge,
    StateProject,
    UntilFn,
    _Branch,
    _Iterate,
    _Map,
    _Node,
)
from ..resource._runtime import run_resources
from ..state import State, StateFactory
from ..state.cas import (
    CommitOutcome,
    FlowCommit,
    TraceRef,
)
from ..state.snapshot import ARM, Cursor, ScopePath, path_str
from ._hook_points import StepHooks, checkpoint_written
from .scope import Position, _Frame


def _build_ctx(target: Any, frame: _Frame, node_id: str | None = None) -> Context[Any]:
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
    traits = frame.runtime._traits
    if isinstance(target, Flow | _Branch | _Iterate | _Map):
        return Context(
            role=None,
            state=frame.scope.state,
            flow=frame.runtime,
            traits=traits,
            halt=frame.halt,
            extra=frame.extra,
            resources=frame.scope.resources,
            _frame=frame,
            _node_id=node_id,
        )
    return Context(
        role=target.role,
        state=frame.scope.state,
        flow=frame.runtime,
        traits=traits,
        halt=frame.halt,
        extra=frame.extra,
        resources=frame.scope.resources,
        _frame=frame,
        _node_id=node_id,
    )


async def _execute_node(
    node: _Node,
    ctx: Context[Any],
    frame: _Frame,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
    step: StepHooks | None = None,
) -> Any:
    """Invoke a node's target with cancellation-safe rescue + optional after hook.

    ``asyncio.CancelledError`` propagates unchanged past any rescue policy.
    Other exceptions surface unless a rescue is attached; the rescue's
    return value (awaited if awaitable) becomes the node's result.

    ``step`` calls the ``before_step`` hooks first: a step a hook denies
    raises :class:`~llm_gent.flow.hooks.HookDenied`, which a rescue policy
    handles like any other exception; a pause stops the step as the halt
    does (:class:`Interrupted` passes every rescue).
    """
    target_name = _target_label(node.target)
    frame.lg.debug("executing node", extra={"target": target_name})
    try:
        if step is not None:
            await step.check(ctx, node_args, node_kwargs)
        result = await _invoke_target(node, ctx, frame, node_args, node_kwargs, node_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if node.rescue is None:
            raise
        frame.lg.warning("rescue policy invoked", extra={"target": target_name, "exception": exc})
        pending_input = node_args[0] if node_args else UNSET
        fallback = node.rescue(exc, pending_input, ctx)
        if inspect.isawaitable(fallback):
            fallback = await fallback
        result = fallback
    if node.after is not None:
        frame.lg.debug("running after hook", extra={"target": target_name})
        hook_result = node.after(result, ctx)
        if inspect.isawaitable(hook_result):
            await hook_result
    return result


async def _invoke_target(
    node: _Node,
    ctx: Context[Any],
    frame: _Frame,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Dispatch a node's target: verb, subflow, or control-flow primitive."""
    target = node.target
    if isinstance(target, Flow):
        return await _run_subflow(
            target,
            frame,
            node.state_fn,
            node.merge_fn,
            node.state_factory,
            node_args,
            node_kwargs,
            node_id,
        )
    if isinstance(target, _Branch):
        return await _run_branch(target, ctx, frame, node_args, node_id)
    if isinstance(target, _Iterate):
        from ._iterate import IterateRunner

        return await IterateRunner(target, frame, node_id).run(node_args)
    if isinstance(target, _Map):
        from ._map import MapRunner

        return await MapRunner(target, frame, node_id).run(node_args)
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
    frame: _Frame,
    state_fn: StateProject | None,
    merge_fn: StateMerge | None,
    state_factory: StateFactory[Any] | None,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Run a subflow node, honoring optional scoped-state projection/merge.

    The child Flow is entered through the node's ``"call"`` boundary
    (:meth:`_Frame.descend`), with the projected state when there is one.
    """
    path = frame.owner_path(node_id)
    child_state = await _enter_scope(frame, path, state_fn, state_factory)
    with _live_scope(frame, path, state_fn, child_state):
        child = frame.descend(node_id, "call", path, child_state)
        result = await run_as_subflow(body, node_args, node_kwargs, child)
        await _merge_state(merge_fn, frame.scope.state, child_state)
    return result


async def run_as_subflow(
    flow: Flow, args: tuple[Any, ...], kwargs: dict[str, Any], frame: _Frame
) -> Any:
    """Walk ``flow`` from ``frame``: the place the walk enters it at.

    ``args`` / ``kwargs`` are the first node's inputs. They are passed
    as two values, not spread, so an input named like one of the
    framework's own parameters reaches the node instead of colliding
    with it.

    ``frame`` is what the node entering the flow hands it
    (:meth:`_Frame.descend`; the root scope at the top level): the
    run, the scope with the node's state, the position. The flow's
    declarations apply on top (:meth:`~llm_gent.flow.run.Scope.child`);
    its resources register with the run while it walks
    (:func:`~llm_gent.flow.resource._runtime.run_resources`).
    """
    if not flow._nodes:
        raise RuntimeError(f"Flow {flow._name!r} has no nodes to run")
    decl = flow.decl
    with run_resources(
        frame.lg,
        decl,
        flow._name or "<anonymous>",
        frame.position.path,
        frame.scopes,
        frame.scope.resources,
    ) as resources:
        scope = frame.scope.child(
            decl,
            resources=resources,
            where=_entry_where(flow, frame.position),
            signals=frame.run.signals,
            lg=frame.lg,
        )
        return await _walk(flow, dataclasses.replace(frame, scope=scope), args, kwargs)


def _entry_where(flow: Flow, position: Position) -> Where:
    """Where ``flow`` starts in the run: the step that entered it and its snapshot path."""
    path = position.ancestor_chain
    return Where(
        flow=flow,
        node_id=path[-1] if path else None,
        node_path=path,
        scope_path=path_str(position.path),
        step_name=flow._name or None,
    )


async def _walk(flow: Flow, frame: _Frame, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Walk ``flow``'s chain under ``frame``."""
    from ._chain import Chain

    label = flow._name or "<anonymous>"
    is_subflow = frame.runtime is not flow
    frame.lg.debug(
        "starting flow run",
        extra={"flow": label, "nodes": len(flow._nodes), "subflow": is_subflow},
    )
    result = await Chain(flow, frame).walk(args, kwargs)
    frame.lg.debug("completed flow run", extra={"flow": label, "subflow": is_subflow})
    return result


@contextlib.contextmanager
def _live_scope(
    frame: _Frame, path: ScopePath, state_fn: StateProject | None, scope: State[Any]
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
    frame.scopes.open(path, scope)
    yield
    frame.scopes.close(path)


@contextlib.contextmanager
def _running(frame: _Frame, path: ScopePath, runner: Cursor) -> Iterator[None]:
    """Keep ``runner``'s cursor in the run's snapshots at ``path`` while the block runs.

    Dropped once the block completes. A block that stops early — the halt
    interrupted it, or it raised — keeps it registered where it stopped:
    the run's halt checkpoint holds it there, and so does any checkpoint a
    sibling (another map item) takes meanwhile, so resume continues at the
    step that stopped. What carries on past the block drops it — the
    enclosing chain moving past the step (e.g. after a rescue policy), a
    map item finishing, the next run's start.
    """
    frame.scopes.open_cursor(path, runner)
    yield
    frame.scopes.close_cursor(path, runner)


async def _enter_scope(
    frame: _Frame,
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
        found, raw = frame.scopes.take_saved(path)
        if found:
            factory = state_factory if state_factory is not None else frame.scope.state._factory
            return _restore_scope_state(frame.scope.state, raw, factory)
    return await _project_state(state_fn, frame.scope.state, state_factory)


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
    frame: _Frame,
    node_id: str,
) -> bool:
    """Evaluate the iterate node's until predicate with the last body result and scoped state."""
    if until_fn is None:
        return False
    ctx = Context(
        role=None,
        state=iterate_state,
        flow=frame.runtime,
        traits=frame.runtime._traits,
        halt=frame.halt,
        extra=frame.extra,
        resources=frame.scope.resources,
        _frame=frame,
        _node_id=node_id,
    )
    verdict = until_fn(result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_branch(
    br: _Branch,
    ctx: Context[Any],
    frame: _Frame,
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
    verdict = await _branch_verdict(br, ctx, frame, node_id, prev_result)
    chosen = br.then_flow if verdict else br.else_flow
    if chosen is None:
        return prev_result
    arm = _BranchArm("then" if verdict else "else")
    return await _run_arm(chosen, arm, frame, node_id, prev_result)


async def _branch_verdict(
    br: _Branch, ctx: Context[Any], frame: _Frame, node_id: str, prev_result: Any
) -> bool:
    """True for the ``then`` arm: the arm a checkout saved at this branch, else ``when``'s verdict.

    A saved arm is taken as is: ``when`` is not evaluated again, since the
    state it reads may have changed since the branch chose.
    """
    found, arm = frame.scopes.take_cursor(frame.owner_path(node_id), ARM)
    if found:
        return bool(arm == "then")
    verdict = br.when(prev_result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_arm(
    chosen: Flow, arm: _BranchArm, frame: _Frame, node_id: str, prev_result: Any
) -> Any:
    """Run the arm a branch took, with the branch's cursor in every snapshot meanwhile."""
    path = frame.owner_path(node_id)
    with _running(frame, path, arm):
        child = frame.descend(node_id, arm.arm, path, frame.scope.state)
        return await run_as_subflow(chosen, (prev_result,), {}, child)


class _BranchArm:
    """Cursor of a running ``.branch``: the arm it took, so resume never re-evaluates ``when``."""

    def __init__(self, arm: str) -> None:
        self.arm = arm

    def cursor(self) -> dict[str, Any]:
        """The arm taken, ``"then"`` or ``"else"``."""
        return {ARM: self.arm}


async def _save_scope_commit(
    frame: _Frame,
    iteration: int,
    node_id: str,
    current_state: State[Any],
    outcome: CommitOutcome,
    trace_ref: tuple[TraceRef, ...] = (),
) -> FlowCommit | None:
    """Save the whole run as a commit in its repo and return it; ``None`` when nothing saves here.

    The save point of :class:`IterateRunner`, :class:`MapItemRunner` and
    :meth:`Context.checkpoint`. It writes only with a repo
    (``frame.checkpoint_ctx``) and a checkpointer on the saving flow or
    above it (``frame.scope.checkpointer``); the save belongs to that innermost
    checkpointer, and a named one's tag moves to the commit. Delegates the
    persistence to :meth:`CheckpointContext.save_scope_commit`.
    """
    ctx, checkpointer = frame.checkpoint_ctx, frame.scope.checkpointer
    if ctx is None or checkpointer is None:
        return None
    commit = await ctx.save_scope_commit(
        frame.position.ancestor_chain,
        iteration,
        node_id,
        frame.scopes,
        current_state,
        outcome,
        trace_ref,
    )
    if checkpointer.name is not None:
        await ctx.put_tag(checkpoint_tag(checkpointer.name), commit.id)
    await checkpoint_written(frame.scope.hooks, frame, frame.run.info, commit, node_id)
    return commit


def _target_label(target: Any) -> str:
    """Return a human-readable label for a node target."""
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
