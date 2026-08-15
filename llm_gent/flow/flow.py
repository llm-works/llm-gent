"""Flow — verb registry, role-routed dispatch, and fluent composition graph.

A :class:`Flow` plays two roles that share one object:

1. **Runtime / registry.** Holds a :class:`SAIAFactory` (for turning roles
   into saia instances, cached per-role), a shared user-owned ``state``
   object, and an optional verb-by-name registry used by :meth:`dispatch`
   and by :class:`Panel`.

2. **Composition graph.** A sequence of nodes built up via the fluent
   methods :meth:`call` / :meth:`then` / :meth:`rescue` / :meth:`after` /
   :meth:`branch` / :meth:`loop` / :meth:`map` and executed by :meth:`run`.
   A node's target is either a verb (any callable carrying a ``.role``),
   another :class:`Flow`, or a control-flow primitive (branch, loop, map)
   whose bodies are themselves subflows. Subflows are recursively run
   against the same runtime, so saia caching is shared across the whole
   tree.

Both roles are optional. A top-level flow that only serves as a verb
registry never needs to call the fluent methods; a subflow that only exists
to structure composition never needs a factory of its own — it borrows from
the runtime it is executed under.

Two state channels are exposed on every :class:`Context`:

- ``ctx.state`` is scoped to the enclosing subflow and shared with the
  parent by reference. The ``state=`` / ``merge=`` kwargs on
  :meth:`Flow.call`, :meth:`Flow.loop`, and :meth:`Flow.map` project an
  isolated child state for the block they contain and (optionally) merge it
  back when the block completes successfully.
- ``ctx.global_state`` is run-wide, provided at the outermost :meth:`run`
  invocation (default: fresh empty ``dict``) and inherited by every subflow.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger

from .context import Context
from .factory import SAIAFactory
from .role import Role


RescuePolicy = Callable[[BaseException, Context], Any]
"""Failure hook: ``(exception, ctx) -> fallback``. May be async."""

AfterHook = Callable[[Any, Context], Any]
"""Success hook: ``(result, ctx) -> None`` (return value ignored). May be async."""

ProjectFn = Callable[[Any], Any]
"""Data-flow projection: transforms the previous node's result into the next input."""

WhenFn = Callable[[Any, Context], Any]
"""Branch predicate: ``(prev_result, ctx) -> bool``. May be async."""

UntilFn = Callable[[Context], Any]
"""Loop stop predicate: ``(ctx) -> bool``. May be async. State lives on ``ctx.state``."""

ItemsFn = Callable[[Any, Context], Any]
"""Map item source: ``(prev_result, ctx) -> iterable``. May be async. Consumed eagerly to a list."""

AggregateFn = Callable[[list[Any]], Any]
"""Map result reducer: ``list[R] -> R'``. May be async. If omitted, .map returns the list as-is."""

StateProject = Callable[[Any], Any]
"""Scoped-state projection: ``(parent_state) -> child_state``. May be async.

Runs once around a :meth:`Flow.call` subflow, once around a :meth:`Flow.loop`
(before the first iteration), and once per item for :meth:`Flow.map`.
"""

StateMerge = Callable[[Any, Any], Any]
"""Scoped-state merge: ``(parent_state, child_state) -> None``. May be async.

Runs only when the isolated block completes successfully. Return value is
ignored — mutate ``parent_state`` in place.
"""


_UNSET: Any = object()
"""Sentinel used to distinguish "kwarg omitted" from "kwarg = None"."""


@dataclass(frozen=True)
class Failure:
    """Placeholder for a failed item in :meth:`Flow.map` when ``strict=False``.

    Exposes the raised exception and the input item that produced it so
    downstream aggregators can partition successes from failures without
    losing either.
    """

    exception: BaseException
    """The exception the item's subflow raised (never :class:`asyncio.CancelledError`)."""

    item: Any
    """The input item whose subflow run failed."""


@dataclass(frozen=True)
class _RunEnv:
    """Per-run environment threaded through the execution helpers.

    Bundles the runtime flow (factory + saia cache + logger source), the
    currently active scoped state, and the run-wide global state so helpers
    do not each need to carry the three as separate positional arguments.
    ``lg`` is cached off ``runtime`` at the top of :meth:`Flow.run` for
    brevity in the debug/warning call sites.
    """

    runtime: Flow
    state: Any
    global_state: Any
    lg: Logger


@dataclass
class _Branch:
    """Composition-graph node: run one of two subflows based on a predicate."""

    when: WhenFn
    then_flow: Flow
    else_flow: Flow | None


@dataclass
class _Loop:
    """Composition-graph node: iterate a subflow until a bound or predicate fires."""

    body: Flow
    until: UntilFn | None
    max_iters: int | None
    deadline: float | None
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None


@dataclass
class _Map:
    """Composition-graph node: fan out a subflow over items and (optionally) reduce."""

    body: Flow
    items: ItemsFn | None
    aggregate: AggregateFn | None
    strict: bool
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None


@dataclass
class _Node:
    """One step in a Flow composition chain."""

    target: Any
    project: ProjectFn | None = None
    rescue: RescuePolicy | None = None
    after: AfterHook | None = None
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None


class Flow:
    """Verb registry + role-routed dispatch + fluent composition graph.

    A ``Flow`` used as the top-level runtime is constructed with a factory.
    Subflows used only for composition can be constructed without one — at
    :meth:`run` time they borrow the factory (and saia cache) of the flow
    that invoked them.
    """

    def __init__(
        self,
        lg: Logger,
        name: str = "",
        *,
        factory: SAIAFactory | None = None,
        state: Any = None,
    ) -> None:
        """Initialize a flow.

        Args:
            lg: Logger instance for tracing execution.
            name: Optional identifier — used in error messages and traces.
                Also lets a flow serve as a named node inside a parent chain.
            factory: Builds role-bound saia instances. Required for a
                top-level runtime. Optional for a subflow — the runtime it
                runs under supplies one.
            state: User-owned shared state object. Verbs read and (typically)
                mutate it in place. Opaque to the flow; may be overridden per
                :meth:`run` invocation.
        """
        self._lg = lg
        self._name = name
        self._factory = factory
        self._state = state
        self._verbs: dict[str, Any] = {}
        self._saia_by_role: dict[Role, Any] = {}
        self._nodes: list[_Node] = []

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The flow's identifier (empty string when unnamed)."""
        return self._name

    @property
    def state(self) -> Any:
        """The flow's default shared state object (user-owned)."""
        return self._state

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, verb: Any, name: str | None = None) -> None:
        """Register a verb under a name (default: the verb's ``__name__``).

        The verb must carry a ``role`` attribute of type :class:`Role`.
        """
        if not callable(verb):
            raise TypeError(f"verb must be callable; got {type(verb).__name__}")
        if not hasattr(verb, "role"):
            raise TypeError(
                f"verb must carry a .role attribute; got {type(verb).__name__} without one"
            )
        if not isinstance(verb.role, Role):
            raise TypeError(f"verb.role must be a Role instance; got {type(verb.role).__name__}")
        resolved_name = name or getattr(verb, "__name__", None)
        if not resolved_name:
            raise TypeError("verb has no __name__ and no explicit name was provided")
        if resolved_name in self._verbs:
            raise ValueError(f"verb {resolved_name!r} already registered")
        verb._registered_name = resolved_name
        self._verbs[resolved_name] = verb

    def registered(self, name: str) -> bool:
        """Return True if a verb is registered under ``name``."""
        return name in self._verbs

    # -------------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------------

    async def dispatch(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a registered verb by name, awaiting its result.

        The verb receives a fresh :class:`Context` as its first argument,
        followed by ``*args`` / ``**kwargs`` from the caller. ``dispatch`` is
        the low-level entrypoint used by :class:`Panel` and by verbs that
        invoke sibling verbs directly; the ctx here is not built by a
        composition run, so ``ctx.global_state`` is a fresh empty ``dict``.
        Verbs that need a live run-wide container must be reached via
        :meth:`run` rather than dispatched ad hoc.
        """
        if name not in self._verbs:
            raise KeyError(f"no verb registered under name {name!r}")
        verb = self._verbs[name]
        self._lg.debug("dispatching verb", extra={"verb": name, "role": verb.role.name})
        saia = self._saia_for(verb.role)
        ctx = Context(
            saia=saia,
            role=verb.role,
            state=self._state,
            global_state={},
            flow=self,
        )
        return await verb(ctx, *args, **kwargs)

    # -------------------------------------------------------------------------
    # Fluent composition
    # -------------------------------------------------------------------------

    def call(
        self,
        target: Any,
        *,
        project: ProjectFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
    ) -> Flow:
        """Append a node to the composition chain.

        ``target`` is a verb (any async callable carrying a ``.role``) or
        another :class:`Flow`. The first appended node receives the args
        passed to :meth:`run`; each subsequent node receives the previous
        node's result (optionally reshaped by ``project``). Note: ``project``
        has no effect on the first node — there is no previous result to
        transform.

        Hooks may be attached inline (kwargs) or via chained
        :meth:`rescue` / :meth:`after` calls — the two forms are equivalent.

        When ``target`` is a :class:`Flow`, ``state`` / ``merge`` open a
        scoped state channel for the subflow: ``state(parent_state)``
        produces the child's ``ctx.state``, and ``merge(parent_state,
        child_state)`` runs once after the subflow returns successfully.
        Both are rejected when ``target`` is a verb (verbs have no scoped
        state to isolate); ``merge`` also requires ``state`` — nothing to
        merge without an isolated child.

        Returns ``self`` for chaining.
        """
        _validate_target(target)
        if target is self:
            label = self._name or "<anonymous>"
            raise ValueError(f"Flow {label!r} cannot call itself as a node")
        if (state is not None or merge is not None) and not isinstance(target, Flow):
            raise TypeError(
                ".call(state=/merge=) is only valid when target is a Flow; "
                f"got {type(target).__name__}"
            )
        _require_state_for_merge(state, merge, ".call")
        self._nodes.append(
            _Node(
                target=target,
                project=project,
                rescue=rescue,
                after=after,
                state_fn=state,
                merge_fn=merge,
            )
        )
        return self

    def then(
        self,
        target: Any,
        *,
        project: ProjectFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
    ) -> Flow:
        """Append a chained node — semantic alias for :meth:`call`.

        Provided for readability: ``.call(a).then(b).then(c)`` reads as a
        pipeline. Positionally identical to :meth:`call` — data flow is
        determined by position in the chain, not by which method was used.
        """
        return self.call(
            target,
            project=project,
            rescue=rescue,
            after=after,
            state=state,
            merge=merge,
        )

    def rescue(self, policy: RescuePolicy) -> Flow:
        """Attach a failure policy to the most recently appended node.

        The policy runs when the node raises anything other than
        :class:`asyncio.CancelledError` (cancellation is never rescued).
        Signature: ``(exception, ctx) -> fallback`` — may be async.
        """
        if not self._nodes:
            raise RuntimeError(".rescue() requires a preceding .call()/.then() step")
        self._nodes[-1].rescue = policy
        return self

    def after(self, hook: AfterHook) -> Flow:
        """Attach a success hook to the most recently appended node.

        The hook runs when the node returns without raising, after any
        ``rescue`` fallback has resolved. Signature: ``(result, ctx) -> None``
        — may be async. Return value ignored (side effects only).
        """
        if not self._nodes:
            raise RuntimeError(".after() requires a preceding .call()/.then() step")
        self._nodes[-1].after = hook
        return self

    def branch(
        self,
        *,
        when: WhenFn,
        then: Any,
        else_: Any = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
    ) -> Flow:
        """Append a conditional node: run ``then`` or ``else_`` based on ``when``.

        Args:
            when: ``(prev_result, ctx) -> bool``. May be async. Truthy → run
                the ``then`` subflow; falsy → run ``else_`` (or pass ``prev_result``
                through unchanged when ``else_`` is ``None``).
            then: A :class:`Flow` or ``lambda f: ...`` callback that mutates a
                fresh Flow. Receives the branch input as its sole positional.
            else_: Same shape as ``then``. Omitted → falsy branch is a no-op
                that returns the branch input.
            rescue: Attached to the branch node — fires if the chosen subflow
                (or the predicate) raises.
            after: Attached to the branch node — fires with the chosen
                subflow's result (or the pass-through input).

        The branch node's result is the chosen subflow's output; it becomes
        the next chain step's input like any other node's result. Both bodies
        share the parent's ``ctx.state``; wrap a body in :meth:`call` if a
        branch arm needs its own scoped state.

        Returns ``self`` for chaining.
        """
        then_flow = _materialize(then, self._lg, "branch.then")
        else_flow = _materialize(else_, self._lg, "branch.else") if else_ is not None else None
        node = _Node(
            target=_Branch(when=when, then_flow=then_flow, else_flow=else_flow),
            rescue=rescue,
            after=after,
        )
        self._nodes.append(node)
        return self

    def loop(
        self,
        body: Any,
        *,
        until: UntilFn | None = None,
        max_iters: int | None = None,
        deadline: float | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
    ) -> Flow:
        """Append a bounded loop: iterate ``body`` until a stop condition holds.

        Each iteration's return becomes the next iteration's input; the first
        iteration receives the loop node's input (previous chain step's
        result). The loop's own result is the last iteration's return.

        Args:
            body: A :class:`Flow` or ``lambda f: ...`` callback for the loop
                body. Runs at least once.
            until: ``(ctx) -> bool``. May be async. Checked **after** each
                iteration completes — truthy → stop. State typically drives
                termination via ``ctx.state``.
            max_iters: Hard upper bound on iteration count. Must be ``>= 1``.
                Reached without ``until`` firing → loop exits with the last
                iteration's result.
            deadline: Optional wall-clock budget in seconds. Checked **between**
                iterations — a running body is not interrupted, so the actual
                elapsed time may exceed ``deadline`` by one iteration.
            rescue: Attached to the loop node — fires if any iteration raises.
            after: Attached to the loop node — fires with the loop's final result.
            state: Scoped-state projection. ``state(parent_state)`` runs once
                before the first iteration; every iteration sees the same
                projected child state on ``ctx.state``. Omitted → iterations
                see the parent's ``state`` by reference.
            merge: Scoped-state merge. ``merge(parent_state, child_state)``
                runs once after the loop exits successfully (via ``until``,
                ``max_iters``, or ``deadline``). Skipped if an iteration
                raises past any ``rescue``. Requires ``state``.

        At least one of ``until`` or ``max_iters`` must be provided so the
        loop is guaranteed to terminate.

        Returns ``self`` for chaining.
        """
        if until is None and max_iters is None:
            raise ValueError(".loop() requires until= or max_iters= (or both) to terminate")
        if max_iters is not None and max_iters < 1:
            raise ValueError(f".loop(max_iters=) must be >= 1; got {max_iters}")
        if deadline is not None and deadline <= 0:
            raise ValueError(f".loop(deadline=) must be > 0; got {deadline}")
        _require_state_for_merge(state, merge, ".loop")
        body_flow = _materialize(body, self._lg, "loop.body")
        node = _Node(
            target=_Loop(
                body=body_flow,
                until=until,
                max_iters=max_iters,
                deadline=deadline,
                state_fn=state,
                merge_fn=merge,
            ),
            rescue=rescue,
            after=after,
        )
        self._nodes.append(node)
        return self

    def map(
        self,
        body: Any,
        *,
        items: ItemsFn | None = None,
        aggregate: AggregateFn | None = None,
        strict: bool = True,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
    ) -> Flow:
        """Append a parallel fan-out: run ``body`` per item concurrently.

        Args:
            body: A :class:`Flow` or ``lambda f: ...`` callback. Each item
                becomes the body's sole positional input.
            items: ``(prev_result, ctx) -> iterable``. May be async. When
                omitted, ``prev_result`` itself is treated as the iterable —
                the common shape when the previous node already produced a list.
            aggregate: ``list[R] -> R'``. Reduces per-item results into the
                map's final output. Omitted → the list is returned as-is
                (order preserved to match input item order).
            strict: ``True`` (default) → the first non-cancellation exception
                propagates out of :meth:`run`. ``False`` → each failing item is
                replaced by a :class:`Failure` sentinel in the results list so
                the aggregator can partition successes from failures.
            rescue: Attached to the map node — fires if the map itself raises
                (item resolution, body exceptions in strict mode, or aggregate).
            after: Attached to the map node — fires with the final (possibly
                aggregated) result.
            state: Scoped-state projection. Runs once **per item**, so each
                item's body sees its own isolated ``ctx.state`` — the safe
                shape for concurrent per-item scratch space. Omitted → every
                item shares the parent's ``state`` by reference (writes
                interleave; caller's responsibility).
            merge: Scoped-state merge. Runs once per item, after that item's
                body returns successfully. Because items run concurrently,
                merges interleave against the shared parent state; callers
                wanting a single sequential fold should use ``aggregate``
                (which runs once after every item completes) instead.
                Requires ``state``.

        Cancellation propagates unconditionally regardless of ``strict``.
        Sibling items keep running when one fails; the wasted work is the
        trade-off for a simple ordering guarantee.

        Returns ``self`` for chaining.
        """
        _require_state_for_merge(state, merge, ".map")
        body_flow = _materialize(body, self._lg, "map.body")
        node = _Node(
            target=_Map(
                body=body_flow,
                items=items,
                aggregate=aggregate,
                strict=strict,
                state_fn=state,
                merge_fn=merge,
            ),
            rescue=rescue,
            after=after,
        )
        self._nodes.append(node)
        return self

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def run(
        self,
        *args: Any,
        state: Any = _UNSET,
        global_state: Any = _UNSET,
        _runtime: Flow | None = None,
        _global_state: Any = _UNSET,
        **kwargs: Any,
    ) -> Any:
        """Execute the composition graph.

        The first node receives ``*args`` / ``**kwargs``. Each subsequent
        node receives one positional — the previous node's result, optionally
        transformed by its ``project`` callable.

        Cancellation is honored end-to-end: :class:`asyncio.CancelledError`
        raised by any node (verb or subflow) propagates out of :meth:`run`
        unchanged, past any ``rescue`` policy.

        Args:
            *args: Positional inputs to the first node.
            state: If provided, overrides ``self.state`` for this run and is
                exposed as ``ctx.state`` to every verb. Omitted → the flow's
                default state is used. A subflow running inside a parent
                chain always receives whatever state the containing node's
                ``state=`` projection produced (or, when no projection is
                set, the parent's active state by reference).
            global_state: Run-wide state exposed as ``ctx.global_state`` on
                every node in the tree. Defaults to a fresh empty ``dict`` at
                the top level so verbs guard on the KEY they need
                (``ctx.global_state.get("budget")``) rather than on the
                container. Explicitly passing another value (dict, dataclass,
                Pydantic model, arbitrary object) overrides the default.
                Subflows inherit the top-level container automatically — the
                kwarg on a subflow's :meth:`run` only takes effect when it is
                invoked as its own top-level run.
            _runtime: Internal — used when this flow runs as a subflow to
                share the outer flow's factory and saia cache. Not part of
                the public surface.
            _global_state: Internal — used to propagate the outermost
                ``global_state`` container down into subflows without each
                nested :meth:`run` re-defaulting to a fresh dict.
            **kwargs: Keyword inputs to the first node.

        Raises:
            RuntimeError: The flow has no nodes, or is running as a
                top-level runtime without a factory.
        """
        if not self._nodes:
            raise RuntimeError(f"Flow {self._name!r} has no nodes to run")
        env = self._build_run_env(state, global_state, _runtime, _global_state)
        label = self._name or "<anonymous>"
        is_subflow = _runtime is not None
        env.lg.debug(
            "starting flow run",
            extra={"flow": label, "nodes": len(self._nodes), "subflow": is_subflow},
        )
        result: Any = _UNSET
        for index, node in enumerate(self._nodes):
            node_args, node_kwargs = _step_inputs(index, node, result, args, kwargs)
            ctx = _build_ctx(node.target, env)
            result = await _execute_node(node, ctx, env, node_args, node_kwargs)
        env.lg.debug("completed flow run", extra={"flow": label, "subflow": is_subflow})
        return result

    def _build_run_env(
        self,
        state: Any,
        global_state: Any,
        _runtime: Flow | None,
        _global_state: Any,
    ) -> _RunEnv:
        """Resolve the effective runtime, scoped state, and global state for a run.

        Encapsulates the subflow-vs-top-level defaults so :meth:`run` stays a
        thin driver over the node loop. Raises when the resolved runtime has
        no factory — a top-level flow must supply one at construction.
        """
        runtime = _runtime if _runtime is not None else self
        if runtime._factory is None:
            label = runtime._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} has no SAIAFactory — provide factory=... "
                "when constructing the top-level Flow"
            )
        active_state = self._state if state is _UNSET else state
        is_subflow = _runtime is not None
        if is_subflow and global_state is _UNSET:
            active_global = {} if _global_state is _UNSET else _global_state
        else:
            active_global = {} if global_state is _UNSET else global_state
        return _RunEnv(
            runtime=runtime, state=active_state, global_state=active_global, lg=runtime._lg
        )

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _saia_for(self, role: Role) -> Any:
        """Return a cached saia for ``role``, building it on first request."""
        cached = self._saia_by_role.get(role)
        if cached is not None:
            return cached
        if self._factory is None:
            label = self._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} has no SAIAFactory — factory= was not supplied at "
                f"construction (needed to build saia for role {role.name!r})"
            )
        built = self._factory.build(role)
        self._saia_by_role[role] = built
        return built


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _validate_target(target: Any) -> None:
    """Reject anything that isn't a verb (callable with a Role) or a Flow."""
    if isinstance(target, Flow):
        return
    if not callable(target):
        raise TypeError(
            f"target must be a verb (callable with .role) or a Flow; got {type(target).__name__}"
        )
    if not hasattr(target, "role"):
        raise TypeError(f"verb target must carry a .role attribute; got {type(target).__name__}")
    if not isinstance(target.role, Role):
        raise TypeError(
            f"verb target .role must be a Role instance; got {type(target.role).__name__}"
        )


def _require_state_for_merge(
    state: StateProject | None, merge: StateMerge | None, method: str
) -> None:
    """Enforce that ``merge=`` is only legal with ``state=`` (nothing to merge otherwise)."""
    if merge is not None and state is None:
        raise ValueError(
            f"{method}(merge=...) requires state= "
            "(nothing to merge back without an isolated child state)"
        )


def _build_ctx(target: Any, env: _RunEnv) -> Context:
    """Build the Context passed to the node's verb (and to its hooks).

    Verb nodes get a role-bound ctx (role + saia populated). Subflow nodes
    and control-flow nodes (branch/loop/map) get an ambient ctx — role and
    saia are ``None`` because those nodes have no single role; each inner
    verb builds its own role-bound ctx as it runs.
    """
    if isinstance(target, Flow | _Branch | _Loop | _Map):
        return Context(
            saia=None,
            role=None,
            state=env.state,
            global_state=env.global_state,
            flow=env.runtime,
        )
    saia = env.runtime._saia_for(target.role)
    return Context(
        saia=saia,
        role=target.role,
        state=env.state,
        global_state=env.global_state,
        flow=env.runtime,
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
        fallback = node.rescue(exc, ctx)
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
    target = node.target
    if isinstance(target, Flow):
        return await _run_subflow(target, env, node.state_fn, node.merge_fn, node_args, node_kwargs)
    if isinstance(target, _Branch):
        return await _run_branch(target, ctx, env, node_args)
    if isinstance(target, _Loop):
        return await _run_loop(target, ctx, env, node_args)
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
    result = await body.run(
        *node_args,
        state=child_state,
        _runtime=env.runtime,
        _global_state=env.global_state,
        **node_kwargs,
    )
    await _merge_state(merge_fn, env.state, child_state)
    return result


async def _project_state(state_fn: StateProject | None, parent_state: Any) -> Any:
    """Build the child state for a scoped block; pass-through when unset."""
    if state_fn is None:
        return parent_state
    child = state_fn(parent_state)
    if inspect.isawaitable(child):
        child = await child
    return child


async def _merge_state(merge_fn: StateMerge | None, parent_state: Any, child_state: Any) -> None:
    """Fold the child state back into the parent; no-op when unset."""
    if merge_fn is None:
        return
    result = merge_fn(parent_state, child_state)
    if inspect.isawaitable(result):
        await result


async def _check_until(until_fn: UntilFn | None, child_state: Any, env: _RunEnv) -> bool:
    """Evaluate the loop's until predicate with a ctx bound to the loop's scoped state."""
    if until_fn is None:
        return False
    ctx = Context(
        saia=None, role=None, state=child_state, global_state=env.global_state, flow=env.runtime
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
    return await chosen.run(
        prev_result,
        state=env.state,
        _runtime=env.runtime,
        _global_state=env.global_state,
    )


async def _run_loop(
    lp: _Loop,
    ctx: Context,
    env: _RunEnv,
    node_args: tuple[Any, ...],
) -> Any:
    """Iterate the loop body under bounds, threading each result to the next.

    Post-check semantics: the body runs at least once, then ``until`` (if
    set) is evaluated. ``max_iters`` and ``deadline`` bound the total
    iteration count and elapsed wall clock respectively. Scoped state is
    projected once before the first iteration; every iteration sees the same
    child state, and the merge fires once after the loop exits successfully.
    """
    child_state = await _project_state(lp.state_fn, env.state)
    result: Any = node_args[0] if node_args else None
    started = time.monotonic()
    iteration = 0
    while True:
        if lp.max_iters is not None and iteration >= lp.max_iters:
            break
        if lp.deadline is not None and time.monotonic() - started >= lp.deadline:
            break
        result = await lp.body.run(
            result,
            state=child_state,
            _runtime=env.runtime,
            _global_state=env.global_state,
        )
        iteration += 1
        if await _check_until(lp.until, child_state, env):
            break
    await _merge_state(lp.merge_fn, env.state, child_state)
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
    every position. Cancellation propagates unconditionally in both modes.
    Scoped state is projected per item; per-item merges run only for items
    that completed successfully.
    """
    prev_result = node_args[0] if node_args else None
    items = await _resolve_items(mp.items, prev_result, ctx)
    if mp.strict:
        coros = [
            _run_map_item_strict(mp.body, item, env, mp.state_fn, mp.merge_fn) for item in items
        ]
        results = list(await asyncio.gather(*coros, return_exceptions=True))
        for r in results:
            if isinstance(r, asyncio.CancelledError):
                raise r
            if isinstance(r, BaseException):
                raise r
    else:
        coros = [_run_map_item(mp.body, item, env, mp.state_fn, mp.merge_fn) for item in items]
        results = list(await asyncio.gather(*coros))
    result = mp.aggregate(results) if mp.aggregate is not None else results
    if inspect.isawaitable(result):
        result = await result
    return result


async def _run_map_item_strict(
    body: Flow,
    item: Any,
    env: _RunEnv,
    state_fn: StateProject | None,
    merge_fn: StateMerge | None,
) -> Any:
    """Run one strict-mode map item; merge fires only when the body succeeds."""
    child_state = await _project_state(state_fn, env.state)
    result = await body.run(
        item,
        state=child_state,
        _runtime=env.runtime,
        _global_state=env.global_state,
    )
    await _merge_state(merge_fn, env.state, child_state)
    return result


async def _run_map_item(
    body: Flow,
    item: Any,
    env: _RunEnv,
    state_fn: StateProject | None,
    merge_fn: StateMerge | None,
) -> Any:
    """Run one non-strict map item; wrap non-cancellation exceptions as :class:`Failure`.

    Merge fires only for items that complete successfully — a failed item's
    partially-mutated child state is discarded.
    """
    child_state = await _project_state(state_fn, env.state)
    try:
        result = await body.run(
            item,
            state=child_state,
            _runtime=env.runtime,
            _global_state=env.global_state,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return Failure(exception=exc, item=item)
    await _merge_state(merge_fn, env.state, child_state)
    return result


async def _resolve_items(items_fn: ItemsFn | None, prev_result: Any, ctx: Context) -> list[Any]:
    """Materialize the map input list from ``items_fn`` (or ``prev_result``)."""
    source = prev_result if items_fn is None else items_fn(prev_result, ctx)
    if inspect.isawaitable(source):
        source = await source
    try:
        return list(source)
    except TypeError as exc:
        raise TypeError(f".map items must be iterable; got {type(source).__name__}") from exc


def _materialize(buildable: Any, lg: Logger, name: str) -> Flow:
    """Turn a :data:`Buildable` (Flow or ``lambda f: ...`` callback) into a Flow.

    A ``Flow`` is returned as-is; a callable is invoked against a fresh Flow
    it may mutate (the return value, if any, is ignored). Anything else is a
    :class:`TypeError` — bad Buildables fail eagerly at build time, not at
    :meth:`Flow.run` time.
    """
    if isinstance(buildable, Flow):
        return buildable
    if not callable(buildable):
        raise TypeError(
            f"expected a Flow or a lambda f: f.call(...) callback for {name!r}; "
            f"got {type(buildable).__name__}"
        )
    fresh = Flow(lg, name)
    buildable(fresh)
    return fresh


def _target_label(target: Any) -> str:
    """Return a human-readable label for a node target."""
    if isinstance(target, Flow):
        return f"Flow({target.name!r})" if target.name else "Flow(<anonymous>)"
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Loop):
        return "loop"
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
