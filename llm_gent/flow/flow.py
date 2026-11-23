# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow — verb registry, role-routed dispatch, and fluent composition graph.

A :class:`Flow` plays two roles that share one object:

1. **Registry.** An optional verb-by-name registry used by :meth:`dispatch`.

2. **Composition graph.** A sequence of nodes built up via the fluent
   methods :meth:`call` / :meth:`then` / :meth:`rescue` / :meth:`after` /
   :meth:`branch` / :meth:`iterate` / :meth:`map` and executed by a
   :class:`~llm_gent.flow.FlowRunner`. A node's target is either a verb (any callable carrying a
   ``.role``), another :class:`Flow`, or a control-flow primitive (branch,
   iterate, map) whose bodies are themselves subflows. Every flow of a run
   uses the run's environment, so one SAIA per role serves the whole tree.

Both roles are optional. A top-level flow that only serves as a verb
registry never needs to call the fluent methods.

At application boundaries, prefer :class:`Factory` from
:mod:`.factory` — it captures the ambient ``lg`` and the shared declarations
once so per-subsystem construction reads as
``f.create("grade").call(...)`` rather than repeating them at every site.
Direct :class:`Flow` construction is still supported and is what the
executor uses internally to materialize lambda-form subflows.

State is exposed on every :class:`Context` as a single :class:`State`
wrapper:

- ``ctx.state.data`` is the enclosing scope's payload — shared with the
  parent by reference by default. The ``state=`` / ``merge=`` kwargs on
  :meth:`Flow.call`, :meth:`Flow.iterate`, and :meth:`Flow.map` project an
  isolated child payload for the block they contain and (optionally) merge it
  back when the block completes successfully.
- ``ctx.state.root().data`` is the run-wide payload — the run's
  ``state=`` argument (default: fresh empty
  ``dict``). Every node in the tree reaches it via the same call regardless
  of nesting depth or per-scope projection.

Execution (the run, the walk, scoped-state projection/merge, history
writing) lives in :mod:`llm_gent.flow.run`, which imports this module and
is never imported by it. The private dataclasses and the :class:`Failure`
sentinel live in :mod:`.nodes`. This module owns the :class:`Flow` class
itself plus the builder-side helpers used by its fluent methods —
:func:`_validate_target`, :func:`_require_state_for_merge`, and the eager
Buildable materializer :func:`_materialize`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import MappingProxyType
from typing import Any, ParamSpec, Self, TypeVar

from appinfra.log import Logger

from ..core.platform import Platform
from ..core.role import Role
from ._node_id import iter_flows
from ._shortcut import Shortcut, check_shortcut
from ._validation import (
    _check_node_name,
    _map_bodies,
    _materialize,
    _require_state_for_merge,
    _validate_target,
    check_concurrency,
)
from .checkpoint import (
    CheckpointPolicy,
    checkpoint_tag,
)
from .context import Context
from .decl import Decl
from .nodes import (
    UNSET,
    AfterHook,
    AggregateFn,
    Checkpointer,
    GuardFn,
    ItemsFn,
    MaxConcurrencyFn,
    OnErrorFn,
    OnItemCompleteFn,
    ProjectFn,
    RescuePolicy,
    StateMerge,
    StateProject,
    UntilFn,
    WhenFn,
    _Branch,
    _Iterate,
    _Map,
    _Node,
)
from .resource import COST, R, ResourceKey, check_resource
from .resource._runtime import Resources
from .resource.cost import check_budget, check_not_a_tracker
from .state import State, StateFactory
from .structure import Structure


P = ParamSpec("P")
R_ = TypeVar("R_")


def _builder(method: Callable[P, R_]) -> Callable[P, R_]:
    """Mark a method that changes the flow: it raises once the flow has been started."""

    @functools.wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R_:
        _require_open(args[0], method.__name__)
        return method(*args, **kwargs)

    return guarded


def _require_open(flow: Any, method: str) -> None:
    """Raise when ``flow`` is frozen: it has been started, and its runs share its structure."""
    if flow._frozen:
        label = flow._name or "<anonymous>"
        raise RuntimeError(
            f"Flow {label!r} has been started and can no longer change (.{method}()): "
            f"its runs share its structure; build the changed flow anew"
        )


class Flow:
    """Verb registry + role-routed dispatch + fluent composition graph.

    A flow is structure: what its runs may use (SAIA factory, traits) is
    the run's environment, given to the runner that starts it, and every
    flow of the run, nested ones included, uses that one.

    Subclassing is supported: every fluent method returns ``Self``, and
    the subflows a ``lambda b: ...`` body builds are of the enclosing
    flow's class, so a subclass's own methods (e.g.
    :func:`~llm_gent.flow.resource_method`) are there in bodies too. A
    subclass keeps this constructor's signature: the framework builds
    those subflows as ``cls(lg=lg, name=name)``.
    """

    def __init__(
        self,
        lg: Logger,
        name: str = "",
        *,
        state_factory: StateFactory[Any] | None = None,
    ) -> None:
        """Initialize a flow.

        Prefer :class:`Factory` at application boundaries — it captures
        the ambient ``lg`` and the shared declarations once so
        Flow-per-subsystem construction doesn't repeat them. Constructing
        :class:`Flow` directly is still supported; the executor uses it
        internally to materialize subflows built with the fluent lambda
        form, and it remains valid for lower-level tests.

        Args:
            lg: Logger instance for tracing execution.
            name: Optional identifier — used in error messages and traces.
                Also lets a flow serve as a named node inside a parent chain.
            state_factory: A :class:`StateFactory` for this flow's state:
                ``state_factory.new()`` builds the state of a run started
                without ``state=``, and ``state_factory.restore(...)``
                rebuilds ``ctx.state.data`` from a checkpoint on resume. For
                state that carries no runtime handles wrap the type in
                :class:`TypeStateFactory`; for state that binds a Logger /
                storage / connection at restore, implement
                :class:`StateFactory` directly. ``None`` (default): a run
                given no state gets a fresh empty ``dict``, which
                round-trips through checkpoints as-is.

        Where runs keep their histories is the runner's
        (:class:`~llm_gent.flow.FlowRunner` ``store=``), not the flow's.
        """
        self._lg = lg
        self._name = name
        self._state_factory = state_factory
        self._shortcut: Shortcut | None = None
        self._resources: dict[ResourceKey[Any], Any] = {}
        self._resource_children: dict[ResourceKey[Any], dict[str, Any]] = {}
        self._checkpointer: Checkpointer | None = None
        self._verbs: dict[str, Any] = {}
        self._nodes: list[_Node] = []
        self._checkpoint_policy: CheckpointPolicy | None = None
        self._hooks: tuple[object, ...] = ()
        self._hooks_replace = False
        self._frozen = False

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The flow's identifier (empty string when unnamed)."""
        return self._name

    @property
    def frozen(self) -> bool:
        """The flow has been started (or runs inside one that has): it no longer changes.

        A started flow's runs share its nodes and declarations, so every
        method that would change them raises from then on.
        """
        return self._frozen

    def _freeze(self) -> None:
        """Freeze this flow and every flow under it: called when a run of it starts."""
        if self._frozen:
            return
        for flow in iter_flows(self):
            flow._frozen = True

    @property
    def decl(self) -> Decl:
        """What this flow declares for its subtree, as the walk applies it on entry.

        Its checkpointer and checkpoint policy, resources and the per-run
        children it asks for, hooks and its shortcut; unset fields inherit
        from the enclosing scope (:meth:`~llm_gent.flow.run.Scope.child`).
        """
        return Decl(
            checkpointer=self._checkpointer,
            policy=self._checkpoint_policy,
            resources=MappingProxyType(self._resources),
            resource_children=MappingProxyType(self._resource_children),
            hooks=self._hooks,
            hooks_replace=self._hooks_replace,
            shortcut=self._shortcut,
        )

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    @_builder
    def register(self, verb: Any, name: str | None = None) -> None:
        """Register a verb under a name (default: the verb's ``__name__``).

        The verb must carry a ``.role`` attribute; its value is either a
        :class:`Role` (role-bound verb) or ``None`` (pure-Python verb —
        ``ctx.saia`` stays ``None`` on dispatch).
        """
        if not callable(verb):
            raise TypeError(f"verb must be callable; got {type(verb).__name__}")
        if not hasattr(verb, "role"):
            raise TypeError(
                f"verb must carry a .role attribute; got {type(verb).__name__} without one"
            )
        if verb.role is not None and not isinstance(verb.role, Role):
            raise TypeError(
                f"verb.role must be a Role instance or None; got {type(verb.role).__name__}"
            )
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

    async def dispatch(
        self,
        name: str,
        *args: Any,
        halt: Any = UNSET,
        cost: Any = UNSET,
        resources: Any = UNSET,
        scope_state: Any = UNSET,
        extra: Any = UNSET,
        platform: Platform,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a registered verb by name, awaiting its result.

        The verb receives a fresh :class:`Context` as its first argument,
        followed by ``*args`` / ``**kwargs`` from the caller. ``dispatch`` is
        the low-level entrypoint for verbs that invoke sibling verbs
        directly.

        Pass ``scope_state=ctx.state`` from an in-flight verb to hand the
        dispatched sibling the live scope payload. Omitting ``scope_state=``
        (or passing ``UNSET``) gives the sibling a fresh state, as a
        run started without ``state=`` gets. A ``State`` instance passes
        through as-is; any other value is wrapped with this flow's
        ``state_factory``.

        Pass ``halt=ctx.halt`` and ``resources=ctx.resources`` from an
        in-flight verb to propagate its effective ambients to the
        dispatched sibling; omitting either (or passing ``UNSET``) gives
        no halt and the resources this flow declares (no per-run children
        such as ``.with_budget()``'s: there is no run). ``cost=`` sets the sibling's ``ctx.cost`` on top of
        those — ``None`` for none.

        Pass ``extra=ctx.extra`` from an in-flight verb to propagate the
        caller-supplied opaque dict to the dispatched sibling. Omitting
        (or passing ``UNSET``) yields a fresh empty dict at the sibling —
        ``dispatch`` has no flow-level ``.with_extra()`` fallback.

        ``platform`` is the platform the sibling runs on: pass
        ``platform=ctx.platform`` from an in-flight verb so its
        ``ctx.saia`` and ``ctx.traits`` are the run's.
        """
        if name not in self._verbs:
            raise KeyError(f"no verb registered under name {name!r}")
        verb = self._verbs[name]
        role_name = verb.role.name if verb.role is not None else None
        self._lg.debug("dispatching verb", extra={"verb": name, "role": role_name})
        payload = self._new_state() if scope_state is UNSET else scope_state
        effective_halt = None if halt is UNSET else halt
        effective_extra: dict[str, Any] = {} if extra is UNSET else extra
        wrapped_state = (
            payload
            if isinstance(payload, State)
            else State(data=payload, _factory=self._state_factory)
        )
        ctx = Context(
            role=verb.role,
            state=wrapped_state,
            flow=self,
            platform=platform,
            halt=effective_halt,
            extra=effective_extra,
            resources=self._dispatch_resources(resources, cost),
        )
        return await verb(ctx, *args, **kwargs)

    def _dispatch_resources(self, resources: Any, cost: Any) -> Resources:
        """The resources a :meth:`dispatch` hands its verb: ``resources``, else this flow's.

        ``cost`` (unless :data:`UNSET`) replaces the cost tracker in them;
        ``None`` removes it.
        """
        found: dict[ResourceKey[Any], Any] = dict(
            self._resources if resources is UNSET else resources
        )
        if cost is not UNSET:
            found.pop(COST, None)
            if cost is not None:
                found[COST] = cost
        return found

    # -------------------------------------------------------------------------
    # Fluent composition
    # -------------------------------------------------------------------------

    @_builder
    def call(
        self,
        target: Any,
        *,
        project: ProjectFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
        state_factory: StateFactory[Any] | None = None,
        name: str | None = None,
    ) -> Self:
        """Append a node to the composition chain.

        ``target`` is a verb (any async callable carrying a ``.role``) or
        another :class:`Flow`. The first appended node receives the args
        passed to :meth:`FlowRunner.start`; each subsequent node receives the previous
        node's result (optionally reshaped by ``project``). Note: ``project``
        has no effect on the first node — there is no previous result to
        transform.

        Threading of the previous result is opt-in at the verb signature:
        ``async def v(ctx)`` drops the value, ``async def v(ctx, prev)``
        (or ``*args``) consumes it. State (``ctx.state.data``) is the
        channel for multi-hop handoffs, loop / branch predicates, and any
        value that must survive checkpoint / resume. Both channels may be
        used together when the returned value has more than one consumer
        (see "Return values vs state" in ``docs/index.md``).

        Hooks may be attached inline (kwargs) or via chained
        :meth:`rescue` / :meth:`after` calls — the two forms are equivalent.

        When ``target`` is a :class:`Flow`, ``state`` / ``merge`` open a
        scoped state channel for the subflow: ``state(parent_state)``
        produces the child's ``ctx.state``, and ``merge(parent_state,
        child_state)`` runs once after the subflow returns successfully.
        Both are rejected when ``target`` is a verb (verbs have no scoped
        state to isolate); ``merge`` also requires ``state`` — nothing to
        merge without an isolated child.

        ``name`` labels the step: it enters the step's node id (see
        :meth:`iterate`).

        Returns ``self`` for chaining.
        """
        _validate_target(target)
        _check_node_name(name, ".call")
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
                state_factory=state_factory,
                name=name,
            )
        )
        return self

    @_builder
    def then(
        self,
        target: Any,
        *,
        project: ProjectFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
        state_factory: StateFactory[Any] | None = None,
        name: str | None = None,
    ) -> Self:
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
            state_factory=state_factory,
            name=name,
        )

    @_builder
    def conclude(
        self,
        target: Any,
        *,
        project: ProjectFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
        state_factory: StateFactory[Any] | None = None,
        name: str | None = None,
    ) -> Self:
        """Append a step like :meth:`then` that a cut does not skip.

        Without a cut it is exactly :meth:`then`. While its chain
        fast-forwards — the chain runs in a :meth:`with_shortcut` region
        whose signal is set — every other step is skipped and the
        ``conclude`` steps still run, in order, each with the last
        completed result (the chain's input when none completed). A chain
        the region reaches before its first step runs them too. A ``then``
        after a ``conclude`` is still skipped.

        It is not a ``finally``: a step that raises, or the run's halt,
        stops the chain as usual. Being a ``conclude`` step does not enter
        its node id or :meth:`root_hash`: switching a step between
        ``then`` and ``conclude`` keeps checkpoints resumable.

        Returns ``self`` for chaining.
        """
        self.call(
            target,
            project=project,
            rescue=rescue,
            after=after,
            state=state,
            merge=merge,
            state_factory=state_factory,
            name=name,
        )
        self._nodes[-1].conclude = True
        return self

    @_builder
    def rescue(self, policy: RescuePolicy) -> Self:
        """Attach a failure policy to the most recently appended node.

        The policy runs when the node raises anything other than
        :class:`asyncio.CancelledError` (cancellation is never rescued).
        Signature: ``(exception, pending_input, ctx) -> fallback`` — may be async.
        """
        if not self._nodes:
            raise RuntimeError(".rescue() requires a preceding .call()/.then() step")
        self._nodes[-1].rescue = policy
        return self

    @_builder
    def after(self, hook: AfterHook) -> Self:
        """Attach a success hook to the most recently appended node.

        The hook runs when the node returns without raising, after any
        ``rescue`` fallback has resolved. Signature: ``(result, ctx) -> None``
        — may be async. Return value ignored (side effects only).
        """
        if not self._nodes:
            raise RuntimeError(".after() requires a preceding .call()/.then() step")
        self._nodes[-1].after = hook
        return self

    @_builder
    def branch(
        self,
        *,
        when: WhenFn,
        then: Any,
        else_: Any = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        name: str | None = None,
    ) -> Self:
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
            name: Stable label for the node's id (see :meth:`iterate`).

        The branch node's result is the chosen subflow's output; it becomes
        the next chain step's input like any other node's result. Both bodies
        share the parent's ``ctx.state``; wrap a body in :meth:`call` if a
        branch arm needs its own scoped state.

        Returns ``self`` for chaining.
        """
        _check_node_name(name, ".branch")
        cls = type(self)
        then_flow = _materialize(then, self._lg, "branch.then", cls)
        else_flow = _materialize(else_, self._lg, "branch.else", cls) if else_ is not None else None
        node = _Node(
            target=_Branch(when=when, then_flow=then_flow, else_flow=else_flow, name=name),
            rescue=rescue,
            after=after,
        )
        self._nodes.append(node)
        return self

    @_builder
    def iterate(
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
        state_factory: StateFactory[Any] | None = None,
        name: str | None = None,
    ) -> Self:
        """Append a bounded iteration: run ``body`` until a stop condition holds.

        Each iteration's return becomes the next iteration's input; the first
        iteration receives the iterate node's input (previous chain step's
        result). The node's own result is the last iteration's return.

        Args:
            body: A :class:`Flow` or ``lambda f: ...`` callback for the body.
                Runs at least once.
            until: ``(result, ctx) -> bool``. May be async. Checked **after**
                each iteration completes — truthy → stop. ``result`` is that
                iteration's body return; ``ctx.state`` carries any mutations
                the body made. Either signal — or both — can drive termination.
            max_iters: Hard upper bound on iteration count. Must be ``>= 1``.
                Reached without ``until`` firing → exits with the last
                iteration's result.
            deadline: Optional wall-clock budget in seconds. Checked **between**
                iterations — a running body is not interrupted, so the actual
                elapsed time may exceed ``deadline`` by one iteration.
            rescue: Attached to the iterate node — fires if any iteration raises.
            after: Attached to the iterate node — fires with the final result.
            state: Scoped-state projection. ``state(parent_state)`` runs once
                before the first iteration; every iteration sees the same
                projected child state on ``ctx.state``. Omitted → iterations
                see the parent's ``state`` by reference.
            merge: Scoped-state merge. ``merge(parent_state, child_state)``
                runs once after the block exits successfully (via ``until``,
                ``max_iters``, or ``deadline``). Skipped if an iteration
                raises past any ``rescue``. Requires ``state``.
            name: Stable label folded into the node's id, so its checkpoints
                stay bound to it when other iterate steps are added, removed
                or reordered in the same chain. Omitted → the node
                is identified by its order among the chain's unnamed iterate
                steps. Must be non-empty when given.

        At least one of ``until`` or ``max_iters`` must be provided so the
        iteration is guaranteed to terminate. The run's halt
        (:meth:`FlowRunner.start` ``halt=``), if set, is checked between iterations and
        terminates the block gracefully once fired.

        Returns ``self`` for chaining.
        """
        if until is None and max_iters is None:
            raise ValueError(".iterate() requires until= or max_iters= (or both) to terminate")
        if max_iters is not None and max_iters < 1:
            raise ValueError(f".iterate(max_iters=) must be >= 1; got {max_iters}")
        if deadline is not None and deadline <= 0:
            raise ValueError(f".iterate(deadline=) must be > 0; got {deadline}")
        _require_state_for_merge(state, merge, ".iterate")
        _check_node_name(name, ".iterate")
        body_flow = _materialize(body, self._lg, "iterate.body", type(self))
        node = _Node(
            target=_Iterate(
                body=body_flow,
                until=until,
                max_iters=max_iters,
                deadline=deadline,
                state_fn=state,
                merge_fn=merge,
                state_factory=state_factory,
                name=name,
            ),
            rescue=rescue,
            after=after,
        )
        self._nodes.append(node)
        return self

    @_builder
    def map(
        self,
        body: Any,
        *,
        items: ItemsFn | None = None,
        aggregate: AggregateFn | None = None,
        strict: bool = True,
        max_concurrency: int | MaxConcurrencyFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
        state_factory: StateFactory[Any] | None = None,
        name: str | None = None,
    ) -> Self:
        """Append a parallel fan-out: run ``body`` per item, or each member once, concurrently.

        A map over items runs one body on each item. A map over members —
        ``body`` a list — runs each member once on the step's input, one
        item per member: an ensemble (``.map([judge_a, judge_b, judge_c],
        aggregate=majority)``). Every option below applies to both. A
        member is identified by what it runs and its order among members
        running the same thing, so a resumed run matches finished members
        to their results after the list was reordered; an added member
        runs, a removed one is dropped.

        Args:
            body: A :class:`Flow`, a verb or a ``lambda f: ...`` callback.
                Each item becomes the body's sole positional input. A list
                of those makes a map over members.
            items: ``(prev_result, ctx) -> iterable``. May be async. When
                omitted, ``prev_result`` itself is treated as the iterable —
                the common shape when the previous node already produced a list.
                Not with members: they run on ``prev_result``.
            aggregate: ``list[R] -> R'``. Reduces per-item results into the
                map's final output. Omitted → the list is returned as-is
                (order preserved to match input item order, or member
                order). :func:`~llm_gent.flow.aggregate.majority`,
                ``unanimous``, ``mean`` and ``weighted`` combine an
                ensemble's votes.
            strict: ``True`` (default) → the first non-cancellation exception
                propagates out of the run. ``False`` → each failing item is
                replaced by a :class:`Failure` sentinel in the results list so
                the aggregator can partition successes from failures. Under
                ``strict=False`` a state-projection failure is also wrapped
                as :class:`Failure` (symmetric with guard/body failures).
            max_concurrency: Cap on in-flight per-item runners; items over
                it wait for a free slot. An ``int >= 1``, or
                ``(items, ctx) -> int`` (may be async) computed once when the
                map starts, with the resolved items — e.g. from the budget
                left in ``ctx.cost``. Not saved: a resumed map computes it
                again; not called when no item is left to run. Omitted →
                unbounded (all items dispatch immediately as one
                ``asyncio.gather``). Load-bearing for callers that need to
                respect an external rate limit (LLM requests, downstream
                service quota) or a budget.
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
                merges interleave against the shared parent state; an async
                merge that awaits between reading and writing ``parent_state``
                can lose updates — sync callbacks are preemption-safe. Callers
                wanting a single sequential fold should use ``aggregate``
                (which runs once after every item completes) instead.
                Requires ``state``.
            name: Stable label for the node's id (see :meth:`iterate`).

        Cancellation propagates unconditionally regardless of ``strict``.
        Sibling items keep running when one fails; the wasted work is the
        trade-off for a simple ordering guarantee.

        Returns ``self`` for chaining.

        Raises:
            TypeError: ``items=`` with a list of members.
            ValueError: An empty list of members.
        """
        _require_state_for_merge(state, merge, ".map")
        _check_node_name(name, ".map")
        if max_concurrency is not None and not callable(max_concurrency):
            check_concurrency(max_concurrency, ".map(max_concurrency=)")
        bodies, keys = _map_bodies(body, items, self._lg, type(self))
        node = _Node(
            target=_Map(
                bodies=bodies,
                member_keys=keys,
                items=items,
                aggregate=aggregate,
                strict=strict,
                state_fn=state,
                merge_fn=merge,
                max_concurrency=max_concurrency,
                state_factory=state_factory,
                name=name,
            ),
            rescue=rescue,
            after=after,
        )
        self._nodes.append(node)
        return self

    @_builder
    def guard(self, fn: GuardFn) -> Self:
        """Attach a per-item skip predicate to the preceding :meth:`map` node.

        ``fn(item, ctx) -> bool`` (sync or async) runs after per-item state
        projection and before the map body. A falsy verdict short-circuits
        the item: the body never runs, no per-item merge fires, and the
        result slot is filled with a :class:`Skipped` sentinel so the
        result list stays 1:1 with the input items.

        Only valid on a map node. Chainable form only — there is no
        ``guard=`` kwarg on :meth:`map` (keeps the signature narrow;
        the semantics only make sense for map).

        Raises:
            TypeError: The preceding node is missing, isn't a map, or
                already has a guard attached.
        """
        target = self._require_map_tail(".guard()")
        if target.guard is not None:
            raise TypeError(".guard() already set on the preceding map node")
        target.guard = fn
        return self

    @_builder
    def on_error(self, fn: OnErrorFn) -> Self:
        """Attach a per-item error hook to the preceding :meth:`map` node.

        ``fn(exception, item, ctx) -> None`` (sync or async) fires whenever
        an item's body raises a non-cancellation exception. The hook runs
        under both ``strict`` modes: in ``strict=True`` before the
        exception propagates, in ``strict=False`` before the item is
        replaced by :class:`Failure`. Return value is ignored; the hook is
        for side-effect narration (logging, tracing), not control flow.

        A hook exception is logged and swallowed so the original per-item
        exception is never masked. Only valid on a map node. Chainable
        form only — there is no ``on_error=`` kwarg on :meth:`map`.

        Raises:
            TypeError: The preceding node is missing, isn't a map, or
                already has an on_error attached.
        """
        target = self._require_map_tail(".on_error()")
        if target.on_error is not None:
            raise TypeError(".on_error() already set on the preceding map node")
        target.on_error = fn
        return self

    @_builder
    def on_item_complete(self, fn: OnItemCompleteFn) -> Self:
        """Attach a per-item completion hook to the preceding :meth:`map` node.

        ``fn(item, outcome, ctx) -> None`` (sync or async) fires once per
        item at its terminal state — for observability (progress bars,
        streaming, adaptive throttling), not control flow.

        ``outcome`` is the value that lands in the map's result list:

        - the body's return value on success (hook fires after per-item
          merge; ``ctx.state`` is the child state, and the merged parent
          is reachable via ``ctx.state.root()`` when projection is used);
        - a :class:`Failure` under both ``strict`` modes when the body
          raises a non-cancellation exception (in ``strict=True`` the
          hook fires before the exception propagates);
        - a :class:`Skipped` when the guard predicate returns falsy.

        A halted map raises :class:`Interrupted` and does not fire the hook
        for items stopped by the halt.

        Cancellation is unconditional and does not fire the hook. A hook
        exception is logged and swallowed so the item's outcome is never
        masked — matches :meth:`on_error`.

        During checkpoint/resume, the hook fires for every item processed
        in the resumed run, including fresh items not part of the checkpoint.

        Only valid on a map node. Chainable form only.

        Raises:
            TypeError: The preceding node is missing, isn't a map, or
                already has an ``on_item_complete`` attached.
        """
        target = self._require_map_tail(".on_item_complete()")
        if target.on_item_complete is not None:
            raise TypeError(".on_item_complete() already set on the preceding map node")
        target.on_item_complete = fn
        return self

    @_builder
    def with_shortcut(self, signal: str, *, drain: bool = False) -> Self:
        """Let the run's ``signal`` cut this flow short: it fast-forwards to its end.

        ``signal`` names one of the run's signals (:meth:`FlowRunner.start`
        ``signals=``). While it is set, this flow and every flow under it
        fast-forward, skipping what they can:

        - a chain starts no new step and ends with its last result (a
          chain that has not started passes its input through), except a
          step appended with :meth:`conclude`, which still runs;
        - an iterate starts no new pass and returns its carried value; a
          map starts no new item (those are :class:`Skipped`);
        - a Loop turn in flight is aborted, and its call returns the
          result SAIA paused it with.

        What is already running is neither stopped nor run again: it
        finishes with what it has (``ctx.fast_forward`` tells a step to
        return early). The step after this flow then runs as usual. A
        signal set before this flow starts does nothing until it does —
        the flow then ends at once — and a signal set after it ended does
        nothing here.

        With ``drain=True`` the region drains instead: it starts nothing
        new in the same way, but a map item that had started runs to its
        end as if the region were not cut — its chain, the flows under it
        and their maps, which may start their own items. A Loop turn in
        such an item is not aborted by the signal (the run's halt still
        pauses it), and ``ctx.draining`` tells its steps to wrap up. The
        map item is the unit of work: a chain step or an iterate pass in
        flight outside started items is cut short as above, so a region
        without a map drains like it fast-forwards. A region on another
        signal around the item still cuts it short.

        A cut is not a halt: it writes no checkpoint, and ``ctx.halt``
        stays unset. The run's halt still stops everything; a halt while
        this flow fast-forwards or drains is recorded with the signal, and
        resume goes on as before: items saved as running count as started.

        Checked at run start: ``signal`` is declared, and no flow under
        this one declares a shortcut on it too (regions on different
        signals may nest).

        Raises:
            ValueError: ``signal`` is not a non-empty str.
            TypeError: ``drain`` is not a bool.

        Returns ``self`` for chaining.
        """
        self._shortcut = check_shortcut(signal, drain)
        return self

    @_builder
    def with_budget(self, budget: float) -> Self:
        """Give each run of this flow its own budget: a child tracker with that limit.

        Each run runs on a child of the enclosing tracker (the run's own,
        :meth:`FlowRunner.start` ``cost=``, or an enclosing budget's) with
        ``budget`` as its limit — cost rolls up and every budget on the
        chain is tracked. A map body runs once per item, an iterate body
        once per pass, a subflow once per ``.call``. Crossing the budget,
        or an enclosing one, latches ``exceeded`` and ``urgent_wrapup`` for
        the agent to read through ``ctx.cost``; gent does not stop the run
        — the agent decides how to keep to its budget. A budget needs a
        run started with ``cost=``: :meth:`FlowRunner.start` raises
        otherwise.

        The child's spend is in every checkpoint taken while the run is in
        progress, and restored when it resumes. ``with_resource(COST,
        budget=budget)``: a per-run child of the
        :data:`~llm_gent.flow.COST` resource (see
        :mod:`llm_gent.flow.resource.cost`).

        Returns ``self`` for chaining.

        Raises:
            TypeError: ``budget`` is not a number.
            ValueError: ``budget`` is not finite and > 0.
        """
        return self.with_resource(COST, budget=check_budget(budget))

    @_builder
    def with_resource(
        self, key: ResourceKey[R], value: R | None = None, /, **child_args: Any
    ) -> Self:
        """Give this flow a resource under ``key``, reachable as ``ctx.resource(key)``.

        - ``with_resource(key, value)`` — every run of this flow, and every
          flow below it, runs with ``value``;
        - ``with_resource(key, **child_args)`` — each run of this flow (a
          map item, an iterate pass, a ``.call``) runs with
          ``resource.child(**child_args)`` of the enclosing ``key``
          resource;
        - ``with_resource(key, value, **child_args)`` — each run runs with
          a child of ``value``.

        ``value`` is of ``key``'s type, a :class:`~llm_gent.flow.Resource`;
        with child arguments, the resource the children come from has the
        optional ``child()``. Its ``snapshot()`` is in the run's checkpoints and handed back to its
        ``restore()`` on resume — the top-level flow's through the
        completion commit, so a later run continues from there; a per-run
        child's while its run is in progress (see
        :mod:`llm_gent.flow.resource._runtime`). Calls for the same key combine:
        a value replaces the resource, child arguments replace the per-run
        request.

        Returns ``self`` for chaining.

        Raises:
            TypeError: ``key`` is not a :class:`~llm_gent.flow.ResourceKey`;
                ``value`` does not implement ``snapshot()`` and
                ``restore(data)``; child arguments for a ``value`` without
                ``child()``. Type checkers report the first two, and a
                ``value`` not of ``key``'s type, statically.
            ValueError: Neither a value nor child arguments; a cost tracker
                as ``value`` under :data:`~llm_gent.flow.COST` (the run's
                tracker is :meth:`FlowRunner.start` ``cost=``).
        """
        check_resource(key, value, child_args)
        check_not_a_tracker(key, value)
        if value is not None:
            self._resources[key] = value
        if child_args:
            self._resource_children[key] = dict(child_args)
        return self

    @_builder
    def with_checkpointer(self, name: str | None = None) -> Self:
        """Declare that saves inside this flow write commits.

        ``ctx.checkpoint()`` and the checkpoint policy write commits in the
        flow and its subtree; in a part of the run with no checkpointer on
        it or above it they write nothing. Declared on any flow — the top
        level or a subflow, a map body, an iterate body — the commits go to
        the run's history (:meth:`FlowRunner.start` ``history=``) and hold the whole
        run; a checkpointer in a run without a history makes :meth:`FlowRunner.start`
        raise.

        A save belongs to the innermost checkpointer enclosing the step
        that saves. With ``name``, each such save also moves the tag
        ``tags/<name>``: ``start=Resume(name)`` goes back to the latest
        checkpoint that part took. Tag names are repo-global, one namespace
        with ``ctx.checkpoint(name)``; two flows with the same name share
        one tag.

        Returns ``self`` for chaining.

        Raises:
            ValueError: ``name`` cannot name a checkpoint (see
                :func:`~llm_gent.flow.checkpoint.checkpoint_tag`).
        """
        if name is not None:
            checkpoint_tag(name)
        self._checkpointer = Checkpointer(name)
        return self

    def root_hash(self) -> str:
        """Structure hash of this flow's composition tree.

        Recorded as ``flow_root_hash`` on every commit this flow writes.
        Equal hashes mean every chain step gets the same node id, so a
        checkpoint written by one flow can be resumed by the other.
        Covers step kinds, positions, targets (a named step: its name)
        and nested flows — not parameters such as ``max_iters`` or
        predicates. See :class:`~llm_gent.flow.structure.Structure`.
        """
        return Structure.of(self).hash

    @_builder
    def with_checkpoint_policy(
        self,
        policy: CheckpointPolicy | None = None,
        /,
        **kwargs: Any,
    ) -> Self:
        """Attach a :class:`CheckpointPolicy` governing implicit saves.

        The policy gates automatic composition-step saves (see
        :attr:`CheckpointPolicy.on_iterate`). It does NOT affect the
        halt-observation save (which always fires when a checkpointer
        is wired) nor the explicit ``ctx.checkpoint()`` trigger (which
        always fires when a verb invokes it).

        Accepts either a ready-made :class:`CheckpointPolicy` (for
        sharing across flows) or its constructor kwargs directly::

            flow.with_checkpoint_policy(on_iterate=True)
            flow.with_checkpoint_policy(CheckpointPolicy(on_iterate=True))
            flow.with_checkpoint_policy(shared_policy)

        Passing both is a programming error.

        A subflow inherits the outer runtime's policy unless it
        attaches its own; a local ``.with_checkpoint_policy`` on a
        subflow overrides the outer policy for that subtree.

        Returns ``self`` for chaining.
        """
        if policy is not None:
            if kwargs:
                raise ValueError(
                    "with_checkpoint_policy accepts either a CheckpointPolicy "
                    "instance or field kwargs, not both"
                )
            self._checkpoint_policy = policy
        else:
            self._checkpoint_policy = CheckpointPolicy(**kwargs)
        return self

    @_builder
    def with_hooks(self, *hooks: object, replace: bool = False) -> Self:
        """Attach hooks (:mod:`llm_gent.flow.hooks`) to this flow and every flow inside it.

        A hook is called at points of the run inside this flow: each step
        (``before_step`` / ``after_step``), each model call of a Loop
        (``after_llm_call``) and each save point (``after_checkpoint``). On
        the top-level flow it is also called for the run itself
        (``before_run`` / ``after_run``) and its halt and final-state
        commits. Hooks attached here run inside the ones this flow inherits:
        their ``before_*`` methods run after, their ``after_*`` methods
        before. With ``replace=True`` the inherited hooks, and the ones
        attached here before, are not called inside this flow. Calling it
        again without ``replace`` adds more hooks, after these.

        A hook's ``child(where)`` is asked for the hook to use each time a
        subflow, branch arm, iterate pass or map item starts under it.
        Hooks are never checkpointed: a resumed run calls the hooks the
        flows have then.

        Returns ``self`` for chaining.
        """
        self._hooks = hooks if replace else (*self._hooks, *hooks)
        self._hooks_replace = self._hooks_replace or replace
        return self

    def _require_map_tail(self, method: str) -> _Map:
        """Return the last node's target if it is a :class:`_Map`, else raise."""
        if not self._nodes:
            raise TypeError(f"{method} requires a preceding .map() node; none in chain")
        target = self._nodes[-1].target
        if not isinstance(target, _Map):
            raise TypeError(
                f"{method} applies to map nodes only; the preceding node is a "
                f"{type(target).__name__}"
            )
        return target

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _new_state(self) -> Any:
        """A fresh state payload: ``state_factory.new()``, else an empty ``dict``."""
        if self._state_factory is None:
            return {}
        return self._state_factory.new()
