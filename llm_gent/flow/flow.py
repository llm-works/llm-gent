# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow — verb registry, role-routed dispatch, and fluent composition graph.

A :class:`Flow` plays two roles that share one object:

1. **Runtime / registry.** Holds a :class:`SAIAFactory` (for turning roles
   into saia instances, cached per-role), a shared user-owned ``state``
   object, and an optional verb-by-name registry used by :meth:`dispatch`.

2. **Composition graph.** A sequence of nodes built up via the fluent
   methods :meth:`call` / :meth:`then` / :meth:`rescue` / :meth:`after` /
   :meth:`branch` / :meth:`iterate` / :meth:`map` and executed by
   :meth:`run`. A node's target is either a verb (any callable carrying a
   ``.role``), another :class:`Flow`, or a control-flow primitive (branch,
   iterate, map) whose bodies are themselves subflows. Subflows are
   recursively run against the same runtime, so saia caching is shared
   across the whole tree.

Both roles are optional. A top-level flow that only serves as a verb
registry never needs to call the fluent methods; a subflow that only exists
to structure composition never needs a factory of its own — it borrows from
the runtime it is executed under.

At application boundaries, prefer :class:`Factory` from
:mod:`.factory` — it captures the ambient ``lg`` and the app-wide
:class:`SAIAFactory` once so per-subsystem construction reads as
``f.create("grade").call(...)`` rather than repeating both at every site.
Direct :class:`Flow` construction is still supported and is what the
executor uses internally to materialize lambda-form subflows.

State is exposed on every :class:`Context` as a single :class:`State`
wrapper:

- ``ctx.state.data`` is the enclosing scope's payload — shared with the
  parent by reference by default. The ``state=`` / ``merge=`` kwargs on
  :meth:`Flow.call`, :meth:`Flow.iterate`, and :meth:`Flow.map` project an
  isolated child payload for the block they contain and (optionally) merge it
  back when the block completes successfully.
- ``ctx.state.root().data`` is the run-wide payload — the outermost
  :meth:`run` invocation's ``state=`` argument (default: fresh empty
  ``dict``). Every node in the tree reaches it via the same call regardless
  of nesting depth or per-scope projection.

Execution helpers (node dispatch, scoped-state projection/merge) live in
:mod:`._executor`; the private dataclasses and the :class:`Failure`
sentinel live in :mod:`.nodes`. This module owns the :class:`Flow` class
itself plus the builder-side helpers used by its fluent methods —
:func:`_validate_target`, :func:`_require_state_for_merge`, and the eager
Buildable materializer :func:`_materialize`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self, get_args

from appinfra.log import Logger

from ..core.cost import CostTracker
from ..core.traits import Registry as TraitRegistry
from ._chain import Chain
from ._checkpoint_ctx import CheckpointContext, check_one_repo
from ._halt_observer import HaltPoint, check_one_halt, is_run_halted
from ._resume import (
    Resume,
    apply_clean_exit_retention,
    commit_halt,
)
from ._shortcut import (
    Shortcut,
    ShortcutRun,
    check_shortcut,
    check_shortcuts,
    check_signal,
    run_shortcut,
    run_signals,
)
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
    CheckpointStore,
    ResumeMode,
    checkpoint_tag,
    is_commit_hash,
)
from .context import Context
from .factory import SAIAFactory
from .nodes import (
    HALTED,
    UNSET,
    AfterHook,
    AggregateFn,
    Checkpointer,
    GuardFn,
    Interrupted,
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
    _RunEnv,
)
from .resource import COST, NO_RESOURCES, R, ResourceKey, check_resource
from .resource._runtime import Resources, check_resources, run_resources
from .resource.cost import check_budget, check_budgets_have_a_tracker, check_cost_tracker
from .role import Role
from .state import State, StateFactory
from .state.snapshot import ScopePath, ScopeRegistry, Snapshot
from .structure import Structure


class Flow:
    """Verb registry + role-routed dispatch + fluent composition graph.

    A ``Flow`` used as the top-level runtime is constructed with a factory.
    Subflows used only for composition can be constructed without one — at
    :meth:`run` time they borrow the factory (and saia cache) of the flow
    that invoked them.

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
        saia_factory: SAIAFactory | None = None,
        state: Any = UNSET,
        traits: TraitRegistry | None = None,
        state_factory: StateFactory[Any] | None = None,
    ) -> None:
        """Initialize a flow.

        Prefer :class:`Factory` at application boundaries — it captures
        the ambient ``lg`` and the app-wide :class:`SAIAFactory` once so
        Flow-per-subsystem construction doesn't repeat them. Constructing
        :class:`Flow` directly is still supported; the executor uses it
        internally to materialize subflows built with the fluent lambda
        form, and it remains valid for lower-level tests.

        Args:
            lg: Logger instance for tracing execution.
            name: Optional identifier — used in error messages and traces.
                Also lets a flow serve as a named node inside a parent chain.
            saia_factory: A :class:`SAIAFactory` that builds role-bound saia
                instances. Required only when role-bound code accesses
                ``ctx.saia`` — verbs that route LLM calls through their own
                configuration can run without one. A subflow borrows the
                factory from the runtime it executes under.
            state: User-owned shared state object. Verbs read and (typically)
                mutate it in place. Opaque to the flow; may be overridden per
                :meth:`run` invocation.
            traits: Optional trait registry surfaced on every dispatched
                :class:`Context` as ``ctx.traits``. When ``None``, verbs see
                ``ctx.traits is None``. A subflow inherits the outer
                runtime's registry (like the saia cache) via the same
                internal handoff, so mounting on the top-level flow is
                enough to reach every nested dispatch.
            state_factory: A :class:`StateFactory` the framework calls on
                :meth:`run` ``resume="latest"`` to reconstruct ``ctx.state.data``
                from the loaded checkpoint: ``state_factory.restore(...)``.
                For state that carries no runtime handles wrap the type in
                :class:`TypeStateFactory`; for state that binds a Logger /
                storage / connection at restore, implement
                :class:`StateFactory` directly. Only consulted when
                :meth:`with_checkpointer` is wired. ``None`` (default)
                treats ``ctx.state.data`` as a plain dict that round-trips
                through the checkpointer as-is.
        """
        self._lg = lg
        self._name = name
        self._saia_factory = saia_factory
        self._state = state
        self._traits = traits
        self._state_factory = state_factory
        self._halt_event: asyncio.Event | None = None
        self._signals: dict[str, asyncio.Event] = {}
        self._shortcut: Shortcut | None = None
        self._resources: dict[ResourceKey[Any], Any] = {}
        self._resource_children: dict[ResourceKey[Any], dict[str, Any]] = {}
        self._checkpoint_ctx: CheckpointContext | None = None
        self._checkpointer: Checkpointer | None = None
        self._verbs: dict[str, Any] = {}
        self._saia_by_role: dict[Role, Any] = {}
        self._nodes: list[_Node] = []
        self._halt_at: HaltPoint | None = None
        self._checkpoint_policy: CheckpointPolicy | None = None
        self._scopes: ScopeRegistry = ScopeRegistry()

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

    @property
    def traits(self) -> TraitRegistry | None:
        """The trait registry this flow was constructed with, or ``None``."""
        return self._traits

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

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
        **kwargs: Any,
    ) -> Any:
        """Dispatch a registered verb by name, awaiting its result.

        The verb receives a fresh :class:`Context` as its first argument,
        followed by ``*args`` / ``**kwargs`` from the caller. ``dispatch`` is
        the low-level entrypoint for verbs that invoke sibling verbs
        directly.

        ``scope_state=`` wins over the flow's construction state — pass
        ``scope_state=ctx.state`` from an in-flight verb to hand the dispatched sibling the live
        scope payload, not the flow's construction default. Omitting
        ``scope_state=`` (or passing ``UNSET``) falls back to ``self._state``,
        defaulting to a fresh empty ``dict`` when none was supplied at
        construction. A ``State`` instance passes through as-is; any other
        value is wrapped with this flow's ``state_factory``.

        Pass ``halt=ctx.halt`` and ``resources=ctx.resources`` from an
        in-flight verb to propagate its effective ambients to the
        dispatched sibling; omitting either (or passing ``UNSET``) defaults
        to this flow's ``.with_halt()`` binding and the resources it
        declares (no per-run children such as ``.with_budget()``'s: there
        is no run). ``cost=`` sets the sibling's ``ctx.cost`` on top of
        those — ``None`` for none.

        Pass ``extra=ctx.extra`` from an in-flight verb to propagate the
        caller-supplied opaque dict to the dispatched sibling. Omitting
        (or passing ``UNSET``) yields a fresh empty dict at the sibling —
        ``dispatch`` has no flow-level ``.with_extra()`` fallback.
        """
        if name not in self._verbs:
            raise KeyError(f"no verb registered under name {name!r}")
        verb = self._verbs[name]
        role_name = verb.role.name if verb.role is not None else None
        self._lg.debug("dispatching verb", extra={"verb": name, "role": role_name})
        payload = (
            (self._state if self._state is not UNSET else {})
            if scope_state is UNSET
            else scope_state
        )
        effective_halt = self._halt_event if halt is UNSET else halt
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
            traits=self._traits,
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
        passed to :meth:`run`; each subsequent node receives the previous
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
        :meth:`iterate`) and is how :meth:`with_shortcut` (``to=``) refers
        to it.

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
        iteration is guaranteed to terminate. An ambient :meth:`with_halt`
        event, if set, is checked between iterations and terminates the
        block gracefully once fired.

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
                propagates out of :meth:`run`. ``False`` → each failing item is
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

    def with_halt(self, event: asyncio.Event) -> Self:
        """Set the run's halt: the app sets ``event`` to pause the whole run.

        Threads ``event`` through the execution environment as ``ctx.halt``,
        available to any verb that wants to observe it. :meth:`map` and
        :meth:`iterate` also check the event at their natural boundaries:
        map raises :class:`Interrupted` when the halt stops an item before
        its body ran; iterate exits the loop between passes. In-flight
        bodies are not interrupted — a verb that needs mid-request
        cancellation should read ``ctx.halt`` itself. A halted
        :meth:`~Flow.run` returns :data:`HALTED`; state is in the halt checkpoint.

        A run has one halt, set on its top-level flow; every subflow
        observes it. :meth:`run` raises when a nested flow sets a different
        event.

        Returns ``self`` for chaining.
        """
        self._halt_event = event
        return self

    def with_signal(self, name: str, event: asyncio.Event) -> Self:
        """Declare the run's signal ``name``: an event the app sets, for :meth:`with_shortcut`.

        Signals belong to the run, like its halt: they are declared on the
        top-level flow, and :meth:`run` raises when a nested flow declares
        one. Which signals are set is recorded in every checkpoint the run
        takes, and resume sets them again before the first step; a run that
        finishes records none.

        Returns ``self`` for chaining.
        """
        name, event = check_signal(name, event)
        self._signals[name] = event
        return self

    def with_shortcut(self, signal: str, to: str | None = None) -> Self:
        """On the run's ``signal``, stop exploring and continue at step ``to`` with what there is.

        ``signal`` names a signal the top-level flow declares with
        :meth:`with_signal`. When the app sets it, this flow stops as the
        run's halt stops it — every part at its next boundary, a Loop turn
        paused — and then continues at once from where it stopped, in
        shortcut mode, until it reaches ``to``:

        - the step that was running runs again from where it stopped; a
          Loop turn held there is not continued: the call returns the
          result SAIA paused it with;
        - an iterate in this flow's chain starts no new pass and returns
          its carried value; a map in it starts no new item (those are
          :class:`Skipped`) and its started items finish;
        - the chain then continues at ``to`` — the ``name=`` of one of its
          steps — skipping the steps before it (``to`` gets the last
          completed result), or ends when ``to`` is ``None``.

        Flows under this one run normally unless they declare a shortcut
        of their own; one signal can drive several. Steps at and after
        ``to`` run normally: a signal set once the chain is there does
        nothing. A run of this flow that starts while the signal is set (a
        later step, the next iterate pass, a map item) starts in shortcut
        mode. The run's halt still stops everything; a halt during a
        shortcut is recorded with it, and resume continues the shortcut.
        A signal set inside a step, without awaiting, stops the flow at
        that step's boundary too.

        Under the shortcut, ``ctx.halt`` is this flow's stop (set by the
        run's halt and by the signal); ``ctx.run_halt`` is the run's halt,
        for a step that pauses the whole run or tells a halt from a cut.

        Checked at run start: ``signal`` is declared, and ``to`` names
        exactly one step of this flow's chain.

        Returns ``self`` for chaining.
        """
        self._shortcut = check_shortcut(signal, to)
        return self

    def with_cost_tracker(self, tracker: CostTracker) -> Self:
        """Run every run of this flow on ``tracker``, reachable as ``ctx.cost``.

        Verbs record LLM and operation costs against it through
        ``ctx.cost``. Its ``snapshot()`` is in the run's checkpoints — on
        the top-level flow, in every commit including the completion
        commit — and handed back to its ``restore()`` on resume: the total
        over the whole history for a plain :class:`CostTracker`, whatever
        a subclass makes of it otherwise. ``with_resource(COST, tracker)``:
        the tracker is the flow's :data:`~llm_gent.flow.COST` resource (see
        :mod:`llm_gent.flow.resource.cost`). A hard
        stop is opt-in on the tracker side: pass the run's halt event to
        both :meth:`CostTracker.__init__` (``halt=``) and :meth:`with_halt`,
        and the tracker sets it on the first cross into ``exceeded``,
        pausing the run. Without a tracker of its own a flow shares the
        enclosing one.

        Returns ``self`` for chaining.

        Raises:
            TypeError: ``tracker`` is not a :class:`CostTracker`.
        """
        return self.with_resource(COST, check_cost_tracker(tracker))

    def with_budget(self, budget: float) -> Self:
        """Give each run of this flow its own budget: a child tracker with that limit.

        Each run runs on a child of the flow's tracker (its own, else the
        enclosing flow's) with ``budget`` as its limit — cost rolls up and
        every budget on the chain is tracked. A map body runs once per
        item, an iterate body once per pass, a subflow once per ``.call``.
        Crossing the budget latches the child's ``exceeded`` and
        ``urgent_wrapup`` for the agent to read through ``ctx.cost``; gent
        does not stop the run — the agent decides how to keep to its
        budget. A budget needs a tracker on this flow or an enclosing one:
        :meth:`run` raises otherwise.

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
            ValueError: Neither a value nor child arguments.
        """
        check_resource(key, value, child_args)
        if value is not None:
            self._resources[key] = value
        if child_args:
            self._resource_children[key] = dict(child_args)
        return self

    def with_checkpoint_store(self, store: CheckpointStore, client_flow_id: str) -> Self:
        """Set the run's repo: a :class:`CheckpointStore` and the agent-owned ``client_flow_id``.

        A run has one repo, set on its top-level flow: every commit the run
        writes — wherever in the flow tree it is taken — holds the whole
        run and goes here. A flow inside another run cannot set one
        (:meth:`run` raises). With a store, the run writes its halt
        checkpoint and, on a clean exit, its final state; saves inside the
        run (``ctx.checkpoint()``, the checkpoint policy) need a
        :meth:`with_checkpointer` on the saving flow or above it. On
        :meth:`run` ``resume=...`` the repo is checked out at start,
        restoring the run's scopes and cursors. On fully successful
        :meth:`run` completion the framework calls
        :meth:`CheckpointStore.gc_history` when the store's ``retention``
        is ``"gc_on_success"``; the default ``"retain"`` keeps the history
        for audit. Cancellation, halt exits, and unhandled exceptions
        preserve the history regardless of retention so a subsequent
        resume can pick up.

        Both arguments bind together — the ``client_flow_id`` scopes every
        save/load/delete call and identifies the resumable history. It
        is agent-owned: the framework never assigns one automatically.

        Resume semantics:

        - **State** hydrates from the commit's snapshot (via
          ``state_factory.restore`` if a ``state_factory`` is bound, else
          passthrough for plain dicts).
        - **Iteration counter** is restored with the iterate's cursor:
          ``max_iters`` is a cumulative bound across resumes — saving in
          pass N and resuming with ``max_iters=M`` runs passes N to M-1.
          A counter that already meets the bound exits without running
          the body.
        - **Ambients** (halt, cost tracker, saia, traits, logger,
          checkpointer itself) are never serialized — they reattach
          from the current runtime, so a resumed run gets fresh
          handles under whichever ``.with_halt`` / ``.with_cost_tracker`` /
          ``.with_traits`` were wired at resume time.
        - **Deadline** is not restored — the wall clock starts fresh
          each run.

        Returns ``self`` for chaining.
        """
        self._checkpoint_ctx = CheckpointContext(store, client_flow_id, lambda: Structure.of(self))
        return self

    def with_checkpointer(self, name: str | None = None) -> Self:
        """Declare that saves inside this flow write commits.

        ``ctx.checkpoint()`` and the checkpoint policy write commits in the
        flow and its subtree; in a part of the run with no checkpointer on
        it or above it they write nothing. Declared on any flow — the top
        level or a subflow, a map body, an iterate body — the commits go to
        the run's repo (:meth:`with_checkpoint_store` on the top-level
        flow) and hold the whole run; a checkpointer in a run without a
        store makes :meth:`run` raise.

        A save belongs to the innermost checkpointer enclosing the step
        that saves. With ``name``, each such save also moves the tag
        ``tags/<name>``: ``run(resume=name)`` goes back to the latest
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

    def _begin_checkpoint_run(self) -> None:
        """Reset the run's repo's run-scoped caches."""
        if self._checkpoint_ctx is not None:
            self._checkpoint_ctx.begin_run()

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
    # Execution
    # -------------------------------------------------------------------------

    async def run(
        self,
        *args: Any,
        state: Any = UNSET,
        resume: ResumeMode | str = "off",
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the composition graph as the top-level runtime.

        The first node receives ``*args`` / ``**kwargs``. Each subsequent
        node receives one positional — the previous node's result, optionally
        transformed by its ``project`` callable.

        Cancellation is honored end-to-end: :class:`asyncio.CancelledError`
        raised by any node (verb or subflow) propagates out of :meth:`run`
        unchanged, past any ``rescue`` policy.

        Args:
            *args: Positional inputs to the first node.
            state: The payload to expose on ``ctx.state`` for this run.
                Framework wraps it as :class:`State` (``ctx.state.data``
                reaches the payload; ``ctx.state.root()`` reaches the
                outermost scope from any subflow). Omitted → the flow's
                construction ``state`` is used, else a fresh empty ``dict``.
                Explicitly passing ``None`` is honored (the payload becomes
                ``None`` and verbs must guard). ``state`` is a bound
                parameter — it is not forwarded to the first node; verbs
                needing it as a kwarg are rejected at :meth:`call` time.
            extra: Caller-supplied per-invocation opaque dict reachable
                via ``ctx.extra`` on every dispatch inside the run.
                Escape hatch for handles the framework does not type
                (tenant IDs, correlation IDs, request-scoped audit hooks,
                per-run callbacks). Framework does not inspect the
                contents and never persists them — ``extra`` never enters
                a checkpoint Blob, Tree, or node content hash. On resume
                the caller re-supplies at :meth:`run`; identity across
                resume is not preserved. ``None`` (default) yields a
                fresh empty dict at the verb.
            resume: How to start from the checkpointed history
                (:data:`~llm_gent.flow.checkpoint.ResumeMode`).
                ``"off"`` (default) runs from ``state`` as given.
                ``"latest"`` checks out the newest commit with state
                (skipping commits without state) and continues
                where it was: its root state replaces ``state``, every
                scope comes back, and every chain, iterate and branch that
                was running continues at its saved step, pass and arm, so
                only the step running at the checkpoint runs again; a Loop
                that paused mid-turn resumes that turn. Payloads are rebuilt
                via ``state_factory`` when bound, else used as plain dicts.
                On an empty history the run proceeds with ``state`` as
                given, appending to the same history; a corrupt history
                raises :class:`HistoryCorrupt`. A commit hash (as
                ``ctx.checkpoint()`` returns) or a checkpoint name (as
                ``ctx.checkpoint(name)`` took) selects one commit: the run
                checks it out the same way and its commits continue from
                there. Once the run commits, the commits after it leave
                the history's line and stay resumable by hash until
                :func:`~llm_gent.flow.collect_unreachable` deletes them; a
                run that fails before its first commit leaves ``HEAD``
                where it was.
                Resuming requires
                :meth:`with_checkpointer`. Bound parameter: not forwarded
                to the first node.
            **kwargs: Keyword inputs to the first node.

        Returns the last step's result, or :data:`HALTED` when the halt
        stopped the run before its end: a step that stops because of the
        halt either completes or raises :class:`Interrupted`, and the run
        then writes its halt checkpoint and returns. The halted run's state
        is in that checkpoint; ``resume="latest"`` continues from it. A halt
        set during the last step, which completed, does not stop the run: it
        finishes and returns that step's result (``None`` included).

        With a checkpointer wired, a fully successful run under
        ``retention="retain"`` commits its final state (tagged
        ``complete``); ``"gc_on_success"`` deletes the history instead. A
        run that raises (or is cancelled) writes no commit: its history's
        head stays its last save, which ``resume="latest"`` continues from.

        Raises:
            RuntimeError: The flow has no nodes to run, OR a resume mode
                was requested without :meth:`with_checkpointer` wired, OR
                a nested flow sets a halt other than this flow's or declares
                a signal, OR a shortcut's signal is not declared or its
                ``to`` names no step (or several) of its chain, OR
                a flow in the tree has ``with_budget(...)`` with no cost
                tracker on it or any flow enclosing it. Missing :class:`SAIAFactory` no longer raises at run
                start — the error surfaces at the first ``ctx.saia``
                access instead, so verbs that don't consume ``ctx.saia``
                can run under a factoryless flow.
            ValueError: ``resume`` is not a :data:`ResumeMode` value, a
                commit hash or a valid checkpoint name; names a commit or
                checkpoint the history does not have; or names a commit
                without state.
        """
        self._check_run_args(resume)
        self._begin_checkpoint_run()
        active_state, saved = await self._start_state(self._wrap_top_state(state), resume)
        self._scopes.begin(active_state, saved)
        self._halt_at = None
        try:
            with run_signals(self):
                result = await self._run_as_subflow(
                    *args,
                    state=active_state,
                    runtime=self,
                    parent_extra=extra,
                    **kwargs,
                )
        except Interrupted:
            # Everything has stopped, each part registered where it stopped.
            await commit_halt(self)
            return HALTED
        await apply_clean_exit_retention(self, active_state)
        return result

    def _check_run_args(self, resume: ResumeMode | str) -> None:
        """Reject an empty flow, a bad ``resume``, misplaced wiring, a bad shortcut, a lone cap.

        Runs before anything is read or written, so a misconfigured run
        leaves no new history behind.
        """
        if not self._nodes:
            raise RuntimeError(f"Flow {self._name!r} has no nodes to run")
        if resume not in get_args(ResumeMode) and not is_commit_hash(resume):
            try:
                checkpoint_tag(resume)
            except ValueError as e:
                raise ValueError(
                    f"resume must be one of {get_args(ResumeMode)}, a commit hash or a "
                    f"checkpoint name; got {resume!r} ({e})"
                ) from None
        if resume != "off" and self._checkpoint_ctx is None:
            label = self._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} was run with resume={resume!r} but has no checkpoint "
                f"store — call .with_checkpoint_store(store, client_flow_id) first"
            )
        check_one_repo(self)
        check_one_halt(self)
        check_shortcuts(self)
        check_budgets_have_a_tracker(self)
        check_resources(self)

    async def _start_state(
        self, fallback: State[Any], resume: ResumeMode | str
    ) -> tuple[State[Any], Snapshot | None]:
        """The run's initial state and the snapshot it continues from (``None`` for a fresh run)."""
        if resume == "off":
            return fallback, None
        if resume == "latest":
            return await Resume(self).checkout(fallback)
        return await Resume(self).checkout_at(resume)

    async def _run_as_subflow(
        self,
        *args: Any,
        state: State[Any],
        runtime: Flow,
        parent_halt: asyncio.Event | None = None,
        parent_resources: Resources = NO_RESOURCES,
        parent_checkpoint_ctx: CheckpointContext | None = None,
        parent_checkpointer: Checkpointer | None = None,
        parent_chain_context: str = "",
        parent_ancestor_chain: tuple[str, ...] = (),
        parent_extra: dict[str, Any] | None = None,
        parent_policy: CheckpointPolicy | None = None,
        parent_path: ScopePath = (),
        parent_shortcuts: tuple[ShortcutRun, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Internal entry: walk nodes with caller-supplied ``State`` and runtime.

        The executor's subflow helpers (``.call`` targeting a Flow, and the
        ``.branch`` / ``.iterate`` / ``.map`` bodies) invoke this directly so
        the outer runtime's factory and saia cache are shared with the
        subflow. State arrives pre-wrapped — top-level wrapping happens once
        in :meth:`run`.

        ``parent_checkpointer`` / ``parent_policy`` are the effective
        ambients from the calling scope — nested subflows fall back to them
        when they have no local ``.with_checkpointer()`` /
        ``.with_checkpoint_policy()``, preserving an intermediate layer's
        ambient through arbitrarily deep nesting. ``parent_halt`` is the
        halt the calling scope observes: the run's halt, or a shortcut's
        stop event. ``parent_checkpoint_ctx`` is the run's repo; repo and
        halt are set on the top-level flow only, so a nested run observes
        its parent's halt; a shortcut this flow declares
        (:func:`~llm_gent.flow._shortcut.run_shortcut`) adds its own stop on
        top. ``parent_shortcuts`` are the enclosing flows'.
        ``parent_resources`` are the calling scope's resources — the cost
        tracker among them; the run's own come from
        :func:`~llm_gent.flow.resource._runtime.run_resources`.

        ``parent_chain_context`` is the hash the executor uses to compute
        this Flow's chain-step node IDs (empty at run root; extended by
        :func:`_descend_context` at each subflow / arm / body boundary).
        ``parent_ancestor_chain`` is the tuple of ancestor ``_Node`` IDs
        from root down to the ``_Node`` whose descent entered this Flow;
        it grows by one on every recursion. ``parent_path`` is this Flow's
        snapshot path (see :attr:`_RunEnv.path`).
        """
        if not self._nodes:
            raise RuntimeError(f"Flow {self._name!r} has no nodes to run")
        scopes = runtime._scopes
        # The run's halt is on the top-level flow (check_one_halt).
        halt = self._halt_event if parent_halt is None else parent_halt
        async with run_shortcut(self, parent_path, scopes, halt, runtime._signals) as shortcut:
            with run_resources(self, parent_path, scopes, parent_resources) as resources:
                env = self._make_run_env(
                    runtime=runtime,
                    state=state,
                    halt=halt,
                    resources=resources,
                    parent_checkpoint_ctx=parent_checkpoint_ctx,
                    parent_checkpointer=parent_checkpointer,
                    parent_chain_context=parent_chain_context,
                    parent_ancestor_chain=parent_ancestor_chain,
                    parent_extra=parent_extra,
                    parent_policy=parent_policy,
                    parent_path=parent_path,
                    shortcut=shortcut,
                    parent_shortcuts=parent_shortcuts,
                )
                return await self._run_in(env, args, kwargs)

    async def _run_in(self, env: _RunEnv, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Walk this Flow under ``env``."""
        label = self._name or "<anonymous>"
        is_subflow = env.runtime is not self
        env.lg.debug(
            "starting flow run",
            extra={"flow": label, "nodes": len(self._nodes), "subflow": is_subflow},
        )
        result = await self._walk(env, args, kwargs)
        env.lg.debug("completed flow run", extra={"flow": label, "subflow": is_subflow})
        return result

    async def _walk(self, env: _RunEnv, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Walk this Flow's chain.

        A run its shortcut stopped walks again at once, in shortcut mode,
        from the positions it stopped at (restaged as a checkout's).
        """
        try:
            return await Chain(self, env).walk(args, kwargs)
        except Interrupted:
            if env.shortcut is None or not env.shortcut.take_over(is_run_halted(env)):
                raise
        env.scopes.restage_under(env.path)
        return await self._walk(env, args, kwargs)

    def _make_run_env(
        self,
        *,
        runtime: Flow,
        state: State[Any],
        halt: asyncio.Event | None,
        resources: Resources,
        parent_checkpoint_ctx: CheckpointContext | None,
        parent_chain_context: str = "",
        parent_ancestor_chain: tuple[str, ...] = (),
        parent_extra: dict[str, Any] | None = None,
        parent_policy: CheckpointPolicy | None = None,
        parent_path: ScopePath = (),
        parent_checkpointer: Checkpointer | None = None,
        shortcut: ShortcutRun | None = None,
        parent_shortcuts: tuple[ShortcutRun, ...] = (),
    ) -> _RunEnv:
        """Resolve local-override-wins ambients and build the per-run environment.

        Local ``.with_checkpointer`` / ``.with_checkpoint_policy`` wins over
        the caller's parent ambients; unset locals fall back to the parent
        so an intermediate layer's ambient survives arbitrarily deep
        nesting. ``halt`` and ``resources`` come resolved; this flow's
        ``shortcut``, when it declares one, puts its stop in place of the
        halt. The repo is the top-level flow's (:func:`check_one_repo`
        keeps nested flows from setting one).

        ``parent_chain_context`` and ``parent_ancestor_chain`` are copied
        verbatim: the descent sites in :mod:`._executor` are the ones
        that extend them when recursing into a subflow / arm / body.
        """
        policy = self._checkpoint_policy or parent_policy or CheckpointPolicy()
        checkpoint_ctx = self._checkpoint_ctx or parent_checkpoint_ctx
        return _RunEnv(
            runtime=runtime,
            state=state,
            lg=runtime._lg,
            halt=shortcut.stop if shortcut is not None else halt,
            checkpoint_ctx=checkpoint_ctx,
            checkpointer=self._checkpointer or parent_checkpointer,
            chain_context=parent_chain_context,
            ancestor_chain=parent_ancestor_chain,
            extra=parent_extra if parent_extra is not None else {},
            policy=policy,
            path=parent_path,
            shortcut=shortcut,
            shortcuts=parent_shortcuts + ((shortcut,) if shortcut is not None else ()),
            resources=resources,
        )

    def _wrap_top_state(self, state: Any) -> State[Any]:
        """Wrap a top-level ``run(state=...)`` payload as :class:`State`.

        Resolves the ``state=UNSET`` sentinel to the flow's construction
        default (or a fresh empty ``dict`` when none was supplied). Explicit
        ``state=None`` is honored — the payload becomes ``None``. Already-
        wrapped :class:`State` inputs are returned unchanged so a caller can
        thread a run-wide state instance across multiple :meth:`run` calls.
        """
        payload = (self._state if self._state is not UNSET else {}) if state is UNSET else state
        if isinstance(payload, State):
            return payload
        return State(data=payload, _factory=self._state_factory)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _saia_for(self, role: Role) -> Any:
        """Return a cached saia for ``role``, building it on first request."""
        cached = self._saia_by_role.get(role)
        if cached is not None:
            return cached
        if self._saia_factory is None:
            label = self._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} has no SAIAFactory — saia_factory= was not supplied at "
                f"construction (needed to build saia for role {role.name!r})"
            )
        built = self._saia_factory.build(role)
        self._saia_by_role[role] = built
        return built
