# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow — verb registry, role-routed dispatch, and fluent composition graph.

A :class:`Flow` plays two roles that share one object:

1. **Runtime / registry.** Holds a :class:`SAIAFactory` (for turning roles
   into saia instances, cached per-role), a shared user-owned ``state``
   object, and an optional verb-by-name registry used by :meth:`dispatch`
   and by :class:`Panel`.

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

At application boundaries, prefer :class:`FlowFactory` from
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
import hashlib
import inspect
import json
from datetime import UTC, datetime
from typing import Any

from appinfra.log import Logger

from ..core.budget import Tracker
from ..core.traits import Registry as TraitRegistry
from ._executor import _build_ctx, _execute_node, _save_halt_checkpoint, _step_inputs
from .checkpoint import CheckpointPolicy, CheckpointStore, maybe_await
from .context import Context
from .factory import SAIAFactory
from .nodes import (
    UNSET,
    AfterHook,
    AggregateFn,
    GuardFn,
    ItemsFn,
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
    _ResumeReplay,
    _RunEnv,
)
from .role import Role
from .state import State, StateFactory
from .state.cas import Commit, CommitMeta, ProducedBy, Tree


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
        saia_factory: SAIAFactory | None = None,
        state: Any = UNSET,
        traits: TraitRegistry | None = None,
        state_factory: StateFactory[Any] | None = None,
    ) -> None:
        """Initialize a flow.

        Prefer :class:`FlowFactory` at application boundaries — it captures
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
                :meth:`run` ``resume=True`` to reconstruct ``ctx.state.data``
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
        self._budget_tracker: Tracker | None = None
        self._checkpointer: CheckpointStore | None = None
        self._client_flow_id: str | None = None
        self._verbs: dict[str, Any] = {}
        self._saia_by_role: dict[Role, Any] = {}
        self._nodes: list[_Node] = []
        self._replay_consumed: bool = False
        self._halt_saved: bool = False
        self._checkpoint_policy: CheckpointPolicy | None = None
        self._pending_saia_turn_bytes: dict[str, bytes] = {}
        self._pending_saia_turn_ancestors: dict[str, tuple[str, ...]] = {}
        self._resume_saia_turn_bytes: dict[str, bytes] = {}

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
        budget: Any = UNSET,
        scope_state: Any = UNSET,
        extra: Any = UNSET,
        **kwargs: Any,
    ) -> Any:
        """Dispatch a registered verb by name, awaiting its result.

        The verb receives a fresh :class:`Context` as its first argument,
        followed by ``*args`` / ``**kwargs`` from the caller. ``dispatch`` is
        the low-level entrypoint used by :class:`Panel` and by verbs that
        invoke sibling verbs directly.

        ``scope_state=`` wins over the flow's construction state — pass
        ``scope_state=ctx.state`` from an in-flight verb (or a Panel, which
        does this automatically) to hand the dispatched sibling the live
        scope payload, not the flow's construction default. Omitting
        ``scope_state=`` (or passing ``UNSET``) falls back to ``self._state``,
        defaulting to a fresh empty ``dict`` when none was supplied at
        construction. A ``State`` instance passes through as-is; any other
        value is wrapped with this flow's ``state_factory``.

        Pass ``halt=ctx.halt`` and ``budget=ctx.budget`` from an in-flight
        verb to propagate its effective ambients to the dispatched sibling;
        omitting either (or passing ``UNSET``) defaults to this flow's
        ``.with_halt()`` / ``.with_budget()`` binding if any.

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
        effective_budget = self._budget_tracker if budget is UNSET else budget
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
            budget=effective_budget,
            extra=effective_extra,
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
        state_factory: StateFactory[Any] | None = None,
    ) -> Flow:
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
                state_factory=state_factory,
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
            state_factory=state_factory,
        )

    def rescue(self, policy: RescuePolicy) -> Flow:
        """Attach a failure policy to the most recently appended node.

        The policy runs when the node raises anything other than
        :class:`asyncio.CancelledError` (cancellation is never rescued).
        Signature: ``(exception, pending_input, ctx) -> fallback`` — may be async.
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
    ) -> Flow:
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
        body_flow = _materialize(body, self._lg, "iterate.body")
        node = _Node(
            target=_Iterate(
                body=body_flow,
                until=until,
                max_iters=max_iters,
                deadline=deadline,
                state_fn=state,
                merge_fn=merge,
                state_factory=state_factory,
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
        max_concurrency: int | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
        state: StateProject | None = None,
        merge: StateMerge | None = None,
        state_factory: StateFactory[Any] | None = None,
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
                the aggregator can partition successes from failures. Under
                ``strict=False`` a state-projection failure is also wrapped
                as :class:`Failure` (symmetric with guard/body failures).
            max_concurrency: Cap on in-flight per-item runners. Must be
                ``>= 1`` when set. Omitted → unbounded (all items dispatch
                immediately as one ``asyncio.gather``). Load-bearing for
                callers that need to respect an external rate limit (LLM
                requests, downstream service quota).
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

        Cancellation propagates unconditionally regardless of ``strict``.
        Sibling items keep running when one fails; the wasted work is the
        trade-off for a simple ordering guarantee.

        Returns ``self`` for chaining.
        """
        _require_state_for_merge(state, merge, ".map")
        if max_concurrency is not None and (
            type(max_concurrency) is not int or max_concurrency < 1
        ):
            raise ValueError(f".map(max_concurrency=) must be an int >= 1; got {max_concurrency}")
        body_flow = _materialize(body, self._lg, "map.body")
        node = _Node(
            target=_Map(
                body=body_flow,
                items=items,
                aggregate=aggregate,
                strict=strict,
                state_fn=state,
                merge_fn=merge,
                max_concurrency=max_concurrency,
                state_factory=state_factory,
            ),
            rescue=rescue,
            after=after,
        )
        self._nodes.append(node)
        return self

    def guard(self, fn: GuardFn) -> Flow:
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

    def on_error(self, fn: OnErrorFn) -> Flow:
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

    def on_item_complete(self, fn: OnItemCompleteFn) -> Flow:
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
        - a :class:`Skipped` when the guard predicate returns falsy or
          when an ambient halt short-circuited the item before its body ran.

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

    def with_halt(self, event: asyncio.Event) -> Flow:
        """Attach an ambient halt signal that reaches every node in this Flow.

        Threads ``event`` through the execution environment as ``ctx.halt``,
        available to any verb that wants to observe it. :meth:`map` and
        :meth:`iterate` also check the event at their natural boundaries:
        map short-circuits any per-item runner that hasn't yet passed its
        halt check to :class:`Skipped`; iterate exits the loop between
        iterations. In-flight bodies are not interrupted — a verb that
        needs mid-request cancellation should read ``ctx.halt`` itself.

        A subflow inherits the outer runtime's halt event automatically;
        calling ``.with_halt`` on a subflow overrides the ambient event for
        that subtree.

        Returns ``self`` for chaining.
        """
        self._halt_event = event
        return self

    def with_budget(self, tracker: Tracker) -> Flow:
        """Attach a :class:`Tracker` reachable as ``ctx.budget``.

        Threads ``tracker`` through the execution environment so verbs
        can record LLM and operation costs against its cap. Auto-halt
        integration is opt-in on the tracker side: pass a shared
        ``asyncio.Event`` to both :meth:`Tracker.__init__` (``halt=``)
        and :meth:`with_halt`, and the tracker sets the event on the
        first cross into ``exceeded``.

        Consumers with hierarchical accounting attach the root tracker
        here; verbs reach descendants via ``ctx.budget.child(...)``.

        A subflow inherits the outer runtime's tracker automatically;
        calling ``.with_budget`` on a subflow overrides it for that subtree.

        Returns ``self`` for chaining.
        """
        self._budget_tracker = tracker
        return self

    def with_checkpointer(self, store: CheckpointStore, client_flow_id: str) -> Flow:
        """Attach a :class:`CheckpointStore` + agent-owned ``client_flow_id``.

        Wires save-at-``.iterate``-boundary saves and, on
        :meth:`run` ``resume=True``, a load-at-start that hydrates the
        run's payload before the first node dispatches. On fully
        successful :meth:`run` completion the framework calls
        :meth:`CheckpointStore.gc_trajectory` when the store's
        ``retention`` is ``"gc_on_success"``; the default ``"retain"``
        keeps the trajectory for audit. Cancellation, halt exits, and
        unhandled exceptions preserve the checkpoint regardless of
        retention so a subsequent resume can pick up.

        Both arguments bind together — the ``client_flow_id`` scopes every
        save/load/delete call and identifies the resumable trajectory. It
        is agent-owned: the framework never assigns one automatically.

        A subflow inherits the outer runtime's checkpointer + id
        automatically; calling ``.with_checkpointer`` on a subflow
        overrides both for that subtree.

        Resume semantics on the wired iterate:

        - **State** hydrates from the commit's root-scope Blob (via
          ``state_factory.restore`` if a ``state_factory`` is bound, else
          passthrough for plain dicts).
        - **Iteration counter** is restored: ``max_iters`` is a
          cumulative bound across resumes — saving at iteration N and
          resuming with ``max_iters=M`` runs ``max(0, M - N)`` further
          passes. A counter that already meets the bound exits without
          re-running the body.
        - **Ambients** (halt, budget, saia, traits, logger,
          checkpointer itself) are never serialized — they reattach
          from the current runtime, so a resumed run gets fresh
          handles under whichever ``.with_halt`` / ``.with_budget`` /
          ``.with_traits`` were wired at resume time.
        - **Deadline** is not restored — the wall clock starts fresh
          each run.

        Returns ``self`` for chaining.
        """
        self._checkpointer = store
        self._client_flow_id = client_flow_id
        return self

    def with_checkpoint_policy(
        self,
        policy: CheckpointPolicy | None = None,
        /,
        **kwargs: Any,
    ) -> Flow:
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
        resume: bool = False,
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
            resume: When ``True`` and :meth:`with_checkpointer` is wired,
                the framework resolves the latest commit via
                :meth:`CheckpointStore.resolve_ref` at start and, if a
                commit exists, reconstructs the scope tree and replaces
                ``state`` with the hydrated payload. A flow with a bound
                ``state_factory=`` reconstructs the payload via
                ``state_factory.restore``; a flow without ``state_factory``
                treats the stored payload as a plain dict. Absent-checkpoint
                resume is a no-op — the run proceeds with ``state`` as given.
                On fully successful completion the trajectory is gc'd when
                the store's ``retention`` is ``"gc_on_success"``. Requires
                :meth:`with_checkpointer` to be wired; raises otherwise.
                Bound parameter: not forwarded to the first node.
            **kwargs: Keyword inputs to the first node.

        Raises:
            RuntimeError: The flow has no nodes to run, OR ``resume=True``
                was requested without :meth:`with_checkpointer` wired.
                Missing :class:`SAIAFactory` no longer raises at run
                start — the error surfaces at the first ``ctx.saia``
                access instead, so verbs that don't consume ``ctx.saia``
                can run under a factoryless flow.
        """
        if resume and self._checkpointer is None:
            label = self._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} was run with resume=True but has no "
                f"checkpointer — call .with_checkpointer(store, client_flow_id) first"
            )
        active_state = self._wrap_top_state(state)
        replay: _ResumeReplay | None = None
        self._resume_saia_turn_bytes = {}
        if resume:
            active_state, replay = await self._hydrate_resume_state(active_state)
        self._replay_consumed = False
        self._halt_saved = False
        self._pending_saia_turn_bytes = {}
        self._pending_saia_turn_ancestors = {}
        result = await self._run_as_subflow(
            *args,
            state=active_state,
            runtime=self,
            parent_replay=replay,
            parent_extra=extra,
            **kwargs,
        )
        self._assert_replay_consumed(replay)
        await self._apply_clean_exit_retention()
        return result

    async def _apply_clean_exit_retention(self) -> None:
        """Apply the store's retention policy on the clean-exit path.

        Halt-triggered exits preserve the trajectory regardless of policy.
        On a clean exit: ``gc_on_success`` prunes; ``retain`` keeps the
        record and stamps a completion marker so a subsequent
        ``run(resume=True)`` doesn't replay the final iterate commit and
        re-execute chain steps after the iterate.
        """
        if (
            self._checkpointer is None
            or self._client_flow_id is None
            or self._halt_saved
            or (self._halt_event is not None and self._halt_event.is_set())
        ):
            return
        if self._checkpointer.retention == "gc_on_success":
            await maybe_await(self._checkpointer.gc_trajectory(self._client_flow_id))
        else:
            await self._stamp_completion_marker()

    async def _stamp_completion_marker(self) -> None:
        """Write a sentinel commit + ref marking the trajectory complete.

        The marker uses a reserved ``node_path="$complete"`` and
        ``produced_by.node_id="$complete"``; :meth:`_hydrate_resume_state`
        detects it and returns a fresh-run replay context.
        """
        assert self._checkpointer is not None
        assert self._client_flow_id is not None
        empty_tree = Tree.from_entries([])
        meta = self._completion_marker_meta()
        commit = Commit.build(root_tree_hash=empty_tree.content_hash, parent_hashes=(), meta=meta)
        cfid = self._client_flow_id
        await maybe_await(
            self._checkpointer.put_object(
                cfid, "tree", empty_tree.content_hash, empty_tree.to_bytes()
            )
        )
        await maybe_await(
            self._checkpointer.put_object(cfid, "commit", commit.content_hash, commit.to_bytes())
        )
        await maybe_await(self._checkpointer.put_ref(cfid, "$complete", 0, commit.content_hash))

    def _completion_marker_meta(self) -> CommitMeta:
        """Build :class:`CommitMeta` for the completion sentinel."""
        from llm_gent import __version__

        assert self._client_flow_id is not None
        return CommitMeta(
            client_flow_id=self._client_flow_id,
            node_path="$complete",
            iteration=0,
            produced_by=ProducedBy(
                node_id="$complete", verb_name=None, role=None, result_hash=None
            ),
            trace_ref=(),
            outcome="ok",
            flow_root_id=self._client_flow_id,
            timestamp_iso=datetime.now(UTC).isoformat(),
            framework_version=__version__,
        )

    def _assert_replay_consumed(self, replay: _ResumeReplay | None) -> None:
        """Raise if a resume request never found its save-point iterate.

        Belt-and-suspenders check called after :meth:`_run_as_subflow`
        returns. :meth:`_assert_replay_reachable` will typically have
        raised earlier — as soon as a Flow entry sees a path head that
        no chain step at that level provides. This post-run raise
        catches the residual cases where the entry-level pre-scan
        matched something but no iterate ever consumed the tail
        (structurally impossible under normal composition, but the
        check costs nothing and keeps the invariant explicit).

        The raise includes the full saved path (root → leaf) so ops
        triage can correlate the ancestor chain with the current
        composition tree and locate the layer where the graph diverged.
        """
        if replay is None or self._replay_consumed:
            return
        target_id = replay.remaining_path[-1] if replay.remaining_path else "<empty>"
        label = self._name or "<anonymous>"
        path_repr = " → ".join(replay.full_path) if replay.full_path else "<empty>"
        raise RuntimeError(
            f"Flow {label!r}: resume checkpoint's save-point iterate "
            f"id {target_id!r} was not found in the composition graph "
            f"during the run — the graph has structurally changed "
            f"since the checkpoint was written. "
            f"Saved path (root→leaf): {path_repr}"
        )

    def _compute_chain_ids(self, env: _RunEnv) -> tuple[str, ...]:
        """Content-addressed node id for every step in this Flow's chain."""
        return tuple(_compute_node_id(env.chain_context, n, i) for i, n in enumerate(self._nodes))

    def _assert_replay_reachable(self, env: _RunEnv, chain_ids: tuple[str, ...]) -> None:
        """Fail-fast: raise if the replay's remaining head is unreachable at this Flow level.

        Called at the top of every chain walk. If a resume replay is
        threaded and its ``remaining_path[0]`` does not match any of
        this level's ``chain_ids``, the composition graph has changed
        since the checkpoint was written and there is no descent from
        this level that could reach the save-point. Raising here
        prevents unrelated chain steps and iterates from running fresh
        under a doomed replay — the whole point of the head-pop
        redesign. No-op when there is no replay, the path is empty,
        or the leaf has already been consumed.
        """
        replay = env.replay
        if replay is None or not replay.remaining_path:
            return
        if env.runtime._replay_consumed:
            return
        head = replay.remaining_path[0]
        if head in chain_ids:
            return
        label = self._name or "<anonymous>"
        ids_repr = ", ".join(chain_ids) if chain_ids else "<empty>"
        full_repr = " → ".join(replay.full_path) if replay.full_path else "<empty>"
        raise RuntimeError(
            f"Flow {label!r}: resume path head {head!r} not found in this level's "
            f"chain step ids [{ids_repr}] — the composition graph has "
            f"structurally changed since the checkpoint was written. "
            f"Saved path (root→leaf): {full_repr}"
        )

    def _resume_start_index(self, env: _RunEnv, chain_ids: tuple[str, ...]) -> int:
        """Return the chain index to start execution from on resume.

        ``0`` on a fresh run (no replay, empty path, or already-consumed
        leaf). On resume, returns the index of the chain step whose id
        matches ``env.replay.remaining_path[0]`` — that step is on the
        save-point ancestor chain (or IS the save-point leaf when
        ``remaining_path`` has shrunk to one entry); every prior chain
        step completed BEFORE the checkpoint was written and re-running
        them would clobber hydrated state (their verbs typically write
        to ``ctx.state.data`` on the way through).

        :meth:`_assert_replay_reachable` guarantees the head is in
        ``chain_ids`` before this fires, so the loop is a lookup, not
        a search.

        Contract on the on-path node when ``start_index > 0``: it runs
        with no ``prev_result`` — its predecessor didn't re-run, so
        there is no return value to thread in. Iterate bodies that
        depend on the outer chain's return value on resume-first-
        iteration must either be at chain index 0 or read from state.
        """
        replay = env.replay
        if replay is None or not replay.remaining_path:
            return 0
        if env.runtime._replay_consumed:
            return 0
        head = replay.remaining_path[0]
        for i, cid in enumerate(chain_ids):
            if cid == head:
                # Single-element remaining path AND target is a plain-verb
                # or Branch/Map/subflow chain step means the halt-observation
                # site saved here — mark consumed so :meth:`_assert_replay_consumed`
                # does not fire. Iterate has its own consumption in
                # :func:`_resolve_iterate_resume`; Branch/Map/subflow do not
                # (after pop the path is empty, nothing inside will consume it).
                if len(replay.remaining_path) == 1 and not isinstance(
                    self._nodes[i].target, _Iterate
                ):
                    env.runtime._replay_consumed = True
                return i
        return 0

    async def _run_as_subflow(
        self,
        *args: Any,
        state: State[Any],
        runtime: Flow,
        parent_halt: asyncio.Event | None = None,
        parent_budget: Tracker | None = None,
        parent_checkpointer: CheckpointStore | None = None,
        parent_client_flow_id: str | None = None,
        parent_chain_context: str = "",
        parent_ancestor_chain: tuple[str, ...] = (),
        parent_replay: _ResumeReplay | None = None,
        parent_extra: dict[str, Any] | None = None,
        parent_policy: CheckpointPolicy | None = None,
        **kwargs: Any,
    ) -> Any:
        """Internal entry: walk nodes with caller-supplied ``State`` and runtime.

        The executor's subflow helpers (``.call`` targeting a Flow, and the
        ``.branch`` / ``.iterate`` / ``.map`` bodies) invoke this directly so
        the outer runtime's factory and saia cache are shared with the
        subflow. State arrives pre-wrapped — top-level wrapping happens once
        in :meth:`run`.

        ``parent_halt`` / ``parent_budget`` / ``parent_checkpointer`` are the
        effective ambients from the calling scope — nested subflows fall
        back to them when they have no local
        ``.with_halt()`` / ``.with_budget()`` / ``.with_checkpointer()``
        override, preserving an intermediate layer's ambient through
        arbitrarily deep nesting. ``parent_client_flow_id`` pairs with
        ``parent_checkpointer``; either both are inherited or a local
        override supplies both.

        ``parent_chain_context`` is the hash the executor uses to compute
        this Flow's chain-step node IDs (empty at run root; extended by
        :func:`_descend_context` at each subflow / arm / body boundary).
        ``parent_ancestor_chain`` is the tuple of ancestor ``_Node`` IDs
        from root down to the ``_Node`` whose descent entered this Flow;
        it grows by one on every recursion. ``parent_replay`` carries a
        pending checkpoint replay when :meth:`run` was invoked with
        ``resume=True``; ``None`` for a fresh run.
        """
        if not self._nodes:
            raise RuntimeError(f"Flow {self._name!r} has no nodes to run")
        env = self._make_run_env(
            runtime=runtime,
            state=state,
            parent_halt=parent_halt,
            parent_budget=parent_budget,
            parent_checkpointer=parent_checkpointer,
            parent_client_flow_id=parent_client_flow_id,
            parent_chain_context=parent_chain_context,
            parent_ancestor_chain=parent_ancestor_chain,
            parent_replay=parent_replay,
            parent_extra=parent_extra,
            parent_policy=parent_policy,
        )
        label = self._name or "<anonymous>"
        is_subflow = runtime is not self
        env.lg.debug(
            "starting flow run",
            extra={"flow": label, "nodes": len(self._nodes), "subflow": is_subflow},
        )
        chain_ids = self._compute_chain_ids(env)
        self._assert_replay_reachable(env, chain_ids)
        start_index = self._resume_start_index(env, chain_ids)
        result = await self._walk_chain(env, chain_ids, start_index, args, kwargs)
        env.lg.debug("completed flow run", extra={"flow": label, "subflow": is_subflow})
        return result

    async def _walk_chain(
        self,
        env: _RunEnv,
        chain_ids: tuple[str, ...],
        start_index: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute chain steps from ``start_index`` onward, threading returns.

        ``start_index > 0`` on resume: predecessors already completed
        before the checkpoint was written; the on-path step at
        ``start_index`` runs with no ``prev_result`` (see
        :meth:`_resume_start_index` for the contract).

        Halt observation: between chain steps (never at the very first
        iteration of this walk, so resume runs at least the halted
        step), if ``env.halt`` is set, stamp a halted commit at the
        not-yet-run step's position and break. On a subsequent
        ``run(resume=True)``, that ref resolves to this commit and the
        walk restarts at the halted step.
        """
        result: Any = UNSET
        for index in range(start_index, len(self._nodes)):
            if await self._observe_chain_halt(env, chain_ids, index, start_index):
                break
            node = self._nodes[index]
            node_id = chain_ids[index]
            node_args: tuple[Any, ...]
            node_kwargs: dict[str, Any]
            if index == start_index and start_index > 0:
                node_args, node_kwargs = (), {}
            else:
                node_args, node_kwargs = _step_inputs(index, node, result, args, kwargs)
            ctx = _build_ctx(node.target, env, node_id)
            result = await _execute_node(node, ctx, env, node_args, node_kwargs, node_id)
        return result

    async def _observe_chain_halt(
        self, env: _RunEnv, chain_ids: tuple[str, ...], index: int, start_index: int
    ) -> bool:
        """Save a halt commit between chain steps when appropriate, return True if saved.

        Only fires at the top-level chain (``env.runtime is self``) with a
        checkpointer bound and past the first step of this walk (so resume
        runs at least the halted step). Nested body chains let halt
        propagate to iterate boundaries where iteration state is consistent.

        When the just-completed step paused SAIA mid-turn (a Loop deposited
        bytes on ``env.runtime._pending_saia_turn_bytes``), lands the halt
        commit at THAT step's node so resume re-dispatches it — its Loop's
        ``__call__`` then picks up the saia_turn entry and hands SAIA
        ``resume=True`` with the rebuilt conversation. Otherwise saves at
        the not-yet-run step (the normal chain-halt case).
        """
        if (
            index <= start_index
            or env.runtime is not self
            or env.checkpointer is None
            or env.halt is None
            or not env.halt.is_set()
        ):
            return False
        just_completed = chain_ids[index - 1]
        halt_node_id = (
            just_completed
            if self._just_completed_owns_pending_saia(env, just_completed)
            else chain_ids[index]
        )
        await _save_halt_checkpoint(env, 0, halt_node_id, env.state)
        return True

    @staticmethod
    def _just_completed_owns_pending_saia(env: _RunEnv, node_id: str) -> bool:
        """True when ``node_id`` is a pending Loop's own id or an ancestor of one.

        Direct match covers the `.call(loop_verb)` case (Loop's ctx._node_id
        IS the chain step's id). Ancestry match covers nested Loops — Loop
        paused inside an iterate body inside the chain step, where the
        pending entry's key is the Loop's descendant id computed under the
        chain step's descent context. Either match means resume should
        re-dispatch the chain step so the Loop's __call__ picks up the
        saia_turn entry.
        """
        pending = env.runtime._pending_saia_turn_bytes
        if node_id in pending:
            return True
        ancestors = env.runtime._pending_saia_turn_ancestors
        return any(node_id in chain for chain in ancestors.values())

    def _make_run_env(
        self,
        *,
        runtime: Flow,
        state: State[Any],
        parent_halt: asyncio.Event | None,
        parent_budget: Tracker | None,
        parent_checkpointer: CheckpointStore | None,
        parent_client_flow_id: str | None,
        parent_chain_context: str = "",
        parent_ancestor_chain: tuple[str, ...] = (),
        parent_replay: _ResumeReplay | None = None,
        parent_extra: dict[str, Any] | None = None,
        parent_policy: CheckpointPolicy | None = None,
    ) -> _RunEnv:
        """Resolve local-override-wins ambients and build the per-run environment.

        Local ``.with_halt`` / ``.with_budget`` / ``.with_checkpointer``
        wins over the caller's parent ambients; unset locals fall back to
        the parent so an intermediate layer's ambient survives arbitrarily
        deep nesting. Checkpointer + ``client_flow_id`` inherit as a pair.

        ``parent_chain_context`` and ``parent_ancestor_chain`` are copied
        verbatim: the descent sites in :mod:`._executor` are the ones
        that extend them when recursing into a subflow / arm / body.
        ``parent_replay`` is the pending resume context, if any.
        """
        halt = self._halt_event if self._halt_event is not None else parent_halt
        budget = self._budget_tracker if self._budget_tracker is not None else parent_budget
        if self._checkpoint_policy is not None:
            policy = self._checkpoint_policy
        elif parent_policy is not None:
            policy = parent_policy
        else:
            policy = CheckpointPolicy()
        checkpointer: CheckpointStore | None
        client_flow_id: str | None
        if self._checkpointer is not None:
            checkpointer = self._checkpointer
            client_flow_id = self._client_flow_id
        else:
            checkpointer = parent_checkpointer
            client_flow_id = parent_client_flow_id
        return _RunEnv(
            runtime=runtime,
            state=state,
            lg=runtime._lg,
            halt=halt,
            budget=budget,
            checkpointer=checkpointer,
            client_flow_id=client_flow_id,
            chain_context=parent_chain_context,
            ancestor_chain=parent_ancestor_chain,
            replay=parent_replay,
            extra=parent_extra if parent_extra is not None else {},
            policy=policy,
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

    async def _hydrate_resume_state(
        self, fallback: State[Any]
    ) -> tuple[State[Any], _ResumeReplay | None]:
        """Load the latest commit and reconstruct state + a replay context.

        Called only when :meth:`run` was invoked with ``resume=True``.

        Sequence:

        1. :meth:`CheckpointStore.resolve_ref` under :attr:`_client_flow_id`
           returns the latest commit hash across every ``node_path``, or
           ``None`` (fresh run — no prior checkpoint).
        2. :meth:`CheckpointStore.get_object` fetches the commit bytes;
           :meth:`Commit.from_bytes` re-derives :class:`CommitMeta` +
           :class:`ProducedBy` + :class:`TraceRef`.
        3. The commit's ``root_tree_hash`` fetches the :class:`Tree`
           object; each :class:`TreeEntry` fetches its Blob. Scope order
           is root → leaf via the zero-padded ``scope_id`` (canonical
           sort).
        4. The root scope's payload rehydrates the top-level
           :class:`State` (via ``state_factory.restore`` when bound,
           passthrough otherwise). The leaf scope's payload rides on the
           :class:`_ResumeReplay` as ``child_state_data`` for the
           save-point iterate to restore its own scope.
        5. ``node_path`` splits on ``"/"`` back into the ancestor chain
           the executor's head-pop replay expects. blake2b hex has no
           slashes, so the round-trip is exact.

        Returns ``(fallback, None)`` when no commit exists yet — the
        caller's ``state=`` (or the flow's construction state) is used
        and no replay is scheduled.
        """
        assert self._checkpointer is not None
        if self._client_flow_id is None:
            label = self._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} was run with resume=True but has no "
                f"client_flow_id — call .with_checkpointer(store, client_flow_id) first"
            )
        loaded = await self._load_latest_commit_scopes()
        if loaded is None:
            return fallback, None
        commit, scope_data = loaded
        # A completion marker (stamped on clean exit under "retain") means
        # the trajectory finished successfully — do not replay.
        if commit.meta.node_path == "$complete":
            return fallback, None
        await self._load_resume_saia_turn_bytes(commit)
        return self._replay_from_commit(commit, scope_data)

    async def _load_resume_saia_turn_bytes(self, commit: Commit) -> None:
        """Pull every saia_turn entry out of ``commit.meta.trace_ref`` and cache the blob bytes.

        Each ``TraceRef(kind="saia_turn", ...)`` on the halted commit
        was stamped by :func:`_stash_pending_saia_turn` with
        ``id=f"{node_id}:{blob_hash}"``. Split on the first colon,
        fetch the blob under ``blob_hash``, and stash
        ``{node_id: blob_bytes}`` on :attr:`_resume_saia_turn_bytes`
        so the Loop at that node can pick its own entry up on its
        first dispatch and hand the reconstructed conversation to
        SAIA with ``resume=True``.

        Silently skips entries whose blob is missing from the store
        — the run then falls back to a fresh dispatch at that Loop.
        Entries whose ``id`` is not in ``node_id:blob_hash`` shape
        are ignored (defensive against future ``saia_turn`` variants
        this framework does not understand).
        """
        assert self._checkpointer is not None
        assert self._client_flow_id is not None
        for ref in commit.meta.trace_ref:
            if ref.kind != "saia_turn":
                continue
            node_id, sep, blob_hash = ref.id.partition(":")
            if not sep or not node_id or not blob_hash:
                continue
            payload = await maybe_await(
                self._checkpointer.get_object(self._client_flow_id, "blob", blob_hash)
            )
            if payload is None:
                continue
            self._resume_saia_turn_bytes[node_id] = payload

    def _replay_from_commit(
        self, commit: Commit, scope_data: list[Any]
    ) -> tuple[State[Any], _ResumeReplay | None]:
        """Split root / middle / leaf scope payloads, return State + replay.

        Root scope hydrates the top-level :class:`State`. Every non-root
        scope rides on ``intermediate_scope_data`` in order; each
        scope-creating descent (``.call(state=)``, ``.iterate(state=)``,
        ``.map(state=)``) consumes the next entry at its own descent
        site via :func:`_consume_scope_data`. A leaf iterate without a
        ``state_fn`` creates no scope of its own and simply reuses the
        parent scope with the fast-forwarded iteration count.
        """
        root_raw = scope_data[0] if scope_data else None
        hydrated_root = (
            root_raw
            if self._state_factory is None or root_raw is None
            else self._state_factory.restore(root_raw if isinstance(root_raw, dict) else {})
        )
        intermediate_raw = tuple(scope_data[1:])
        path_tuple = tuple(commit.meta.node_path.split("/")) if commit.meta.node_path else ()
        return (
            State(data=hydrated_root, _factory=self._state_factory),
            _ResumeReplay(
                remaining_path=path_tuple,
                full_path=path_tuple,
                iteration=commit.meta.iteration,
                child_state_data=None,
                intermediate_scope_data=intermediate_raw,
            ),
        )

    async def _load_latest_commit_scopes(self) -> tuple[Commit, list[Any]] | None:
        """Resolve the latest commit + walk its tree, returning ``(Commit, scope_data)``.

        ``scope_data`` is root → leaf JSON payloads (one per :class:`Tree`
        entry). Returns ``None`` when the ref, commit, tree, or any blob
        is missing — the caller treats each miss as "no resumable
        checkpoint" and falls through to a fresh run.
        """
        assert self._checkpointer is not None
        assert self._client_flow_id is not None
        commit_hash = await maybe_await(self._checkpointer.resolve_ref(self._client_flow_id))
        if commit_hash is None:
            return None
        commit_bytes = await maybe_await(
            self._checkpointer.get_object(self._client_flow_id, "commit", commit_hash)
        )
        if commit_bytes is None:
            return None
        commit = Commit.from_bytes(commit_bytes)
        tree_bytes = await maybe_await(
            self._checkpointer.get_object(self._client_flow_id, "tree", commit.root_tree_hash)
        )
        if tree_bytes is None:
            return None
        tree = Tree.from_bytes(tree_bytes)
        scope_data: list[Any] = []
        for entry in tree.entries:
            blob = await maybe_await(
                self._checkpointer.get_object(self._client_flow_id, "blob", entry.child_hash)
            )
            if blob is None:
                return None
            scope_data.append(json.loads(blob.decode("utf-8")))
        return commit, scope_data

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


# -----------------------------------------------------------------------------
# Builder-side validators
# -----------------------------------------------------------------------------


def _validate_target(target: Any) -> None:
    """Reject anything that isn't a verb (callable with .role) or a Flow.

    ``.role`` may be ``None`` — the pure-Python-verb form produced by
    ``@verb`` without a role. The framework still dispatches such verbs;
    they just cannot read ``ctx.saia`` (which stays ``None``).
    """
    if isinstance(target, Flow):
        return
    if not callable(target):
        raise TypeError(
            f"target must be a verb (callable with .role) or a Flow; got {type(target).__name__}"
        )
    if not hasattr(target, "role"):
        raise TypeError(f"verb target must carry a .role attribute; got {type(target).__name__}")
    if target.role is not None and not isinstance(target.role, Role):
        raise TypeError(
            f"verb target .role must be a Role instance or None; got {type(target.role).__name__}"
        )
    _reject_reserved_kwarg(target, "state")
    _reject_reserved_kwarg(target, "runtime")
    _reject_reserved_kwarg(target, "resume")


def _reject_reserved_kwarg(verb: Any, name: str) -> None:
    """Reject a verb whose signature declares a :meth:`Flow.run`-bound kwarg.

    ``Flow.run(state=...)`` binds ``state`` to seed the top-level payload
    and strips it before forwarding kwargs to the first node. A verb whose
    signature declares ``state`` as a keyword-visible parameter would never
    see a value passed via ``.run()`` — silent misbehavior. Rejected at
    build time so the collision surfaces where the graph is authored.

    The first positional is conventionally ``ctx`` and is skipped: the
    framework always binds it, so its name doesn't collide with any run
    kwarg. Ignores verbs whose signature cannot be introspected (C
    callables, some partials) — those cannot statically collide with a
    bound kwarg either. ``**kwargs``-only verbs are allowed: :meth:`run`
    already strips the reserved name before forwarding, so the verb's
    ``**kwargs`` never sees it.
    """
    try:
        sig = inspect.signature(verb)
    except (TypeError, ValueError):
        return
    for param in list(sig.parameters.values())[1:]:
        if param.name != name:
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            verb_name = getattr(verb, "__name__", type(verb).__name__)
            raise TypeError(
                f"verb {verb_name!r} declares reserved parameter {name!r} — "
                f"Flow.run({name}=...) binds this name and it would never "
                f"reach the verb; rename the parameter"
            )
        return


def _require_state_for_merge(
    state: StateProject | None, merge: StateMerge | None, method: str
) -> None:
    """Enforce that ``merge=`` is only legal with ``state=`` (nothing to merge otherwise)."""
    if merge is not None and state is None:
        raise ValueError(
            f"{method}(merge=...) requires state= "
            "(nothing to merge back without an isolated child state)"
        )


def _materialize(buildable: Any, lg: Logger, name: str) -> Flow:
    """Turn a :data:`Buildable` (Flow, verb, or ``lambda f: ...`` callback) into a Flow.

    A ``Flow`` is returned as-is. A callable carrying a :class:`Role` on
    ``.role`` (a verb — whether a module-level ``@verb`` function or a
    bound instance method) is wrapped as a single-node flow calling that
    verb. Any other callable is invoked against a fresh Flow it may mutate
    (the return value, if any, is ignored). Anything else is a
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
    if hasattr(buildable, "role"):
        # A verb — role attribute may be a Role (role-bound) or None
        # (pure-Python verb). Either way, wrap as a single-node subflow.
        # Defense-in-depth: validate role here (also checked in fresh.call).
        role = buildable.role
        if role is not None and not isinstance(role, Role):
            raise TypeError(
                f"verb target .role must be a Role instance or None; got {type(role).__name__}"
            )
        fresh = Flow(lg=lg, name=name)
        fresh.call(buildable)
        return fresh
    fresh = Flow(lg=lg, name=name)
    buildable(fresh)
    return fresh


_NODE_ID_DIGEST_SIZE = 8
"""Byte length of the blake2b digest for node IDs — 64 bits, 16 hex chars.

Sized for headroom: a composition tree of a few thousand nodes has
essentially zero birthday-collision risk at 2^32, which is the design
guarantee behind treating node IDs as globally unique in the graph.
Bumping to 16 (128 bits) would leave zero doubt but doubles envelope
size; 8 is the deliberate default.
"""


def _target_qualname(target: Any) -> str:
    """Stable identity string for a node target — feeds :func:`_compute_node_id`.

    Verb / plain callable → ``verb:<__module__>.<__qualname__>`` (module
    prefix prevents cross-module collisions between two functions with
    the same qualname). :class:`Flow` subflow → ``flow:<name>`` (or
    ``flow:<anonymous>`` when unnamed). Composition primitives
    (:class:`_Branch`, :class:`_Iterate`, :class:`_Map`) → the
    primitive's kind string; the primitive's identity flows from the
    outer :class:`_Node`'s chain position and its enclosing
    ``chain_context``, not from any label on the primitive itself.
    """
    if isinstance(target, Flow):
        return f"flow:{target.name or '<anonymous>'}"
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Iterate):
        return "iterate"
    if isinstance(target, _Map):
        return "map"
    module = getattr(target, "__module__", "?")
    qualname = getattr(target, "__qualname__", type(target).__name__)
    return f"verb:{module}.{qualname}"


def _node_kind(node: _Node) -> str:
    """Chain-step kind for the composition tree hash: ``call`` / ``branch`` / ``iterate`` / ``map``."""
    target = node.target
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Iterate):
        return "iterate"
    if isinstance(target, _Map):
        return "map"
    return "call"


def _compute_node_id(chain_context: str, node: _Node, position: int) -> str:
    """Runtime content-addressed node ID for the chain step at ``position``.

    Composes the enclosing Flow's ``chain_context`` with this node's
    local key ``(kind, position, target_qualname)``. The chain_context
    is itself a hash chain from the run's root down through every
    subflow / branch-arm / iterate-body descent above this Flow (see
    :func:`_descend_context`), so the resulting node ID is globally
    unique across the entire composition tree — a shared subflow used
    at two call sites produces two distinct IDs for the same underlying
    ``_Node`` because their ``chain_context`` values differ.

    Deterministic: identical composition graphs produce identical IDs
    across processes / Python versions (blake2b is stable and every
    input is a Unicode-canonical string). Collision-free at the design
    level for any well-formed graph — see :data:`_NODE_ID_DIGEST_SIZE`
    for the birthday-collision margin.
    """
    payload = f"{chain_context}|{_node_kind(node)}|{position}|{_target_qualname(node.target)}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_NODE_ID_DIGEST_SIZE).hexdigest()


def _descend_context(parent_node_id: str, boundary: str) -> str:
    """Chain context for a Flow entered from ``parent_node_id`` via ``boundary``.

    ``boundary`` names the slot of the parent node this Flow fills:
    ``"body"`` (an :class:`_Iterate` body), ``"then"`` / ``"else"``
    (arms of a :class:`_Branch`), or ``"call"`` (a subflow reached from
    :meth:`Flow.call`). Baking the boundary into the descended context
    keeps the two arms of a branch and the body of an iterate at
    identity-distinct positions even when they share a target.
    """
    payload = f"{parent_node_id}|{boundary}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_NODE_ID_DIGEST_SIZE).hexdigest()
