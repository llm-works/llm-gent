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
import inspect
from typing import Any

from appinfra.log import Logger

from ..core.traits import Registry as TraitRegistry
from ._executor import _build_ctx, _execute_node, _step_inputs
from .context import Context
from .factory import SAIAFactory
from .nodes import (
    UNSET,
    AfterHook,
    AggregateFn,
    GuardFn,
    ItemsFn,
    OnErrorFn,
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
from .role import Role
from .state import State


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
        saia_f: SAIAFactory | None = None,
        state: Any = UNSET,
        traits: TraitRegistry | None = None,
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
            saia_f: A :class:`SAIAFactory` that builds role-bound saia
                instances. The ``_f`` suffix carries the framework-wide
                policy: any ``saia_f=`` kwarg takes a factory, never a
                saia instance. Required only when role-bound code
                accesses ``ctx.saia`` — verbs that route LLM calls
                through their own configuration can run without one.
                A subflow borrows the factory from the runtime it
                executes under.
            state: User-owned shared state object. Verbs read and (typically)
                mutate it in place. Opaque to the flow; may be overridden per
                :meth:`run` invocation.
            traits: Optional trait registry surfaced on every dispatched
                :class:`Context` as ``ctx.traits``. When ``None``, verbs see
                ``ctx.traits is None``. A subflow inherits the outer
                runtime's registry (like the saia cache) via the same
                internal handoff, so mounting on the top-level flow is
                enough to reach every nested dispatch.
        """
        self._lg = lg
        self._name = name
        self._saia_f = saia_f
        self._state = state
        self._traits = traits
        self._halt_event: asyncio.Event | None = None
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

    @property
    def traits(self) -> TraitRegistry | None:
        """The trait registry this flow was constructed with, or ``None``."""
        return self._traits

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
        composition run, so ``ctx.state`` wraps this flow's construction
        state (defaulting to a fresh empty ``dict`` when none was supplied).
        Verbs that need a live run-wide payload from a :meth:`run` invocation
        must be reached via :meth:`run` rather than dispatched ad hoc.
        """
        if name not in self._verbs:
            raise KeyError(f"no verb registered under name {name!r}")
        verb = self._verbs[name]
        self._lg.debug("dispatching verb", extra={"verb": name, "role": verb.role.name})
        payload = self._state if self._state is not UNSET else {}
        ctx = Context(
            role=verb.role,
            state=payload if isinstance(payload, State) else State(data=payload),
            flow=self,
            traits=self._traits,
            halt=self._halt_event,
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
    ) -> Flow:
        """Append a bounded iteration: run ``body`` until a stop condition holds.

        Each iteration's return becomes the next iteration's input; the first
        iteration receives the iterate node's input (previous chain step's
        result). The node's own result is the last iteration's return.

        Args:
            body: A :class:`Flow` or ``lambda f: ...`` callback for the body.
                Runs at least once.
            until: ``(ctx) -> bool``. May be async. Checked **after** each
                iteration completes — truthy → stop. State typically drives
                termination via ``ctx.state``.
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

    async def run(self, *args: Any, state: Any = UNSET, **kwargs: Any) -> Any:
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
            **kwargs: Keyword inputs to the first node.

        Raises:
            RuntimeError: The flow has no nodes to run. Missing
                :class:`SAIAFactory` no longer raises at run start — the
                error surfaces at the first ``ctx.saia`` access instead,
                so verbs that don't consume ``ctx.saia`` can run under a
                factoryless flow.
        """
        active_state = self._wrap_top_state(state)
        return await self._run_as_subflow(*args, state=active_state, runtime=self, **kwargs)

    async def _run_as_subflow(
        self,
        *args: Any,
        state: State,
        runtime: Flow,
        parent_halt: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Any:
        """Internal entry: walk nodes with caller-supplied ``State`` and runtime.

        The executor's subflow helpers (``.call`` targeting a Flow, and the
        ``.branch`` / ``.iterate`` / ``.map`` bodies) invoke this directly so
        the outer runtime's factory and saia cache are shared with the
        subflow. State arrives pre-wrapped — top-level wrapping happens once
        in :meth:`run`.

        ``parent_halt`` is the effective halt from the calling scope — nested
        subflows fall back to it when they have no local ``.with_halt()``
        override, preserving an intermediate layer's halt through arbitrarily
        deep nesting.
        """
        if not self._nodes:
            raise RuntimeError(f"Flow {self._name!r} has no nodes to run")
        halt = self._halt_event if self._halt_event is not None else parent_halt
        env = _RunEnv(runtime=runtime, state=state, lg=runtime._lg, halt=halt)
        label = self._name or "<anonymous>"
        is_subflow = runtime is not self
        env.lg.debug(
            "starting flow run",
            extra={"flow": label, "nodes": len(self._nodes), "subflow": is_subflow},
        )
        result: Any = UNSET
        for index, node in enumerate(self._nodes):
            node_args, node_kwargs = _step_inputs(index, node, result, args, kwargs)
            ctx = _build_ctx(node.target, env)
            result = await _execute_node(node, ctx, env, node_args, node_kwargs)
        env.lg.debug("completed flow run", extra={"flow": label, "subflow": is_subflow})
        return result

    def _wrap_top_state(self, state: Any) -> State:
        """Wrap a top-level ``run(state=...)`` payload as :class:`State`.

        Resolves the ``state=UNSET`` sentinel to the flow's construction
        default (or a fresh empty ``dict`` when none was supplied). Explicit
        ``state=None`` is honored — the payload becomes ``None``. Already-
        wrapped :class:`State` inputs are returned unchanged so a caller can
        thread a run-wide state instance across multiple :meth:`run` calls.
        """
        payload = (self._state if self._state is not UNSET else {}) if state is UNSET else state
        return payload if isinstance(payload, State) else State(data=payload)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _saia_for(self, role: Role) -> Any:
        """Return a cached saia for ``role``, building it on first request."""
        cached = self._saia_by_role.get(role)
        if cached is not None:
            return cached
        if self._saia_f is None:
            label = self._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} has no SAIAFactory — saia_f= was not supplied at "
                f"construction (needed to build saia for role {role.name!r})"
            )
        built = self._saia_f.build(role)
        self._saia_by_role[role] = built
        return built


# -----------------------------------------------------------------------------
# Builder-side validators
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
    _reject_reserved_kwarg(target, "state")
    _reject_reserved_kwarg(target, "runtime")


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
    if isinstance(getattr(buildable, "role", None), Role):
        fresh = Flow(lg=lg, name=name)
        fresh.call(buildable)
        return fresh
    fresh = Flow(lg=lg, name=name)
    buildable(fresh)
    return fresh
