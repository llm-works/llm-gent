# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Loop — Flow-body primitive wrapping one saia.complete() invocation.

A ``Loop`` is a Flow-body citizen: it carries a :class:`Role` and is
dispatched by a :class:`Flow` like an ``@verb`` function or a bound-method
verb. Its body is a single ``saia.complete(...)`` invocation, wired with:

- lifecycle hooks (``on_iteration`` / ``on_executor_ready`` / ``on_cost``
  / ``on_complete`` / ``on_paused`` / ``on_cancelled`` / ``on_failed`` /
  ``on_finally``)
- halt bridging to SAIA's ``abort_signal``

Halt resolution rule: an explicit ``Loop(halt=X)`` at construction wins
over ambient ``ctx.halt`` — matches the ``ctx.saia`` precedent. Whichever
is effective becomes SAIA's ``abort_signal``.

Loop is CAS-native for durable pause capture — when a paused result
includes a conversation and a
:class:`~llm_saia.core.conversation.ConversationFactory` is wired,
the conversation is serialized and published for the halt-observation
site to stamp as a Blob referenced by ``TraceRef(kind="saia_turn", ...)``
on the Flow's halt commit.

:class:`LoopFactory` bundles the cross-cutting config (logger, SAIAFactory,
halt) so consumers wire once at the app boundary and ``.create(role,
**hooks)`` many Loops. It mirrors :class:`FlowFactory`'s shape so a
shared halt event threads uniformly across a mixed Loop-and-Flow tree.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from appinfra.log import Logger
from llm_saia import SAIA
from llm_saia.core.conversation import ConversationFactory

from .checkpoint import maybe_await
from .context import Context
from .factory import SAIAFactory
from .role import Role
from .state.saia_turn import SaiaTurnEnvelope


# ----------------------------------------------------------------------------
# Hook types
# ----------------------------------------------------------------------------


OnIteration = Callable[[int, Any, Context[Any]], Any]
"""``(iteration, response, ctx) -> None`` — bridges to SAIA's per-turn hook.

The second positional is the raw SAIA ``ChatResponse`` for that turn
— consumers wire per-turn narration and checkpoint save here. May be
async; return value ignored.
"""

OnComplete = Callable[[Any, Context[Any]], Any]
"""``(saia_result, ctx) -> saia_result | override | None`` — fires after
:meth:`saia.complete` returns non-paused.

Skipped when the run paused. May be async. If the hook returns a non-``None``
value, that value replaces the raw SAIA result as :meth:`Loop.__call__`'s
return; returning ``None`` (or an implicit fall-through) preserves the raw
result. Use this to map the SAIA ``TaskResult`` to a domain-specific shape
without a closure smuggling the value out.
"""

OnPaused = Callable[[Any, Context[Any]], Any]
"""``(saia_result, ctx) -> saia_result | override | None`` — fires after
:meth:`saia.complete` returns paused (mirror of :data:`OnComplete`).

Runs instead of ``on_complete``. Return-value semantics match
:data:`OnComplete`: non-``None`` replaces the raw SAIA result. May be async.
Consumers wire paused-status side effects (recorder marks, resumable-run
notifications) here rather than around ``await loop(...)``.
"""

OnCancelled = Callable[[Context[Any]], Any]
"""``(ctx) -> None`` — fires when :meth:`saia.complete` raises
:class:`asyncio.CancelledError`.

Runs before the cancellation is re-raised, then :data:`OnFinally` fires.
The checkpoint is intentionally NOT deleted on this path so a subsequent
resume can pick up. May be async; return value ignored.
"""

OnFailed = Callable[[Exception, Context[Any]], Any]
"""``(exc, ctx) -> None`` — fires when :meth:`saia.complete` raises any
non-cancellation :class:`Exception`.

Runs before the exception is re-raised, then :data:`OnFinally` fires. The
checkpoint is intentionally NOT deleted on this path. May be async; return
value ignored.
"""

OnFinally = Callable[[Context[Any]], Any]
"""``(ctx) -> None`` — fires last on every dispatch, regardless of outcome.

Runs after exactly one of :data:`OnComplete` / :data:`OnPaused` /
:data:`OnCancelled` / :data:`OnFailed` (and after :data:`OnCost` on the
non-exception paths). Symmetric with a ``finally:`` block wrapped around
``await loop(...)`` — for cleanup that must run whether the loop completed,
paused, was cancelled, or failed. May be async; return value ignored.
"""

OnExecutorReady = Callable[[Any, Context[Any]], Any]
"""``(saia, ctx) -> None`` — fires once per dispatch after ``ctx.saia`` is resolved.

Runs after the enclosing flow builds a role-bound saia and before
:meth:`saia.complete`, on every ``Loop.__call__`` (so once per iteration
when the Loop is an ``.iterate`` or ``.map`` body). Consumers reach into
the saia instance's tool executor to inject per-run values that aren't
known at saia-factory-construction time (``run_config``, ``campaign_id``,
``budget``, etc.). May be async; return value ignored.
"""

OnCost = Callable[[Any, Context[Any]], Any]
"""``(saia_result, ctx) -> None`` — fires after :meth:`saia.complete` for cost accounting.

Distinct from ``on_complete``: resource cost is a separate concern from
lifecycle, and runs even on a paused result. Consumers typically inspect
``result.trace`` / token counts here. May be async; return value ignored.
"""


# ----------------------------------------------------------------------------
# Loop
# ----------------------------------------------------------------------------


class Loop:
    """Flow-body primitive wrapping one ``saia.complete()`` invocation.

    Carries a :class:`Role` so a :class:`Flow` dispatches it like any
    verb: the flow builds a role-bound ``ctx.saia`` and hands the Loop a
    :class:`Context`. Loop then drives one ``saia.complete(...)``,
    bridging its lifecycle hooks and halt event to SAIA's own turn loop.

    Halt resolution: ``Loop(halt=X)`` at construction wins over ambient
    ``ctx.halt`` — matches the ``ctx.saia`` precedent. Whichever is
    effective becomes SAIA's ``abort_signal``.
    """

    def __init__(
        self,
        role: Role,
        *,
        saia: SAIA | None = None,
        halt: asyncio.Event | None = None,
        conversation_factory: ConversationFactory | None = None,
        on_iteration: OnIteration | None = None,
        on_complete: OnComplete | None = None,
        on_paused: OnPaused | None = None,
        on_cancelled: OnCancelled | None = None,
        on_failed: OnFailed | None = None,
        on_finally: OnFinally | None = None,
        on_executor_ready: OnExecutorReady | None = None,
        on_cost: OnCost | None = None,
    ) -> None:
        """Initialize a Loop.

        Args:
            role: The role under which this Loop runs. Also determines
                which saia the enclosing flow binds to ``ctx.saia`` when
                ``saia=`` is not supplied.
            saia: Optional explicit SAIA instance. When set, ``Loop`` uses
                it directly and bypasses the enclosing flow's
                :class:`SAIAFactory` for THIS Loop. Intended for consumers
                that own the SAIA + its tool executor externally (e.g.
                wiring per-run state onto the executor after
                construction). Mirrors the ``halt`` precedent —
                construction-time explicit wins over ambient
                ``ctx.saia``.
            halt: Optional explicit halt event. When set, this event
                (not ``ctx.halt``) becomes SAIA's ``abort_signal``.
                Matches the ``ctx.saia`` precedent: explicit at
                construction wins over ambient.
            conversation_factory: Optional
                :class:`~llm_saia.core.conversation.ConversationFactory`.
                When wired, an ``on_paused`` result triggers the
                framework to capture the conversation's
                :meth:`to_dict` payload as canonical bytes on
                :attr:`_paused_bytes` for the halt-observation site
                to stamp into the Flow's CAS commit. Consumers that
                don't need durable pause capture can leave it
                ``None``; capture becomes a no-op.
            on_iteration: Bridges to SAIA's per-turn hook.
            on_complete: Fires after a non-paused ``saia.complete``.
                Non-``None`` return replaces the raw SAIA result as
                :meth:`__call__`'s return.
            on_paused: Fires instead of ``on_complete`` when the result is
                paused. Same return-value semantics.
            on_cancelled: Fires when ``saia.complete`` raises
                :class:`asyncio.CancelledError`; cancellation is re-raised
                after the hook (and after ``on_finally``).
            on_failed: Fires when ``saia.complete`` raises any other
                :class:`Exception`; the exception is re-raised after the
                hook (and after ``on_finally``).
            on_finally: Fires last on every dispatch, regardless of
                outcome. Symmetric with a ``finally:`` block around
                ``await loop(...)``.
            on_executor_ready: Fires on each dispatch after ``ctx.saia``
                is resolved, before ``saia.complete`` — for per-run
                injection into the tool executor.
            on_cost: Fires after ``saia.complete`` for cost accounting
                (runs on both complete and paused results; NOT on the
                cancelled / failed paths since no result exists there).
        """
        self._role = role
        self._saia = saia
        self._halt = halt
        self._conversation_factory = conversation_factory
        self._on_iteration = on_iteration
        self._on_complete = on_complete
        self._on_paused = on_paused
        self._on_cancelled = on_cancelled
        self._on_failed = on_failed
        self._on_finally = on_finally
        self._on_executor_ready = on_executor_ready
        self._on_cost = on_cost
        self._paused_bytes: bytes | None = None

    @property
    def role(self) -> Role:
        """The role this Loop dispatches under.

        Presence of this attribute (of type :class:`Role`) is what makes
        Loop a Buildable — the flow's target validator and materializer
        accept it wherever they accept an ``@verb`` function.
        """
        return self._role

    async def __call__(
        self,
        ctx: Context[Any],
        task: str | None = None,
        *,
        conversation: Any = None,
    ) -> Any:
        """Dispatch one ``saia.complete`` under this Loop's role.

        Args:
            ctx: The dispatching flow's context. ``ctx.saia`` runs
                ``saia.complete``; ``ctx.halt`` is fallback for
                ``abort_signal`` when no explicit halt was given at
                construction.
            task: The task/prompt handed to ``saia.complete``.
                ``None`` is only valid when resuming from a
                ``saia_turn`` checkpoint that carries the saved task —
                :meth:`_prepare_dispatch` restores it before
                :meth:`saia.complete` runs. Any other dispatch with
                ``task=None`` raises ``TypeError``.
            conversation: Optional conversation-like object passed
                through to ``saia.complete`` for prior history. On
                resume the saved conversation overrides this.

        Returns:
            Whatever ``saia.complete`` returns (a ``TaskResult`` in
            SAIA's vocab), UNLESS ``on_complete`` (non-paused path) or
            ``on_paused`` (paused path) returned a non-``None`` value —
            that value replaces the raw result. ``on_finally`` fires
            after either path.

        Raises:
            asyncio.CancelledError: Re-raised after ``on_cancelled`` and
                ``on_finally`` fire.
            Exception: Re-raised after ``on_failed`` and ``on_finally``
                fire.
            RuntimeError: ``ctx.saia`` is ``None`` — either the ctx has
                no role or the enclosing flow had no SAIAFactory.
            TypeError: ``task`` is ``None`` and no resume entry
                supplies one.
        """
        try:
            # Inside the outer try so a raise from _prepare_dispatch (JSON
            # decode fail, ConversationFactory.create_from_state raise, etc.)
            # still fires on_finally per its "runs last on every dispatch"
            # contract. on_failed's scope stays around saia.complete only.
            saia, complete_kwargs, conversation, task = self._prepare_dispatch(
                ctx, task, conversation
            )
            if task is None:
                raise TypeError(
                    f"Loop at node {ctx._node_id!r} dispatched without a task and "
                    "no saia_turn resume entry supplied one."
                )
            if self._on_executor_ready is not None:
                await maybe_await(self._on_executor_ready(saia, ctx))
            result = await self._run_saia_complete(saia, task, complete_kwargs, ctx)
            # saia.complete returned successfully — drop the resume entry now.
            # A raise inside _run_saia_complete leaves it in place so a
            # rescue-then-iterate-retry re-consumes it with resume=True.
            self._release_resume_entry(ctx)
            override = await self._after_run(result, ctx, task, conversation)
            return override if override is not None else result
        finally:
            if self._on_finally is not None:
                await maybe_await(self._on_finally(ctx))

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _require_saia(self, ctx: Context[Any]) -> Any:
        """Return the effective SAIA — explicit ``Loop(saia=X)`` wins over ``ctx.saia``."""
        if self._saia is not None:
            return self._saia
        saia = ctx.saia
        if saia is None:
            raise RuntimeError(
                "Loop requires ctx.saia; dispatch under a Flow with a "
                f"SAIAFactory (ctx.role={ctx.role!r})"
            )
        return saia

    def _resolve_halt(self, ctx: Context[Any]) -> asyncio.Event | None:
        """Explicit ``Loop(halt=X)`` wins over ambient ``ctx.halt``."""
        return self._halt if self._halt is not None else ctx.halt

    def _make_iter_bridge(self, ctx: Context[Any]) -> Callable[[int, Any], Awaitable[None]] | None:
        """Return a SAIA-compatible per-turn bridge, or ``None`` when unwired."""
        hook = self._on_iteration
        if hook is None:
            return None

        async def bridge(iteration: int, response: Any) -> None:
            await maybe_await(hook(iteration, response, ctx))

        return bridge

    async def _run_saia_complete(
        self, saia: Any, task: str, complete_kwargs: dict[str, Any], ctx: Context[Any]
    ) -> Any:
        """Await ``saia.complete`` with the on_cancelled / on_failed hooks wired.

        Extracted from :meth:`__call__` to keep it within the strict
        function-size limit. The two hooks fire against
        ``saia.complete``'s exceptions ONLY; a raise from
        :meth:`_prepare_dispatch` or :meth:`_after_run` does not
        invoke them.
        """
        try:
            return await saia.complete(task, **complete_kwargs)
        except asyncio.CancelledError:
            if self._on_cancelled is not None:
                await maybe_await(self._on_cancelled(ctx))
            raise
        except Exception as exc:
            if self._on_failed is not None:
                await maybe_await(self._on_failed(exc, ctx))
            raise

    async def _after_run(self, result: Any, ctx: Context[Any], task: str, conversation: Any) -> Any:
        """Cost hook, then paused-vs-complete branching + return override.

        On the paused path, capture ``task`` alongside the
        conversation's serialized :meth:`to_dict` payload as
        canonical bytes on :attr:`_paused_bytes` when a
        :class:`ConversationFactory` is wired — the halt-observation
        site reads it to stamp a ``TraceRef(kind="saia_turn", ...)``
        on the Flow's CAS halt commit. Persisting the task lets
        :meth:`_consume_resume_entry` restore both when a direct
        :class:`Loop` chain step resumes at index > 0 with empty
        ``node_args``. Capture is a no-op when no factory is wired
        or when no conversation object flowed through this dispatch.

        Returns the value from ``on_paused`` / ``on_complete`` when
        the hook returned non-``None`` — :meth:`__call__` uses it to
        replace the raw SAIA result. ``None`` means "no override, keep
        raw result".
        """
        if self._on_cost is not None:
            await maybe_await(self._on_cost(result, ctx))
        if getattr(result, "paused", False):
            self._capture_paused(ctx, task, conversation)
            if self._on_paused is not None:
                return await maybe_await(self._on_paused(result, ctx))
            return None
        # Clear this Loop's stale paused bytes on non-paused completion so a
        # later halt-save doesn't stamp conversation state from an earlier
        # pause of this Loop. Sibling Loops' entries stay put.
        env = ctx._env
        if env is not None and ctx._node_id is not None:
            env.pending_saia_turns.remove(ctx._node_id)
        if self._on_complete is not None:
            return await maybe_await(self._on_complete(result, ctx))
        return None

    def _prepare_dispatch(
        self, ctx: Context[Any], task: str | None, conversation: Any
    ) -> tuple[Any, dict[str, Any], Any, str | None]:
        """Resolve saia, build ``saia.complete`` kwargs, return the effective task + conversation.

        On dispatch entry: pop this Loop's stale pending-turn entry
        from the runtime, reset :attr:`_paused_bytes`, and consume
        any resume entry left by :meth:`Flow._hydrate_resume_state`.
        When a resume entry is present, its saved task AND
        reconstructed :class:`Conversation` replace the caller's
        AND ``resume=True`` is added to the ``saia.complete`` kwargs.
        The task-restore lets a direct :class:`Loop` chain step at
        index > 0 resume — Chain's resume contract calls
        the target with empty ``node_args``, so without the saved
        task ``Loop.__call__`` would have none.

        Returns ``(saia, complete_kwargs, effective_conversation,
        effective_task)``. :meth:`__call__` hands the effective
        values to :meth:`saia.complete` and :meth:`_after_run`.
        """
        saia = self._require_saia(ctx)
        self._paused_bytes = None
        env = ctx._env
        if env is not None and ctx._node_id is not None:
            env.pending_saia_turns.remove(ctx._node_id)
        resumed_task, resumed_conversation, is_resume = self._consume_resume_entry(ctx)
        if is_resume:
            task = resumed_task
            conversation = resumed_conversation
        complete_kwargs: dict[str, Any] = {
            "on_iteration": self._make_iter_bridge(ctx),
            "conversation": conversation,
            "abort_signal": self._resolve_halt(ctx),
        }
        if is_resume:
            complete_kwargs["resume"] = True
        return saia, complete_kwargs, conversation, task

    def _consume_resume_entry(self, ctx: Context[Any]) -> tuple[str | None, Any, bool]:
        """Rebuild task + Conversation from this Loop's resume entry, leaving the entry in place.

        Reads ``env.resume_saia_turns`` at
        ``ctx._node_id``. When an entry is present, decodes the
        canonical-json envelope ``{"task": ..., "conversation":
        ...}`` and hands the conversation-state payload to
        :attr:`_conversation_factory`'s ``create_from_state`` to
        reconstruct the paused ``Conversation``.

        Returns ``(saved_task, reconstructed_conversation, True)``
        when a resume entry was found, ``(None, None, False)``
        otherwise. Raises :class:`RuntimeError` when an entry exists
        but no :class:`ConversationFactory` is wired — a fresh
        dispatch would silently drop the SAIA turn the halted commit
        captured.

        The entry is intentionally NOT popped here.
        :meth:`_release_resume_entry` drops it only after
        ``saia.complete`` returns successfully — a raise from
        ``on_executor_ready`` or ``saia.complete`` leaves the entry
        in place so a rescue-then-iterate-retry re-reconstructs the
        same task/conversation instead of falling through to a fresh
        dispatch without ``resume=True``.
        """
        env = ctx._env
        node_id = ctx._node_id
        if env is None or node_id is None:
            return None, None, False
        payload = env.resume_saia_turns.load(node_id)
        if payload is None:
            return None, None, False
        if self._conversation_factory is None:
            raise RuntimeError(
                f"Loop at node {node_id!r} scheduled to resume a paused "
                "SAIA turn but no ConversationFactory is wired; the "
                "reconstructed conversation would be lost. Wire a "
                "ConversationFactory matching the format SAIA used at "
                "save time."
            )
        envelope = SaiaTurnEnvelope.from_bytes(payload)
        conversation = self._conversation_factory.create_from_state(envelope.conversation)
        return envelope.task, conversation, True

    def _release_resume_entry(self, ctx: Context[Any]) -> None:
        """Drop this Loop's resume entry — safe to call unconditionally after ``saia.complete``.

        Popped here (not in :meth:`_consume_resume_entry`) so a raise
        from ``on_executor_ready`` or ``saia.complete`` leaves the
        entry available for a rescue-then-iterate-retry to consume
        again with ``resume=True``.
        """
        env = ctx._env
        node_id = ctx._node_id
        if env is None or node_id is None:
            return
        env.resume_saia_turns.release(node_id)

    def _capture_paused(self, ctx: Context[Any], task: str, conversation: Any) -> None:
        """Serialize the paused task + conversation to canonical bytes.

        Publishes to two seams:

        - :attr:`_paused_bytes` on this Loop instance — introspection
          surface for tests and consumers that already hold a Loop
          reference.
        - ``env.pending_saia_turns`` on the top-level Flow runtime
          — keyed by ``ctx._node_id`` so each Loop's
          bytes stay distinct (concurrent ``.map`` bodies, sibling
          Loops in a chain, and nested Loops in an iterate body all
          share one runtime). The halt-observation site drains it
          to stamp one ``TraceRef(kind="saia_turn", ...)`` per
          entry on the halt commit.

        The blob envelope is ``{"task": <str>, "conversation":
        <to_dict()>}`` — persisting ``task`` alongside the
        conversation state so :meth:`_consume_resume_entry` restores
        both when a direct :class:`Loop` chain step resumes at
        index > 0 with empty ``node_args``.

        No-op when this Loop was constructed without a
        :class:`ConversationFactory`, when no conversation object
        flowed through this dispatch, or when ``ctx._node_id`` is
        unset.
        """
        if self._conversation_factory is None or conversation is None:
            return
        to_dict = getattr(conversation, "to_dict", None)
        if to_dict is None:
            return
        payload = SaiaTurnEnvelope(task=task, conversation=to_dict()).to_bytes()
        self._paused_bytes = payload
        env = ctx._env
        if env is not None and ctx._node_id is not None:
            env.pending_saia_turns.add(ctx._node_id, payload, env.ancestor_chain)


# ----------------------------------------------------------------------------
# LoopFactory
# ----------------------------------------------------------------------------


class LoopFactory:
    """App-scoped factory for :class:`Loop` — captures cross-cutting config once.

    Bundles the ambient logger, :class:`SAIAFactory`, checkpointer, and
    halt event so consumers wire once at the application boundary and
    ``.create(role, **hooks)`` many Loops. Mirrors :class:`FlowFactory`'s
    ``with_saia_factory`` / ``with_halt`` shape so a shared event threads
    uniformly across a mixed Loop-and-Flow tree::

        loop_f = LoopFactory(lg, saia_factory=sf).with_halt(shared_event)
        flow_f = FlowFactory(lg, saia_factory=sf).with_halt(shared_event)

    The ``SAIAFactory`` on this factory is held for future
    standalone-Loop use (not required today — Flow-body Loops read
    ``ctx.saia`` from the enclosing flow's factory). Consumers are free
    to pass ``saia_factory=None`` when they only wire Loops into Flows.
    """

    def __init__(
        self,
        lg: Logger,
        *,
        saia_factory: SAIAFactory | None = None,
        halt: asyncio.Event | None = None,
        conversation_factory: ConversationFactory | None = None,
    ) -> None:
        """Capture the ambient environment for subsequent :meth:`create` calls.

        Args:
            lg: Logger threaded to consumers via this factory's
                :attr:`lg` accessor; also carried forward on the
                ``with_*`` derivations.
            saia_factory: Optional :class:`SAIAFactory`. Reserved for
                standalone-Loop use; Flow-body Loops read ``ctx.saia``
                from the enclosing Flow's factory.
            halt: Optional :class:`asyncio.Event` used as the default
                halt for every built Loop. Per-``create`` overrides
                win (same explicit-wins rule the Loop itself uses).
            conversation_factory: Optional
                :class:`~llm_saia.core.conversation.ConversationFactory`.
                Every :meth:`create` inherits it as the Loop's default
                factory unless per-call overridden. Wire once at the
                app boundary so every Loop this factory builds captures
                paused conversations into the Flow's CAS halt commit.
        """
        self._lg = lg
        self._saia_factory = saia_factory
        self._halt = halt
        self._conversation_factory = conversation_factory

    @property
    def lg(self) -> Logger:
        """The logger captured at construction."""
        return self._lg

    @property
    def saia_factory(self) -> SAIAFactory | None:
        """The SAIAFactory captured at construction, or ``None``."""
        return self._saia_factory

    @property
    def halt(self) -> asyncio.Event | None:
        """The default halt event captured at construction, or ``None``."""
        return self._halt

    @property
    def conversation_factory(self) -> ConversationFactory | None:
        """The default conversation factory captured at construction, or ``None``."""
        return self._conversation_factory

    def create(
        self,
        role: Role,
        *,
        saia: SAIA | None = None,
        halt: asyncio.Event | None = None,
        conversation_factory: ConversationFactory | None = None,
        on_iteration: OnIteration | None = None,
        on_complete: OnComplete | None = None,
        on_paused: OnPaused | None = None,
        on_cancelled: OnCancelled | None = None,
        on_failed: OnFailed | None = None,
        on_finally: OnFinally | None = None,
        on_executor_ready: OnExecutorReady | None = None,
        on_cost: OnCost | None = None,
    ) -> Loop:
        """Build a :class:`Loop` inheriting this factory's captured defaults.

        Per-``create`` ``halt=`` / ``conversation_factory=`` override
        the factory defaults. ``saia=`` pins an explicit SAIA instance
        on the resulting Loop, bypassing the enclosing flow's
        :class:`SAIAFactory`. Hooks are per-Loop and never inherited.
        """
        return Loop(
            role,
            saia=saia,
            halt=halt if halt is not None else self._halt,
            conversation_factory=(
                conversation_factory
                if conversation_factory is not None
                else self._conversation_factory
            ),
            on_iteration=on_iteration,
            on_complete=on_complete,
            on_paused=on_paused,
            on_cancelled=on_cancelled,
            on_failed=on_failed,
            on_finally=on_finally,
            on_executor_ready=on_executor_ready,
            on_cost=on_cost,
        )

    def with_saia_factory(self, saia_factory: SAIAFactory) -> LoopFactory:
        """Return a new :class:`LoopFactory` whose SAIAFactory is swapped."""
        return LoopFactory(
            self._lg,
            saia_factory=saia_factory,
            halt=self._halt,
            conversation_factory=self._conversation_factory,
        )

    def with_halt(self, event: asyncio.Event) -> LoopFactory:
        """Return a new :class:`LoopFactory` whose halt event is swapped.

        Every Loop subsequently built with :meth:`create` inherits
        ``event`` as its default halt (per-``create`` overrides win).
        Wire once at the factory to thread the same halt through an
        entire agent shape — pair with
        :meth:`FlowFactory.with_halt` on the same event so Loops and
        Flows halt in lockstep.
        """
        return LoopFactory(
            self._lg,
            saia_factory=self._saia_factory,
            halt=event,
            conversation_factory=self._conversation_factory,
        )

    def with_conversation_factory(self, conversation_factory: ConversationFactory) -> LoopFactory:
        """Return a new :class:`LoopFactory` whose conversation factory is swapped."""
        return LoopFactory(
            self._lg,
            saia_factory=self._saia_factory,
            halt=self._halt,
            conversation_factory=conversation_factory,
        )
