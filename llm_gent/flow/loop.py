# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Loop — Flow-body primitive wrapping one saia.complete() invocation.

A ``Loop`` is a Flow-body citizen: it carries a :class:`Role` and is
dispatched by a :class:`Flow` like an ``@verb`` function or a bound-method
verb. Its body is a single ``saia.complete(...)`` invocation, wired with:

- lifecycle hooks (``on_start`` / ``on_resume`` / ``on_iteration`` /
  ``on_complete`` / ``on_executor_ready`` / ``on_cost``)
- an optional checkpointer seam (3-method Protocol mirroring
  :class:`appware.CheckpointStore`) — ``Loop`` loads at start and deletes
  on successful completion; save timing is the consumer's responsibility
  (via ``on_iteration``, closing over the checkpointer they gave the Loop)
- halt bridging to SAIA's ``abort_signal``

Halt resolution rule: an explicit ``Loop(halt=X)`` at construction wins
over ambient ``ctx.halt`` — matches the ``ctx.saia`` precedent. Whichever
is effective becomes SAIA's ``abort_signal``.

:class:`LoopFactory` bundles the cross-cutting config (logger, SAIAFactory,
checkpointer, halt) so consumers wire once at the app boundary and
``.create(role, **hooks)`` many Loops. It mirrors :class:`FlowFactory`'s
shape so a shared halt event threads uniformly across a mixed Loop-and-Flow
tree.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from appinfra.log import Logger

from .context import Context
from .factory import SAIAFactory
from .role import Role


# ----------------------------------------------------------------------------
# CheckpointStore Protocol
# ----------------------------------------------------------------------------


class CheckpointStore(Protocol):
    """3-method Protocol for persisting Loop checkpoints.

    Mirrors :class:`appware.CheckpointStore` (``save_checkpoint`` /
    ``load_checkpoint`` / ``delete_checkpoint``) so an existing consumer
    store composes without adaptation. Update-in-place is intentionally
    absent — no consumer today mutates a checkpoint atomically, so the
    minimal contract holds.

    :class:`Loop` drives load-at-start and delete-on-successful-completion;
    the save timing is the consumer's responsibility (typically wired
    inside ``on_iteration`` so each turn's state is persisted before the
    next).
    """

    def save_checkpoint(self, scope_id: str, run_id: int, state: dict[str, Any]) -> None:
        """Persist a snapshot for later resume."""
        ...

    def load_checkpoint(self, scope_id: str, run_id: int | None = None) -> dict[str, Any] | None:
        """Return the snapshot, or ``None`` if no matching checkpoint exists.

        ``run_id=None`` should return the latest checkpoint under
        ``scope_id`` per the consumer's convention.
        """
        ...

    def delete_checkpoint(self, scope_id: str, run_id: int | None = None) -> None:
        """Delete the snapshot — called after a successful (non-paused) run."""
        ...


# ----------------------------------------------------------------------------
# Hook types
# ----------------------------------------------------------------------------


OnStart = Callable[[Context], Any]
"""``(ctx) -> None`` — fires before :meth:`saia.complete` when not resuming.

May be async. Return value is ignored.
"""

OnResume = Callable[[dict[str, Any], Context], Any]
"""``(checkpoint_state, ctx) -> None`` — fires when a checkpoint was loaded.

Runs instead of ``on_start``. Consumer decides how to hydrate the run
from ``checkpoint_state``. May be async; return value ignored.
"""

OnIteration = Callable[[int, Any, Context], Any]
"""``(iteration, response, ctx) -> None`` — bridges to SAIA's per-turn hook.

The second positional is the raw SAIA ``ChatResponse`` for that turn
— consumers wire per-turn narration and checkpoint save here. May be
async; return value ignored.
"""

OnComplete = Callable[[Any, Context], Any]
"""``(saia_result, ctx) -> None`` — fires after :meth:`saia.complete` returns non-paused.

Skipped when the run paused. May be async; return value ignored.
"""

OnExecutorReady = Callable[[Any, Context], Any]
"""``(saia, ctx) -> None`` — fires once after ``ctx.saia`` is resolved.

Runs after the enclosing flow builds a role-bound saia and before
:meth:`saia.complete`. Consumers reach into the saia instance's tool
executor to inject per-run values that aren't known at
saia-factory-construction time (``run_config``, ``campaign_id``,
``budget``, etc.). May be async; return value ignored.
"""

OnCost = Callable[[Any, Context], Any]
"""``(saia_result, ctx) -> None`` — fires after :meth:`saia.complete` for cost accounting.

Distinct from ``on_complete``: resource cost is a separate concern from
lifecycle, and runs even on a paused result. Consumers typically inspect
``result.trace`` / token counts here. May be async; return value ignored.
"""


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if awaitable; return it as-is otherwise."""
    if inspect.isawaitable(value):
        return await value
    return value


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
        halt: asyncio.Event | None = None,
        checkpointer: CheckpointStore | None = None,
        on_start: OnStart | None = None,
        on_resume: OnResume | None = None,
        on_iteration: OnIteration | None = None,
        on_complete: OnComplete | None = None,
        on_executor_ready: OnExecutorReady | None = None,
        on_cost: OnCost | None = None,
    ) -> None:
        """Initialize a Loop.

        Args:
            role: The role under which this Loop runs. Determines which
                saia the enclosing flow binds to ``ctx.saia``.
            halt: Optional explicit halt event. When set, this event
                (not ``ctx.halt``) becomes SAIA's ``abort_signal``.
                Matches the ``ctx.saia`` precedent: explicit at
                construction wins over ambient.
            checkpointer: Optional 3-method store. When set, Loop
                loads-at-start (for resume) and deletes-on-complete
                (only on non-paused results); save timing is the
                consumer's, wired through ``on_iteration``.
            on_start: Fires before ``saia.complete`` when not resuming.
            on_resume: Fires instead of ``on_start`` when a checkpoint
                was loaded — receives the loaded state.
            on_iteration: Bridges to SAIA's per-turn hook.
            on_complete: Fires after a non-paused ``saia.complete``.
            on_executor_ready: Fires once after ``ctx.saia`` is
                resolved, before ``saia.complete`` — for per-run
                injection into the tool executor.
            on_cost: Fires after ``saia.complete`` for cost accounting
                (runs even on paused results).
        """
        self._role = role
        self._halt = halt
        self._checkpointer = checkpointer
        self._on_start = on_start
        self._on_resume = on_resume
        self._on_iteration = on_iteration
        self._on_complete = on_complete
        self._on_executor_ready = on_executor_ready
        self._on_cost = on_cost

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
        ctx: Context,
        task: str,
        *,
        scope_id: str | None = None,
        run_id: int | None = None,
        conversation: Any = None,
    ) -> Any:
        """Dispatch one ``saia.complete`` under this Loop's role.

        Args:
            ctx: The dispatching flow's context. ``ctx.saia`` runs
                ``saia.complete``; ``ctx.halt`` is fallback for
                ``abort_signal`` when no explicit halt was given at
                construction.
            task: The task/prompt handed to ``saia.complete``.
            scope_id: Checkpoint scope id — identifies a resumable
                trajectory. Required for checkpoint load/delete;
                omitted → the checkpointer is not consulted regardless
                of its wiring.
            run_id: Checkpoint run id — scopes to a specific attempt.
                ``None`` → load the latest checkpoint under
                ``scope_id`` (per the store's convention).
            conversation: Optional conversation-like object passed
                through to ``saia.complete`` for prior history.

        Returns:
            Whatever ``saia.complete`` returns (a ``TaskResult`` in
            SAIA's vocab).

        Raises:
            RuntimeError: ``ctx.saia`` is ``None`` — either the ctx has
                no role or the enclosing flow had no SAIAFactory.
        """
        saia = self._require_saia(ctx)
        checkpoint = self._load_checkpoint(scope_id, run_id)
        await self._before_run(saia, ctx, checkpoint)
        result = await saia.complete(
            task,
            on_iteration=self._make_iter_bridge(ctx),
            conversation=conversation,
            abort_signal=self._resolve_halt(ctx),
            resume=checkpoint is not None,
        )
        await self._after_run(result, ctx, scope_id, run_id)
        return result

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    @staticmethod
    def _require_saia(ctx: Context) -> Any:
        """Return ``ctx.saia`` or raise an informative error."""
        saia = ctx.saia
        if saia is None:
            raise RuntimeError(
                "Loop requires ctx.saia; dispatch under a Flow with a "
                f"SAIAFactory (ctx.role={ctx.role!r})"
            )
        return saia

    def _resolve_halt(self, ctx: Context) -> asyncio.Event | None:
        """Explicit ``Loop(halt=X)`` wins over ambient ``ctx.halt``."""
        return self._halt if self._halt is not None else ctx.halt

    def _load_checkpoint(self, scope_id: str | None, run_id: int | None) -> dict[str, Any] | None:
        """Return the checkpoint state, or ``None`` when not consulted."""
        if self._checkpointer is None or scope_id is None:
            return None
        return self._checkpointer.load_checkpoint(scope_id, run_id)

    async def _before_run(self, saia: Any, ctx: Context, checkpoint: dict[str, Any] | None) -> None:
        """Fire ``on_executor_ready`` and the start/resume lifecycle hook."""
        if self._on_executor_ready is not None:
            await _maybe_await(self._on_executor_ready(saia, ctx))
        if checkpoint is not None:
            if self._on_resume is not None:
                await _maybe_await(self._on_resume(checkpoint, ctx))
        elif self._on_start is not None:
            await _maybe_await(self._on_start(ctx))

    def _make_iter_bridge(self, ctx: Context) -> Callable[[int, Any], Awaitable[None]] | None:
        """Return a SAIA-compatible per-turn bridge, or ``None`` when unwired."""
        hook = self._on_iteration
        if hook is None:
            return None

        async def bridge(iteration: int, response: Any) -> None:
            await _maybe_await(hook(iteration, response, ctx))

        return bridge

    async def _after_run(
        self,
        result: Any,
        ctx: Context,
        scope_id: str | None,
        run_id: int | None,
    ) -> None:
        """Cost hook (always), then checkpoint delete + ``on_complete`` on non-paused."""
        if self._on_cost is not None:
            await _maybe_await(self._on_cost(result, ctx))
        if getattr(result, "paused", False):
            return
        if self._checkpointer is not None and scope_id is not None:
            self._checkpointer.delete_checkpoint(scope_id, run_id)
        if self._on_complete is not None:
            await _maybe_await(self._on_complete(result, ctx))


# ----------------------------------------------------------------------------
# LoopFactory
# ----------------------------------------------------------------------------


class LoopFactory:
    """App-scoped factory for :class:`Loop` — captures cross-cutting config once.

    Bundles the ambient logger, :class:`SAIAFactory`, checkpointer, and
    halt event so consumers wire once at the application boundary and
    ``.create(role, **hooks)`` many Loops. Mirrors :class:`FlowFactory`'s
    ``with_saia_f`` / ``with_halt`` shape so a shared event threads
    uniformly across a mixed Loop-and-Flow tree::

        loop_f = LoopFactory(lg, saia_f=saia_f).with_halt(shared_event)
        flow_f = FlowFactory(lg, saia_f=saia_f).with_halt(shared_event)

    The ``SAIAFactory`` on this factory is held for future
    standalone-Loop use (not required today — Flow-body Loops read
    ``ctx.saia`` from the enclosing flow's factory). Consumers are free
    to pass ``saia_f=None`` when they only wire Loops into Flows.
    """

    def __init__(
        self,
        lg: Logger,
        *,
        saia_f: SAIAFactory | None = None,
        checkpointer: CheckpointStore | None = None,
        halt: asyncio.Event | None = None,
    ) -> None:
        """Capture the ambient environment for subsequent :meth:`create` calls.

        Args:
            lg: Logger threaded to consumers via this factory's
                :attr:`lg` accessor; also carried forward on the
                ``with_*`` derivations.
            saia_f: Optional :class:`SAIAFactory`. Reserved for
                standalone-Loop use; Flow-body Loops read ``ctx.saia``
                from the enclosing Flow's factory.
            checkpointer: Optional :class:`CheckpointStore`. Every
                :meth:`create` inherits it as the Loop's default
                checkpointer unless per-call overridden.
            halt: Optional :class:`asyncio.Event` used as the default
                halt for every built Loop. Per-``create`` overrides
                win (same explicit-wins rule the Loop itself uses).
        """
        self._lg = lg
        self._saia_f = saia_f
        self._checkpointer = checkpointer
        self._halt = halt

    @property
    def lg(self) -> Logger:
        """The logger captured at construction."""
        return self._lg

    @property
    def saia_f(self) -> SAIAFactory | None:
        """The SAIAFactory captured at construction, or ``None``."""
        return self._saia_f

    def create(
        self,
        role: Role,
        *,
        halt: asyncio.Event | None = None,
        checkpointer: CheckpointStore | None = None,
        on_start: OnStart | None = None,
        on_resume: OnResume | None = None,
        on_iteration: OnIteration | None = None,
        on_complete: OnComplete | None = None,
        on_executor_ready: OnExecutorReady | None = None,
        on_cost: OnCost | None = None,
    ) -> Loop:
        """Build a :class:`Loop` inheriting this factory's captured defaults.

        Per-``create`` ``halt=`` / ``checkpointer=`` override the factory
        defaults (same explicit-wins rule the Loop itself uses). Hooks
        are per-Loop and never inherited.
        """
        return Loop(
            role,
            halt=halt if halt is not None else self._halt,
            checkpointer=(checkpointer if checkpointer is not None else self._checkpointer),
            on_start=on_start,
            on_resume=on_resume,
            on_iteration=on_iteration,
            on_complete=on_complete,
            on_executor_ready=on_executor_ready,
            on_cost=on_cost,
        )

    def with_saia_f(self, saia_f: SAIAFactory) -> LoopFactory:
        """Return a new :class:`LoopFactory` whose SAIAFactory is swapped."""
        return LoopFactory(
            self._lg,
            saia_f=saia_f,
            checkpointer=self._checkpointer,
            halt=self._halt,
        )

    def with_checkpointer(self, checkpointer: CheckpointStore | None) -> LoopFactory:
        """Return a new :class:`LoopFactory` whose checkpointer is swapped."""
        return LoopFactory(
            self._lg,
            saia_f=self._saia_f,
            checkpointer=checkpointer,
            halt=self._halt,
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
            saia_f=self._saia_f,
            checkpointer=self._checkpointer,
            halt=event,
        )
