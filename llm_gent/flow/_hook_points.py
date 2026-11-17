# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Where Flow calls its hooks: building the events and calling the hooks in effect.

Kept out of the executor, the chain and the Loop so each call site stays
short. A part of a run without hooks (``env.hooks is None``) builds no
events: :meth:`StepHooks.start` returns ``None`` and the other helpers
return at once.

Run events (``before_run`` / ``after_run``) and the run's own checkpoints
(halt, final state) go to the top-level flow's hooks; step, LLM-call and
save-point events go to the hooks in effect where they happen.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ._node_id import iter_flows
from .context import Context
from .hooks import (
    Actor,
    CheckpointEvent,
    Decision,
    HookDenied,
    Hooks,
    LLMCallEvent,
    RunEvent,
    RunInfo,
    RunOutcome,
    StepEvent,
    StepKind,
    StepOutcome,
    TraceIds,
    Where,
)
from .nodes import Interrupted, _Branch, _Iterate, _Map, _Node, _RunEnv, step_name
from .state.cas import FlowCommit
from .state.snapshot import path_str
from .structure import Structure


if TYPE_CHECKING:
    from .flow import Flow


def new_activity_id() -> str:
    """A fresh invocation id."""
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


def step_kind(target: Any) -> StepKind:
    """What a chain step's target is."""
    from .flow import Flow
    from .loop import Loop

    if isinstance(target, Flow):
        return "subflow"
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Iterate):
        return "iterate"
    if isinstance(target, _Map):
        return "map"
    return "loop" if isinstance(target, Loop) else "verb"


def _actor(target: Any, ctx: Context[Any], kind: StepKind) -> Actor:
    """The verb and role a step runs as; composition steps have neither."""
    if kind not in ("verb", "loop"):
        return Actor()
    verb = getattr(target, "__name__", None) or type(target).__name__
    return Actor(verb=verb, role=None if ctx.role is None else ctx.role.name)


def _env_where(env: _RunEnv, node_id: str | None, flow: Flow | None = None) -> Where:
    """Where a step ``node_id`` of the chain under ``env`` is; the chain's own place without one."""
    if node_id is None:
        path = env.ancestor_chain
        return Where(
            flow=flow,
            node_id=path[-1] if path else None,
            node_path=path,
            scope_path=path_str(env.path),
        )
    return Where(
        flow=flow,
        node_id=node_id,
        node_path=(*env.ancestor_chain, node_id),
        scope_path=path_str(env.owner_path(node_id)),
    )


def _run_info(env: _RunEnv) -> RunInfo:
    info = env.runtime._run_info
    assert info is not None, "hooks are in effect, so the run built its RunInfo"
    return info


class StepHooks:
    """The hooks around one execution of a chain step.

    :meth:`start` gives the step its activity id and the environment it
    runs in (whose events belong to it). :meth:`check` calls
    ``before_step``; :meth:`finish` calls ``after_step``.
    """

    def __init__(self, env: _RunEnv, hooks: Hooks, flow: Flow, node: _Node, node_id: str) -> None:
        self.activity_id = new_activity_id()
        self.env = dataclasses.replace(env, activity_id=self.activity_id)
        self.decision: Decision | None = None
        self._hooks = hooks
        self._parent = env.activity_id
        self._target = node.target
        kind = step_kind(node.target)
        self._where = dataclasses.replace(
            _env_where(env, node_id, flow), kind=kind, step_name=step_name(node)
        )
        self._actor = Actor()
        self._ctx: Context[Any] | None = None
        self._inputs: tuple[tuple[Any, ...], dict[str, Any]] = ((), {})

    @classmethod
    def start(cls, env: _RunEnv, flow: Flow, node: _Node, node_id: str) -> StepHooks | None:
        """The hooks around this step, or ``None`` when no hooks are in effect."""
        return None if env.hooks is None else cls(env, env.hooks, flow, node, node_id)

    async def check(self, ctx: Context[Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Call ``before_step``; deny raises :class:`HookDenied`, pause halts the run.

        Raises:
            HookDenied: A hook denied the step.
            Interrupted: A hook paused the run (its halt is set).
            RuntimeError: A hook paused the run, but the run has no halt.
        """
        self._ctx, self._inputs = ctx, (args, kwargs)
        self._actor = _actor(self._target, ctx, self._where.kind or "verb")
        if not self._hooks.has("before_step"):
            return
        decision = await self._hooks.before("before_step", self._event("before_step"))
        if decision.allowed:
            return
        self.decision = decision
        if decision.verdict == "pause":
            _pause(self.env, decision)
        raise HookDenied("before_step", decision)

    async def finish(
        self, outcome: StepOutcome, result: Any = None, error: BaseException | None = None
    ) -> None:
        """Call ``after_step``; a step a hook denied or paused ends ``denied`` / ``paused``.

        A cancelled step stays ``cancelled``; ``decision`` still records the verdict.
        """
        if not self._hooks.has("after_step"):
            return
        if self.decision is not None and outcome != "cancelled":
            outcome = "paused" if self.decision.verdict == "pause" else "denied"
        event = self._event("after_step", result=result, error=error, outcome=outcome)
        await self._hooks.after("after_step", event)

    def _event(self, name: Any, **fields: Any) -> StepEvent:
        args, kwargs = self._inputs
        return StepEvent(
            name=name,
            ts=_now(),
            run=_run_info(self.env),
            where=self._where,
            activity_id=self.activity_id,
            parent_activity_id=self._parent,
            actor=self._actor,
            ctx=self._ctx,
            args=args,
            kwargs=kwargs,
            decision=self.decision,
            **fields,
        )


def _pause(env: _RunEnv, decision: Decision) -> None:
    """Halt the run before the step: set its halt and stop the step as interrupted.

    Raises:
        Interrupted: Always, once the halt is set.
        RuntimeError: The run has no halt to set.
    """
    if env.halt is None:
        raise RuntimeError(
            f"{decision.hook} paused the run before a step, but the run has no halt: "
            f"attach one with Flow.with_halt()"
        )
    env.halt.set()
    raise Interrupted()


def llm_call_hook(
    ctx: Context[Any], conversation: Any, verb: str, role: str | None
) -> Callable[[Any], Awaitable[None]] | None:
    """An ``after_llm_call`` caller for one Loop call, or ``None`` when no hook wants it.

    ``conversation`` is the one SAIA appends the turn to: when the hook
    runs it holds what the model was sent, before the response. ``verb``
    and ``role`` name the Loop making the calls.
    """
    env = ctx._env
    if env is None or env.hooks is None or not env.hooks.has("after_llm_call"):
        return None
    hooks, where = env.hooks, _env_where(env, ctx._node_id)

    async def call(response: Any) -> None:
        event = LLMCallEvent(
            name="after_llm_call",
            ts=_now(),
            run=_run_info(env),
            where=where,
            activity_id=new_activity_id(),
            parent_activity_id=env.activity_id,
            actor=Actor(verb=verb, role=role, model=getattr(response, "model", None)),
            trace=TraceIds(call_id=getattr(response, "call_id", None) or None),
            ctx=ctx,
            conversation=conversation,
            response=response,
        )
        await hooks.after("after_llm_call", event)

    return call


async def checkpoint_written(
    hooks: Hooks | None,
    env: _RunEnv | None,
    run: RunInfo | None,
    commit: FlowCommit,
    node_id: str | None = None,
) -> None:
    """Call ``after_checkpoint`` for ``commit``; a no-op without hooks that want it."""
    if hooks is None or run is None or not hooks.has("after_checkpoint"):
        return
    where = Where() if env is None else _env_where(env, node_id)
    event = CheckpointEvent(
        name="after_checkpoint",
        ts=_now(),
        run=run,
        where=where,
        activity_id=new_activity_id(),
        parent_activity_id=run.run_id if env is None else env.activity_id,
        commit_id=commit.id,
        outcome=commit.meta.outcome,
    )
    await hooks.after("after_checkpoint", event)


class RunHooks:
    """The hooks of one :meth:`Flow.run`: builds its :class:`RunInfo`, calls the run events.

    The run's info exists when any flow of the run has hooks, so hooks
    attached only to subflows still get it; ``before_run`` / ``after_run``
    go to the top-level flow's hooks.
    """

    def __init__(self, flow: Flow, resumed_from: str | None) -> None:
        self.hooks: Hooks | None = Hooks(flow._lg, flow._hooks) if flow._hooks else None
        self.info: RunInfo | None = None
        if any(f._hooks for f in iter_flows(flow)):
            ctx = flow._checkpoint_ctx
            self.info = RunInfo(
                run_id=new_activity_id(),
                flow_name=flow._name,
                structure_hash=Structure.of(flow).hash,
                repo=None if ctx is None else ctx.client_flow_id,
                resumed_from=resumed_from,
            )
        self._where = Where(flow=flow)

    async def before(self) -> Decision | None:
        """Call ``before_run``; the decision when a hook stopped the run, else ``None``."""
        if self.hooks is None or not self.hooks.has("before_run"):
            return None
        decision = await self.hooks.before("before_run", self._event("before_run"))
        return None if decision.allowed else decision

    async def after(self, outcome: RunOutcome, error: BaseException | None = None) -> None:
        """Call ``after_run``."""
        if self.hooks is None or not self.hooks.has("after_run"):
            return
        await self.hooks.after("after_run", self._event("after_run", outcome=outcome, error=error))

    def _event(self, name: Any, **fields: Any) -> RunEvent:
        assert self.info is not None
        return RunEvent(
            name=name,
            ts=_now(),
            run=self.info,
            where=self._where,
            activity_id=self.info.run_id,
            **fields,
        )
