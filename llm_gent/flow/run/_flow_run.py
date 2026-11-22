# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One run of a top-level flow: check the inputs, start, walk, finish.

The run's own objects (its :class:`~._run.Run`, the root scope, the
position) are made here; the walk itself is :func:`~._executor.run_as_subflow`
from the root.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any, get_args

from ..checkpoint import ResumeMode, Retention, checkpoint_tag, is_commit_hash
from ..flow import Flow
from ..hooks import Decision, HookDenied, RunOutcome
from ..nodes import HALTED, UNSET, Interrupted
from ..resource._runtime import check_resources
from ..resource.cost import check_budgets_have_a_tracker
from ..state import State
from ..state.snapshot import Snapshot
from ..structure import Structure
from ._checkpoint_ctx import CheckpointContext, check_checkpointers
from ._executor import run_as_subflow
from ._hook_points import RunHooks
from ._resume import Resume, apply_clean_exit_retention, commit_halt
from ._run import Run
from ._shortcut import check_shortcuts, check_signals, run_signals
from .scope import Position, Scope, _RunEnv


async def run_flow(
    flow: Flow,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    state: Any = UNSET,
    halt: asyncio.Event | None = None,
    signals: Mapping[str, asyncio.Event] | None = None,
    history: str | None = None,
    retention: Retention = "retain",
    resume: ResumeMode | str = "off",
    restart: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Run ``flow`` as a top-level run; :meth:`Flow.run` documents the arguments."""
    run_signals_ = check_signals(signals)
    _check_run_args(flow, resume, restart, run_signals_, history, retention)
    run = _new_run(flow, halt, run_signals_, history, retention)
    active_state, saved = await _start_state(
        run, lambda: _wrap_top_state(flow, state), resume, restart
    )
    run.scopes.begin(active_state, saved)
    hooks = run.hooks = RunHooks(run)
    run.info = hooks.info
    stopped = await hooks.before()
    if stopped is not None:
        return await _stopped_before_run(hooks, stopped)
    try:
        outcome, value = await _walk_and_finalize(run, active_state, extra, args, kwargs)
    except asyncio.CancelledError as e:
        await hooks.after("cancelled", e)
        raise
    except Exception as e:
        await hooks.after("failed", e)
        raise
    await hooks.after(outcome)
    return value


def _new_run(
    flow: Flow,
    halt: asyncio.Event | None,
    signals: dict[str, asyncio.Event],
    history: str | None,
    retention: Retention,
) -> Run:
    """The state of a new run of ``flow``, with its own writer on ``history``.

    Raises:
        appinfra.cas.InvalidNameError: ``history`` cannot name a repository.
    """
    ctx = None
    if history is not None:
        assert flow._checkpoint_store is not None  # _check_history
        ctx = CheckpointContext(
            flow._checkpoint_store, history, lambda: Structure.of(flow), retention
        )
    return Run(flow=flow, halt=halt, signals=signals, checkpoint_ctx=ctx)


async def _walk_and_finalize(
    run: Run,
    state: State[Any],
    extra: dict[str, Any] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[RunOutcome, Any]:
    """Walk the flow, then commit its halt or apply clean-exit retention.

    Returns ``("halted", HALTED)`` when the halt stopped it, else ``("ok", result)``.
    """
    try:
        result = await _walk_top(run, state, extra, args, kwargs)
    except Interrupted:
        # Everything has stopped, each part registered where it stopped.
        await commit_halt(run)
        return "halted", HALTED
    await apply_clean_exit_retention(run, state)
    return "ok", result


async def _walk_top(
    run: Run,
    state: State[Any],
    extra: dict[str, Any] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Walk the run's flow as its top level, with its signals set up.

    The walk starts from the root scope (the run's state, nothing
    declared yet) at the root position, whose activity is the run.
    """
    info = run.info
    env = _RunEnv(
        run=run,
        lg=run.lg,
        scope=Scope(state=state),
        position=Position(activity_id=None if info is None else info.run_id),
        extra=extra if extra is not None else {},
    )
    with run_signals(run):
        return await run_as_subflow(run.flow, args, kwargs, env)


async def _stopped_before_run(hooks: RunHooks, decision: Decision) -> Any:
    """A ``before_run`` hook stopped the run: return :data:`HALTED` on a pause, else raise.

    Nothing ran and nothing is committed: a paused run started again
    reaches ``before_run`` again.

    Raises:
        HookDenied: A hook denied the run.
    """
    if decision.verdict == "pause":
        await hooks.after("halted")
        return HALTED
    await hooks.after("denied")
    raise HookDenied("before_run", decision)


def _check_run_args(
    flow: Flow,
    resume: ResumeMode | str,
    restart: str | None,
    signals: Mapping[str, asyncio.Event],
    history: str | None,
    retention: Retention,
) -> None:
    """Reject an empty flow, a bad start, a history without a store, a bad shortcut, a lone cap.

    Runs before anything is read or written, so a misconfigured run
    leaves no new history behind.
    """
    if not flow._nodes:
        raise RuntimeError(f"Flow {flow._name!r} has no nodes to run")
    if retention not in get_args(Retention):
        raise ValueError(f"retention must be 'retain' or 'gc_on_success'; got {retention!r}")
    _check_history(flow, history)
    _check_start(flow, resume, restart, history)
    check_checkpointers(flow, history)
    check_shortcuts(flow, signals)
    check_budgets_have_a_tracker(flow)
    check_resources(flow)


def _check_history(flow: Flow, history: str | None) -> None:
    """Reject a ``history`` on a flow without a checkpoint store."""
    if history is not None and flow._checkpoint_store is None:
        raise RuntimeError(
            f"Flow {flow._name or '<anonymous>'!r} was run with history={history!r} but has "
            f"no checkpoint store: build it with checkpoint_store= (Flow or Factory)"
        )


def _check_start(
    flow: Flow, resume: ResumeMode | str, restart: str | None, history: str | None
) -> None:
    """Reject a bad ``resume`` or ``restart``, both together, or either without a history."""
    if resume not in get_args(ResumeMode) and not is_commit_hash(resume):
        _require_checkpoint_name(
            resume,
            f"resume must be one of {get_args(ResumeMode)}, a commit hash or a checkpoint name",
        )
    if restart is not None:
        if resume != "off":
            raise ValueError(
                f"run() takes resume or restart, not both; got resume={resume!r}, "
                f"restart={restart!r}"
            )
        if restart != "latest" and not is_commit_hash(restart):
            _require_checkpoint_name(
                restart, "restart must be 'latest', a commit hash or a checkpoint name"
            )
    start = f"resume={resume!r}" if restart is None else f"restart={restart!r}"
    if (resume != "off" or restart is not None) and history is None:
        label = flow._name or "<anonymous>"
        raise RuntimeError(
            f"Flow {label!r} was run with {start} but no history: pass "
            f"run(history=...) to name the history to start from"
        )


def _require_checkpoint_name(value: Any, what: str) -> None:
    """Raise ``ValueError`` led by ``what`` when ``value`` cannot name a checkpoint."""
    try:
        checkpoint_tag(value)
    except ValueError as e:
        raise ValueError(f"{what}; got {value!r} ({e})") from None


async def _start_state(
    run: Run,
    fallback: Callable[[], State[Any]],
    resume: ResumeMode | str,
    restart: str | None,
) -> tuple[State[Any], Snapshot | None]:
    """The run's initial state and the snapshot it starts from (``None`` for a fresh run).

    ``fallback`` builds the state of a run that checks nothing out: a
    fresh run, or a resume on a history with no commit to continue. A
    run that checks out a commit never calls it, so a state factory
    whose ``new()`` needs inputs still resumes without ``state=``.

    Records on ``run`` the commit it checked out (``None`` for a fresh
    run) for the run's hook events.
    """
    if resume == "off" and restart is None:
        return fallback(), None
    checkout = Resume(run)
    if restart is not None:
        started = await checkout.restart(restart, fallback)
    elif resume == "latest":
        started = await checkout.checkout(fallback)
    else:
        started = await checkout.checkout_at(resume)
    run.resumed_from = checkout.checked_out
    return started


def _wrap_top_state(flow: Flow, state: Any) -> State[Any]:
    """Wrap a top-level ``state=`` payload as :class:`State`.

    Resolves the ``state=UNSET`` sentinel to a fresh state
    (:meth:`Flow._new_state`).
    Explicit ``state=None`` is honored — the payload becomes ``None``.
    Already-wrapped :class:`State` inputs are returned unchanged so a
    caller can thread a run-wide state instance across several runs.
    """
    payload = flow._new_state() if state is UNSET else state
    if isinstance(payload, State):
        return payload
    return State(data=payload, _factory=flow._state_factory)
