# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One run of a top-level flow: check the inputs, start, walk, finish.

:func:`start_run` checks the inputs and makes the run's own objects (its
:class:`~._run.Run`, the root scope, the position), then executes it in a
task; the walk itself is :func:`~._executor.run_as_subflow` from the root.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_args

from appinfra.cas import Store
from appinfra.log import Logger

from ...core.cost import Tracker as CostTracker
from ...core.env import Environment
from ..checkpoint import Retention
from ..flow import Flow
from ..hooks import Decision, HookDenied, RunOutcome
from ..nodes import HALTED, UNSET, Interrupted
from ..resource._runtime import check_resources
from ..resource.cost import COST, check_budgets_have_a_tracker, check_cost_tracker
from ..state import State
from ..state.snapshot import Snapshot
from ..structure import Structure
from ._checkpoint_ctx import CheckpointContext, check_checkpointers
from ._executor import run_as_subflow
from ._hook_points import RunHooks
from ._resume import Checkout, apply_clean_exit_retention, commit_halt
from ._run import Run
from ._shortcut import check_shortcuts, check_signals, run_signals
from .scope import Position, Scope, _Frame
from .start import FRESH, Fresh, Restart, Resume, Start


@dataclass(frozen=True)
class Wiring:
    """What every run a runner starts shares: its logger, its environment, its history store."""

    lg: Logger
    env: Environment
    store: Store | None


def start_run(
    wiring: Wiring,
    flow: Flow,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    state: Any = UNSET,
    cost: CostTracker | None = None,
    halt: asyncio.Event | None = None,
    signals: Mapping[str, asyncio.Event] | None = None,
    history: str | None = None,
    retention: Retention = "retain",
    start: Start = FRESH,
    extra: dict[str, Any] | None = None,
) -> Run:
    """Check a run of ``flow`` and start it in a task; :meth:`FlowRunner.start` documents it.

    Every check runs here, before anything is read or written, so a
    misconfigured run raises from this call and leaves no history behind.

    Raises:
        RuntimeError: No running event loop, besides the input checks.
    """
    asyncio.get_running_loop()
    run_signals_ = check_signals(signals)
    if cost is not None:
        check_cost_tracker(cost)
    _check_run_args(wiring, flow, start, run_signals_, history, retention, cost is not None)
    run = _new_run(wiring, flow, halt, run_signals_, history, retention)
    # The run's own tracker: its spend rolls up to ``cost``, its accounting is the run's.
    run.cost = None if cost is None else cost.child()
    # The checks passed: from here on the runs share the flow's structure.
    flow._freeze()
    run.task = asyncio.create_task(
        _execute(run, state, start, extra, args, kwargs),
        name=f"flow-run {flow._name or '<anonymous>'}",
    )
    return run


async def _execute(
    run: Run,
    state: Any,
    start: Start,
    extra: dict[str, Any] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Start ``run`` from its state, walk it with its hooks, and return its result."""
    flow = run.flow
    active_state, saved = await _start_state(run, lambda: _wrap_top_state(flow, state), start)
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
    wiring: Wiring,
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
        assert wiring.store is not None  # _check_history
        ctx = CheckpointContext(wiring.store, history, lambda: Structure.of(flow), retention)
    return Run(
        flow=flow,
        lg=wiring.lg,
        env=wiring.env,
        halt_event=halt,
        signals=signals,
        checkpoint_ctx=ctx,
    )


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
    frame = _Frame(
        run=run,
        lg=run.lg,
        scope=Scope(state=state),
        position=Position(activity_id=None if info is None else info.run_id),
        extra=extra if extra is not None else {},
    )
    with run_signals(run):
        return await run_as_subflow(run.flow, args, kwargs, frame)


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
    wiring: Wiring,
    flow: Flow,
    start: Start,
    signals: Mapping[str, asyncio.Event],
    history: str | None,
    retention: Retention,
    tracked: bool,
) -> None:
    """Reject an empty flow, a bad start, a history without a store, a bad shortcut, a lone cap.

    ``tracked``: the run was started with a cost tracker. Runs before
    anything is read or written, so a misconfigured run leaves no new
    history behind.
    """
    if not flow._nodes:
        raise RuntimeError(f"Flow {flow._name!r} has no nodes to run")
    if retention not in get_args(Retention):
        raise ValueError(f"retention must be 'retain' or 'gc_on_success'; got {retention!r}")
    _check_history(wiring, flow, history)
    _check_start(flow, start, history)
    check_checkpointers(flow, history)
    check_shortcuts(flow, signals)
    check_budgets_have_a_tracker(flow, tracked=tracked)
    check_resources(flow, provided=frozenset({COST}) if tracked else frozenset())


def _check_history(wiring: Wiring, flow: Flow, history: str | None) -> None:
    """Reject a ``history`` on a runner without a store to keep it in."""
    if history is not None and wiring.store is None:
        raise RuntimeError(
            f"Flow {flow._name or '<anonymous>'!r} was started with history={history!r} but "
            f"the runner has no store: build it with FlowRunner(lg, store=...)"
        )


def _check_start(flow: Flow, start: Start, history: str | None) -> None:
    """Reject a ``start`` that is not a :data:`Start`, or one from a commit without a history."""
    if not isinstance(start, Fresh | Resume | Restart):
        raise TypeError(
            f"start= takes Fresh(), Resume(at) or Restart(at); got {type(start).__name__}"
        )
    if not isinstance(start, Fresh) and history is None:
        label = flow._name or "<anonymous>"
        raise RuntimeError(
            f"Flow {label!r} was started with start={start!r} but no history: pass "
            f"history= to name the history to start from"
        )


async def _start_state(
    run: Run, fallback: Callable[[], State[Any]], start: Start
) -> tuple[State[Any], Snapshot | None]:
    """The run's initial state and the snapshot it starts from (``None`` for a fresh run).

    ``fallback`` builds the state of a run that checks nothing out: a
    fresh run, or a resume on a history with no commit to continue. A
    run that checks out a commit never calls it, so a state factory
    whose ``new()`` needs inputs still resumes without ``state=``.

    Records on ``run`` the commit it checked out (``None`` for a fresh
    run) for the run's hook events.
    """
    if isinstance(start, Fresh):
        return fallback(), None
    checkout = Checkout(run)
    if isinstance(start, Restart):
        started = await checkout.restart(start.at, fallback)
    elif start.at == "latest":
        started = await checkout.checkout(fallback)
    else:
        started = await checkout.checkout_at(start.at)
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
