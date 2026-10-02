# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost tracking as flow context: the cost tracker each run of a flow runs on.

Cost is what calls and operations cost; a budget is the limit cost is
checked against. A flow's cost context:

- :meth:`Flow.with_cost_tracker` — every run of the flow runs on that
  :class:`~llm_gent.core.cost.CostTracker`;
- :meth:`Flow.with_budget` — each run of the flow runs on a child of its
  tracker (its own, else the enclosing one) with that budget, so spend
  rolls up and every budget on the chain applies. A map body runs once per
  item, an iterate body once per pass, a subflow once per ``.call``:
  per-item, per-pass, per-call budgets;
- neither — the run shares the enclosing tracker.

A run with a budget stops when its budget is crossed, with halt
semantics, on a halt event of its own (:class:`RunCost`); that event is
also set when the enclosing halt is, so the run stops on either. A run
whose own budget stopped it ends with no result and the enclosing flow
carries on (:meth:`Flow._walk`).

A run's own tracker (its explicit one, or its budgeted child) is a
position like a cursor: its spend is in every checkpoint taken while the
run is in progress, at the run's path
(:data:`~llm_gent.flow.state.snapshot.COST`), and restored when the run
resumes, before its first step.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.cost import CostTracker
from ._node_id import _child_flows
from .state.snapshot import COST, ScopePath, ScopeRegistry, path_str


if TYPE_CHECKING:
    from .flow import Flow


def check_budget(budget: object) -> float:
    """``budget`` when :meth:`Flow.with_budget` accepts it: a finite number > 0.

    Raises:
        TypeError: Not a number.
        ValueError: Not finite and > 0.
    """
    if isinstance(budget, bool) or not isinstance(budget, int | float):
        raise TypeError(f"with_budget takes a number; got {type(budget).__name__}")
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError(f"a budget must be finite and > 0; got {budget!r}")
    return float(budget)


def check_cost_tracker(tracker: object) -> CostTracker:
    """``tracker`` when :meth:`Flow.with_cost_tracker` accepts it.

    Raises:
        TypeError: Not a :class:`CostTracker`.
    """
    if not isinstance(tracker, CostTracker):
        raise TypeError(f"with_cost_tracker takes a CostTracker; got {type(tracker).__name__}")
    return tracker


def check_budgets_have_a_tracker(root: Flow) -> None:
    """Raise when a flow in ``root``'s composition tree has a budget and no tracker for it.

    A budget makes a child of the flow's tracker (its own, else the
    enclosing one), which supplies the pricing; without one there is
    nothing to make it from.

    Raises:
        RuntimeError: A flow with ``with_budget(...)`` and no
            ``with_cost_tracker(...)`` on it or any flow enclosing it.
    """
    seen: set[tuple[int, bool]] = set()
    stack: list[tuple[Flow, bool]] = [(root, False)]
    while stack:
        flow, covered = stack.pop()
        if (id(flow), covered) in seen:
            continue
        seen.add((id(flow), covered))
        covered = covered or flow._cost_tracker is not None
        if flow._budget is not None and not covered:
            raise RuntimeError(_no_tracker_message(flow, flow._budget))
        covered = covered or flow._budget is not None
        for node in flow._nodes:
            stack.extend((child, covered) for _, child in _child_flows(node))


def _no_tracker_message(flow: Flow, budget: float) -> str:
    """The error for a flow with a budget and no cost tracker to make its child from."""
    label = flow._name or "<anonymous>"
    return (
        f"Flow {label!r} has with_budget({budget}) but no cost tracker: attach a "
        f"CostTracker with with_cost_tracker(tracker) to it or an enclosing flow "
        f"(e.g. the top-level one)"
    )


@dataclass(frozen=True)
class RunCost:
    """The cost context one run of a flow runs under.

    ``tracker`` is ``ctx.cost`` for the run. ``halt`` is the event the
    run observes. ``stop`` is the run's own halt event when it has a
    budget (``halt`` is then ``stop``), set on crossing the budget or when
    ``enclosing`` — the halt the run would observe without one — is set.
    """

    tracker: CostTracker | None
    halt: asyncio.Event | None
    stop: asyncio.Event | None = None
    enclosing: asyncio.Event | None = None

    def stopped_by_budget(self) -> bool:
        """True when the run's own stop is set and the enclosing halt is not."""
        if self.stop is None or not self.stop.is_set():
            return False
        return self.enclosing is None or not self.enclosing.is_set()


@contextlib.asynccontextmanager
async def run_cost(
    flow: Flow,
    path: ScopePath,
    scopes: ScopeRegistry,
    parent_cost: CostTracker | None,
    parent_halt: asyncio.Event | None,
) -> AsyncIterator[RunCost]:
    """The cost context of one run of ``flow`` at ``path``, for the duration of the run.

    A run with a tracker of its own keeps it registered at ``path`` — after
    restoring the spend a checked-out snapshot saved there — and drops it
    once the run completes; a run that stops keeps it, for the run's halt
    checkpoint. A budgeted run's link to the enclosing halt ends with the
    run.

    Raises:
        RuntimeError: ``flow`` has a budget and no tracker to make its
            child from (a flow reached outside its parent's composition
            tree, e.g. through ``ctx.flow.dispatch``).
    """
    # The run's halt is on the top-level flow (check_one_halt); a nested run
    # observes what its parent does, which under a budget is the budget's stop.
    enclosing = flow._halt_event if parent_halt is None else parent_halt
    tracker = flow._cost_tracker if flow._cost_tracker is not None else parent_cost
    if flow._budget is None:
        if flow._cost_tracker is None:
            yield RunCost(parent_cost, enclosing)
            return
        context = RunCost(flow._cost_tracker, enclosing)
    else:
        if tracker is None:
            raise RuntimeError(_no_tracker_message(flow, flow._budget))
        stop = asyncio.Event()
        context = RunCost(tracker.child(flow._budget, halt=stop), stop, stop, enclosing)
    assert context.tracker is not None
    _restore(scopes, path, context.tracker)
    cursor = _CostCursor(context.tracker)
    scopes.open_cursor(path, cursor)
    link = follow(enclosing, context.stop)
    try:
        yield context
        scopes.close_cursor(path, cursor)
    finally:
        if link is not None:
            link.cancel()


def _restore(scopes: ScopeRegistry, path: ScopePath, tracker: CostTracker) -> None:
    """Restore ``tracker`` from the spend a checked-out snapshot saved at ``path``, if any."""
    found, saved = scopes.take_cursor(path, COST)
    if not found:
        return
    try:
        tracker.restore(
            float(saved["spent"]), {k: float(v) for k, v in saved["costs_by_op"].items()}
        )
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        where = path_str((*path, COST))
        raise TypeError(f"cursor at {where!r} cannot be restored: {e}") from e


def follow(
    enclosing: asyncio.Event | None, stop: asyncio.Event | None
) -> asyncio.Task[None] | None:
    """Set ``stop`` once ``enclosing`` is set; the task doing it, ``None`` when nothing to do."""
    if enclosing is None or stop is None:
        return None
    if enclosing.is_set():
        stop.set()
        return None

    async def follow() -> None:
        await enclosing.wait()
        stop.set()

    return asyncio.create_task(follow())


class _CostCursor:
    """Cursor of a run's own tracker: its accounting, restored when the run resumes."""

    def __init__(self, tracker: CostTracker) -> None:
        self.tracker = tracker

    def cursor(self) -> dict[str, Any]:
        """``{"cost": {"spent", "costs_by_op"}}``."""
        return {COST: self.tracker.snapshot()}
