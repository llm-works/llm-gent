# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost tracking as flow context: the cost tracker each run of a flow runs on.

Cost is what calls and operations cost; a budget is the limit cost is
checked against. Gent tracks cost; the agent decides how to keep to a
budget. A flow's cost context:

- :meth:`Flow.with_cost_tracker` — every run of the flow runs on that
  :class:`~llm_gent.core.cost.CostTracker`;
- :meth:`Flow.with_budget` — each run of the flow runs on a child of its
  tracker (its own, else the enclosing one) with that budget, so spend
  rolls up and every budget on the chain is tracked. A map body runs once
  per item, an iterate body once per pass, a subflow once per ``.call``:
  per-item, per-pass, per-call budgets;
- neither — the run shares the enclosing tracker.

Crossing a budget latches the child's ``exceeded`` / ``urgent_wrapup``
for the agent to read through ``ctx.cost``; it does not stop the run. A
hard stop is the app's choice: a tracker built with ``halt=`` the run's
halt event pauses the run when that tracker's budget is crossed.

The running cost is reconstructed across pause, resume and shortcut: a
tracker's spend (and spend by op) is a position like a cursor, in every
checkpoint taken while it is in use and restored before the run's first
step on resume.

- A budgeted run's child is saved at the run's path
  (:data:`~llm_gent.flow.state.snapshot.COST`) while the run is in
  progress.
- A tracker a flow declares is saved at that flow's path
  (:data:`~llm_gent.flow.state.snapshot.TRACKER`) while its run is in
  progress; the top-level flow's stays through the run's end, so the
  completion commit holds it too and a later session continues the
  total. A tracker a flow inherits — the same object as its parent's —
  is saved once, where it is first declared.

Spend is therefore cumulative over the whole history. A per-session
budget is the app's limit, set on the restored spend (``update_budget``).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.cost import CostTracker
from ._node_id import _child_flows
from .state.snapshot import COST, TRACKER, ScopePath, ScopeRegistry, path_str


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

    ``tracker`` is ``ctx.cost`` for the run; ``halt`` is the halt it
    observes (the run's, from the top-level flow).
    """

    tracker: CostTracker | None
    halt: asyncio.Event | None


@contextlib.asynccontextmanager
async def run_cost(
    flow: Flow,
    path: ScopePath,
    scopes: ScopeRegistry,
    parent_cost: CostTracker | None,
    parent_halt: asyncio.Event | None,
) -> AsyncIterator[RunCost]:
    """The cost context of one run of ``flow`` at ``path``, for the duration of the run.

    The tracker ``flow`` declares (one its parent does not already run on)
    and a budgeted run's child are kept registered at ``path`` — after
    restoring the spend a checked-out snapshot saved there — while the run
    is in progress; a run that stops keeps them, for the run's halt
    checkpoint. The top-level flow's tracker stays registered after the
    run completes, for the completion commit.

    Raises:
        RuntimeError: ``flow`` has a budget and no tracker to make its
            child from (a flow reached outside its parent's composition
            tree, e.g. through ``ctx.flow.dispatch``).
    """
    # The run's halt is on the top-level flow (check_one_halt); a nested run
    # observes what its parent does.
    halt = flow._halt_event if parent_halt is None else parent_halt
    own = flow._cost_tracker
    tracker = own if own is not None else parent_cost
    declared = own if own is not None and own is not parent_cost else None
    with _kept(scopes, path, TRACKER, declared, past_the_run=path == ()):
        if flow._budget is None:
            yield RunCost(tracker, halt)
            return
        if tracker is None:
            raise RuntimeError(_no_tracker_message(flow, flow._budget))
        child = tracker.child(flow._budget)
        with _kept(scopes, path, COST, child, past_the_run=False):
            yield RunCost(child, halt)


@contextlib.contextmanager
def _kept(
    scopes: ScopeRegistry,
    path: ScopePath,
    entry: str,
    tracker: CostTracker | None,
    *,
    past_the_run: bool,
) -> Iterator[None]:
    """Keep ``tracker``'s accounting at ``path`` as ``entry`` while the block runs.

    Restores it first from a checked-out snapshot. Dropped when the block
    completes, unless ``past_the_run``; kept when it stops early (the halt
    checkpoint needs it). Nothing when ``tracker`` is ``None``.
    """
    if tracker is None:
        yield
        return
    _restore(scopes, path, entry, tracker)
    cursor = _TrackerCursor(tracker, entry)
    scopes.open_cursor(path, cursor)
    yield
    if not past_the_run:
        scopes.close_cursor(path, cursor)


def _restore(scopes: ScopeRegistry, path: ScopePath, entry: str, tracker: CostTracker) -> None:
    """Restore ``tracker`` from the spend a checked-out snapshot saved at ``path``, if any."""
    found, saved = scopes.take_cursor(path, entry)
    if not found:
        return
    try:
        tracker.restore(
            float(saved["spent"]), {k: float(v) for k, v in saved["costs_by_op"].items()}
        )
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        where = path_str((*path, entry))
        raise TypeError(f"cursor at {where!r} cannot be restored: {e}") from e


class _TrackerCursor:
    """Cursor of a tracker kept in the run's snapshots: its accounting, as ``entry``."""

    def __init__(self, tracker: CostTracker, entry: str) -> None:
        self.tracker = tracker
        self.entry = entry

    def cursor(self) -> dict[str, Any]:
        """``{entry: {"spent", "costs_by_op"}}``."""
        return {self.entry: self.tracker.snapshot()}
