# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Budgets as flow context: the tracker each run of a flow runs on.

:meth:`Flow.with_budget` sets a flow's budget context:

- a :class:`~llm_gent.core.budget.Tracker` — every run of the flow runs on
  that tracker;
- a cap (``float``) — each run of the flow runs on a child of the
  enclosing tracker with that cap, so spend rolls up and every cap on the
  chain applies. A map body runs once per item, an iterate body once per
  pass, a subflow once per ``.call``: per-item, per-pass, per-call caps;
- nothing — the run shares the enclosing tracker.

A capped run stops when its cap is crossed, with halt semantics, on a halt
event of its own (:class:`RunBudget`); that event is also set when the
enclosing halt is, so the run stops on either. A run whose own budget
stopped it ends with no result and the enclosing flow carries on
(:meth:`Flow._walk`).

A run's own tracker (its explicit one, or its capped child) is a position
like a cursor: its spend is in every checkpoint taken while the run is in
progress, at the run's path (:data:`~llm_gent.flow.state.snapshot.BUDGET`),
and restored when the run resumes, before its first step.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.budget import Tracker
from ._node_id import _child_flows
from .state.snapshot import BUDGET, ScopePath, ScopeRegistry, path_str


if TYPE_CHECKING:
    from .flow import Flow


Budget = Tracker | float
"""What :meth:`Flow.with_budget` takes: an explicit tracker, or a cap for a child per run."""


def check_budget(budget: object) -> Budget:
    """``budget`` when :meth:`Flow.with_budget` accepts it.

    Raises:
        TypeError: Neither a :class:`Tracker` nor a number.
        ValueError: A cap that is not finite and > 0.
    """
    if isinstance(budget, Tracker):
        return budget
    if isinstance(budget, bool) or not isinstance(budget, int | float):
        raise TypeError(f"with_budget takes a Tracker or a cap; got {type(budget).__name__}")
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError(f"a budget cap must be finite and > 0; got {budget!r}")
    return float(budget)


def check_caps_have_a_tracker(root: Flow) -> None:
    """Raise when a capped flow in ``root``'s composition tree has no tracker above it.

    A cap makes a child of the enclosing tracker, which supplies the
    pricing; without one there is nothing to make it from.

    Raises:
        RuntimeError: A flow with ``with_budget(cap)`` and no
            ``with_budget(tracker)`` on any flow enclosing it.
    """
    seen: set[tuple[int, bool]] = set()
    stack: list[tuple[Flow, bool]] = [(root, False)]
    while stack:
        flow, covered = stack.pop()
        if (id(flow), covered) in seen:
            continue
        seen.add((id(flow), covered))
        own = flow._budget
        if isinstance(own, float) and not covered:
            raise RuntimeError(_no_tracker_message(flow, own))
        covered = covered or own is not None
        for node in flow._nodes:
            stack.extend((child, covered) for _, child in _child_flows(node))


def _no_tracker_message(flow: Flow, cap: float) -> str:
    label = flow._name or "<anonymous>"
    return (
        f"Flow {label!r} has with_budget({cap}) but no tracker encloses it: attach a "
        f"Tracker with with_budget(tracker) to an enclosing flow (e.g. the top-level one)"
    )


@dataclass(frozen=True)
class RunBudget:
    """The budget context one run of a flow runs under.

    ``tracker`` is ``ctx.budget`` for the run. ``halt`` is the event the
    run observes. ``stop`` is the run's own halt event when it is capped
    (``halt`` is then ``stop``), set on crossing the cap or when
    ``enclosing`` — the halt the run would observe uncapped — is set.
    """

    tracker: Tracker | None
    halt: asyncio.Event | None
    stop: asyncio.Event | None = None
    enclosing: asyncio.Event | None = None

    def stopped_by_budget(self) -> bool:
        """True when the run's own stop is set and the enclosing halt is not."""
        if self.stop is None or not self.stop.is_set():
            return False
        return self.enclosing is None or not self.enclosing.is_set()


@contextlib.asynccontextmanager
async def run_budget(
    flow: Flow,
    path: ScopePath,
    scopes: ScopeRegistry,
    parent_budget: Tracker | None,
    parent_halt: asyncio.Event | None,
) -> AsyncIterator[RunBudget]:
    """The budget context of one run of ``flow`` at ``path``, for the duration of the run.

    A run with a tracker of its own keeps it registered at ``path`` — after
    restoring the spend a checked-out snapshot saved there — and drops it
    once the run completes; a run that stops keeps it, for the run's halt
    checkpoint. A capped run's link to the enclosing halt ends with the run.

    Raises:
        RuntimeError: ``flow`` is capped and no tracker encloses it (a flow
            reached outside its parent's composition tree, e.g. through
            ``ctx.flow.dispatch``).
    """
    enclosing = flow._halt_event if flow._halt_event is not None else parent_halt
    own = flow._budget
    if own is None:
        yield RunBudget(parent_budget, enclosing)
        return
    if isinstance(own, Tracker):
        context = RunBudget(own, enclosing)
    else:
        if parent_budget is None:
            raise RuntimeError(_no_tracker_message(flow, own))
        stop = asyncio.Event()
        context = RunBudget(parent_budget.child(own, halt=stop), stop, stop, enclosing)
    assert context.tracker is not None
    _restore(scopes, path, context.tracker)
    cursor = _BudgetCursor(context.tracker)
    scopes.open_cursor(path, cursor)
    link = _link(enclosing, context.stop)
    try:
        yield context
        scopes.close_cursor(path, cursor)
    finally:
        if link is not None:
            link.cancel()


def _restore(scopes: ScopeRegistry, path: ScopePath, tracker: Tracker) -> None:
    """Restore ``tracker`` from the spend a checked-out snapshot saved at ``path``, if any."""
    found, saved = scopes.take_cursor(path, BUDGET)
    if not found:
        return
    try:
        tracker.restore(
            float(saved["spent"]), {k: float(v) for k, v in saved["costs_by_op"].items()}
        )
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        where = path_str((*path, BUDGET))
        raise TypeError(f"cursor at {where!r} cannot be restored: {e}") from e


def _link(enclosing: asyncio.Event | None, stop: asyncio.Event | None) -> asyncio.Task[None] | None:
    """Set ``stop`` once ``enclosing`` is set; ``None`` when there is nothing to link."""
    if enclosing is None or stop is None:
        return None
    if enclosing.is_set():
        stop.set()
        return None

    async def follow() -> None:
        await enclosing.wait()
        stop.set()

    return asyncio.create_task(follow())


class _BudgetCursor:
    """Cursor of a run's own tracker: its accounting, restored when the run resumes."""

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker

    def cursor(self) -> dict[str, Any]:
        """``{"budget": {"spent", "costs_by_op"}}``."""
        return {BUDGET: self.tracker.snapshot()}
