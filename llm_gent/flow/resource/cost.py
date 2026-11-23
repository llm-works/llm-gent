# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost tracking: the run's cost tracker is a resource, under :data:`COST`.

Cost is what calls and operations cost; a budget is the limit cost is
checked against. Gent tracks cost; the agent decides how to keep to a
budget. A :class:`~llm_gent.core.cost.Tracker` implements the
:class:`~llm_gent.flow.Resource` protocol (``snapshot()`` /
``restore(data)``, and ``child(budget)`` for a budget per run).

- :meth:`FlowRunner.start` ``(flow, cost=tracker)`` — the run tracks its
  cost on a tracker of its own, ``tracker.child()``: spend rolls up to
  ``tracker``, the run's accounting is the run's. A tracker is never the
  flow's: one on the structure would be shared by every run of it, so
  ``with_resource(COST, tracker)`` raises;
- :meth:`Flow.with_budget` ``(limit)`` — ``with_resource(COST,
  budget=limit)``: each run of the flow runs on a child of the enclosing
  tracker with that budget, so spend rolls up and every budget on the
  chain is tracked. A map body runs once per item, an iterate body once
  per pass, a subflow once per ``.call``: per-item, per-pass, per-call
  budgets;
- neither — the run shares the enclosing tracker;
- ``ctx.cost`` — ``ctx.resource(COST, None)``.

Crossing a budget, or one above it, latches ``exceeded`` /
``urgent_wrapup`` for the agent to read through ``ctx.cost``; it does not
stop the run. A hard stop is the app's choice: a tracker built with
``halt=`` the run's halt event pauses the run when that tracker's budget
is crossed.

The running cost is reconstructed across pause, resume and shortcut the
way every resource's accounting is (:mod:`llm_gent.flow.resource._runtime`):
the run's own tracker with the top-level flow's resources, through the
completion commit, so a later run of the history continues the total; a
budgeted run's child while the run is in progress. A resumed run restores
into its own new tracker, never into the one it was started with, so the
caller's tracker sees only what each run spends. A
:class:`~llm_gent.core.cost.Tracker` restores the saved spend as is, so
the run's spend is cumulative over the whole history. What resume means
for the spend is the tracker's: a subclass keeps keys of its own in
``snapshot()`` and decides in ``restore()`` — continue, rebase, ignore,
amend the cap. The run's tracker and budgeted runs' children come from
``tracker.child()``, so a subclass that overrides it puts them on its own
class too.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...core.cost import Tracker as CostTracker
from .base import ResourceKey


if TYPE_CHECKING:
    from ..flow import Flow


COST = ResourceKey[CostTracker]("cost")
"""The cost tracker's resource key: the run's (``start(cost=)``), ``with_budget``, ``ctx.cost``."""


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
    """``tracker`` when :meth:`FlowRunner.start` ``cost=`` accepts it.

    Raises:
        TypeError: Not a :class:`CostTracker`.
    """
    if not isinstance(tracker, CostTracker):
        raise TypeError(f"start(cost=) takes a CostTracker; got {type(tracker).__name__}")
    return tracker


def check_not_a_tracker(key: object, value: object) -> None:
    """Raise when ``with_resource`` is given a cost tracker to keep on the structure.

    A tracker on a flow or factory would be shared by every run of the
    flow; the run's tracker is the run's, given to
    :meth:`FlowRunner.start` (``cost=``). Child arguments
    (``with_budget``) are accepted.

    Raises:
        ValueError: ``key`` is :data:`COST` and ``value`` is set.
    """
    if key is COST and value is not None:
        raise ValueError(
            "a cost tracker is the run's, not the flow's: pass it to "
            "FlowRunner.start(flow, cost=tracker); with_budget(limit) gives a run a budget"
        )


def check_budgets_have_a_tracker(root: Flow, *, tracked: bool) -> None:
    """Raise when a flow in ``root``'s composition tree has a budget in a run without a tracker.

    A budget makes a child of the enclosing tracker (the run's, or an
    enclosing budget's), which supplies the pricing; a run started
    without ``cost=`` has none to make it from. ``tracked``: the run was
    started with one. Checked before the generic resource checks so the
    error names the cost API.

    Raises:
        RuntimeError: A flow with ``with_budget(...)`` in a run started
            without ``cost=``.
    """
    if tracked:
        return
    from .._node_id import iter_flows  # .._node_id imports nodes, which imports this package

    for flow in iter_flows(root):
        budget = flow._resource_children.get(COST)
        if budget is not None:
            raise RuntimeError(_no_tracker_message(flow, budget.get("budget")))


def _no_tracker_message(flow: Flow, budget: object) -> str:
    """The error for a flow with a budget in a run that tracks no cost."""
    label = flow._name or "<anonymous>"
    return (
        f"Flow {label!r} has with_budget({budget}) but the run tracks no cost: "
        f"start it with FlowRunner.start(flow, cost=tracker)"
    )
