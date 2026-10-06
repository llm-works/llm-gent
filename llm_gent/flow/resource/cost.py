# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost tracking: the cost tracker is a flow resource, under :data:`COST`.

Cost is what calls and operations cost; a budget is the limit cost is
checked against. Gent tracks cost; the agent decides how to keep to a
budget. A :class:`~llm_gent.core.cost.CostTracker` implements the
:class:`~llm_gent.flow.Resource` protocol (``snapshot()`` /
``restore(data)``, and ``child(budget)`` for a budget per run), and the
cost API is sugar over :meth:`Flow.with_resource` with :data:`COST`:

- :meth:`Flow.with_cost_tracker` ``(tracker)`` — ``with_resource(COST,
  tracker)``: every run of the flow runs on that tracker;
- :meth:`Flow.with_budget` ``(limit)`` — ``with_resource(COST,
  budget=limit)``: each run of the flow runs on a child of its tracker
  (its own, else the enclosing one) with that budget, so spend rolls up
  and every budget on the chain is tracked. A map body runs once per
  item, an iterate body once per pass, a subflow once per ``.call``:
  per-item, per-pass, per-call budgets;
- neither — the run shares the enclosing tracker;
- ``ctx.cost`` — ``ctx.resource(COST, None)``.

Crossing a budget latches the child's ``exceeded`` / ``urgent_wrapup``
for the agent to read through ``ctx.cost``; it does not stop the run. A
hard stop is the app's choice: a tracker built with ``halt=`` the run's
halt event pauses the run when that tracker's budget is crossed.

The running cost is reconstructed across pause, resume and shortcut the
way every resource's accounting is (:mod:`llm_gent.flow.resource._runtime`):
the top-level flow's tracker through the completion commit, so a later
session continues the total; a budgeted run's child while the run is in
progress. A :class:`~llm_gent.core.cost.CostTracker` restores the saved
spend as is, so its spend is cumulative over the whole history. What
resume means for the spend is the tracker's: a subclass keeps keys of
its own in ``snapshot()`` (a session's baseline, say) and decides in
``restore()`` — continue, rebase, ignore, amend the cap. Budgeted runs'
children come from ``tracker.child()``, so a subclass that overrides it
puts them on its own class too.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...core.cost import CostTracker
from .base import ResourceKey


if TYPE_CHECKING:
    from ..flow import Flow


COST = ResourceKey[CostTracker]("cost")
"""The cost tracker's resource key: ``with_cost_tracker`` / ``with_budget`` / ``ctx.cost``."""


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
    nothing to make it from. Checked before the generic resource checks
    so the error names the cost API.

    Raises:
        RuntimeError: A flow with ``with_budget(...)`` and no
            ``with_cost_tracker(...)`` on it or any flow enclosing it.
    """
    from .._node_id import _child_flows  # .._node_id imports nodes, which imports this package

    seen: set[tuple[int, bool]] = set()
    stack: list[tuple[Flow, bool]] = [(root, False)]
    while stack:
        flow, covered = stack.pop()
        if (id(flow), covered) in seen:
            continue
        seen.add((id(flow), covered))
        covered = covered or COST in flow._resources
        budget = flow._resource_children.get(COST)
        if budget is not None and not covered:
            raise RuntimeError(_no_tracker_message(flow, budget.get("budget")))
        covered = covered or budget is not None
        for node in flow._nodes:
            stack.extend((child, covered) for _, child in _child_flows(node))


def _no_tracker_message(flow: Flow, budget: object) -> str:
    """The error for a flow with a budget and no cost tracker to make its child from."""
    label = flow._name or "<anonymous>"
    return (
        f"Flow {label!r} has with_budget({budget}) but no cost tracker: attach a "
        f"CostTracker with with_cost_tracker(tracker) to it or an enclosing flow "
        f"(e.g. the top-level one)"
    )
