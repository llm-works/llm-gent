# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost trait — agent-scoped cost tracker.

Wraps a :class:`CostTracker` for lifecycle management via the agent
Registry. Trait use is optional: consumers can build a :class:`CostTracker`
directly and pass it to :meth:`Flow.with_cost_tracker` without going
through the trait. The trait adds:

- Agent-mounted access (``agent.require_trait(CostTrait).tracker``).
- ``on_stop`` emits a final summary log with total spend and budget.
- A natural home for future declarative wiring (yaml / manifest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...cost import CostTracker
from ..base import BaseTrait


if TYPE_CHECKING:
    from ...agent import Agent


class CostTrait(BaseTrait):
    """Mounts a :class:`CostTracker` on the agent.

    Construction is separate from the tracker to keep the tracker
    reusable outside the trait system (Flow-only wiring, tests, etc.).

    Example::

        tracker = CostTracker(lg, pricing, budget=1.0, halt=halt_event)
        agent.add_trait(CostTrait(agent, tracker=tracker))
        flow.with_halt(halt_event).with_cost_tracker(tracker)
    """

    def __init__(self, agent: Agent, *, tracker: CostTracker) -> None:
        """Mount the trait with a caller-owned :class:`CostTracker`."""
        super().__init__(agent)
        self._tracker = tracker

    @property
    def tracker(self) -> CostTracker:
        """The wrapped :class:`CostTracker`."""
        return self._tracker

    def on_start(self) -> None:
        """Trace lifecycle start; no resources to initialize."""
        self.agent.lg.trace(
            "cost trait started",
            extra={"agent": self.agent.name, "budget": self._tracker.budget},
        )

    def on_stop(self) -> None:
        """Log a final summary (budget / spent / exceeded)."""
        self.agent.lg.info(
            "cost trait stopped",
            extra={
                "agent": self.agent.name,
                "budget": self._tracker.budget,
                "spent": self._tracker.spent,
                "exceeded": self._tracker.exceeded,
            },
        )


__all__ = ["CostTrait"]
