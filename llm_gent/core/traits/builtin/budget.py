# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Budget trait — agent-scoped cost tracker.

Wraps a :class:`Tracker` for lifecycle management via the agent
Registry. Trait use is optional: consumers can build a :class:`Tracker`
directly and pass it to :meth:`Flow.with_budget` without going through
the trait. The trait adds:

- Agent-mounted access (``agent.require_trait(BudgetTrait).tracker``).
- ``on_stop`` emits a final summary log with total spend and cap.
- A natural home for future declarative wiring (yaml / manifest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...budget import Tracker
from ..base import BaseTrait


if TYPE_CHECKING:
    from ...agent import Agent


class BudgetTrait(BaseTrait):
    """Mounts a :class:`Tracker` on the agent.

    Construction is separate from the tracker to keep the tracker
    reusable outside the trait system (Flow-only wiring, tests, etc.).

    Example::

        tracker = Tracker(lg, pricing, budget=1.0, halt=halt_event)
        agent.add_trait(BudgetTrait(agent, tracker=tracker))
        flow.with_halt(halt_event).with_budget(tracker)
    """

    def __init__(self, agent: Agent, *, tracker: Tracker) -> None:
        """Mount the trait with a caller-owned :class:`Tracker`."""
        super().__init__(agent)
        self._tracker = tracker

    @property
    def tracker(self) -> Tracker:
        """The wrapped :class:`Tracker`."""
        return self._tracker

    def on_start(self) -> None:
        """Trace lifecycle start; no resources to initialize."""
        self.agent.lg.trace(
            "budget trait started",
            extra={"agent": self.agent.name, "budget": self._tracker.budget},
        )

    def on_stop(self) -> None:
        """Log a final summary (budget / spent / exceeded)."""
        self.agent.lg.info(
            "budget trait stopped",
            extra={
                "agent": self.agent.name,
                "budget": self._tracker.budget,
                "spent": self._tracker.spent,
                "exceeded": self._tracker.exceeded,
            },
        )


__all__ = ["BudgetTrait"]
