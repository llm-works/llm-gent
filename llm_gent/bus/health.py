# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Heartbeat health shared by the swarm's agent registries.

:class:`AgentRecord` derives an entry's health from its last heartbeat;
:class:`AgentTable` holds entries by id under a lock and answers the
health queries.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from enum import StrEnum
from typing import Generic, TypeVar


class AgentHealth(StrEnum):
    """Agent health states derived from heartbeat recency."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


class AgentType(StrEnum):
    """How the agent is managed by the hub."""

    INJECTED = "injected"
    EXTERNAL = "external"


class AgentRecord:
    """Health from heartbeat recency, for an entry dataclass declaring these fields."""

    id: str
    last_heartbeat: datetime
    restart_count: int
    dead_timeout: float
    """Seconds without a heartbeat before the agent counts as dead."""

    def is_alive(self, timeout_secs: float | None = None) -> bool:
        """Check if agent has sent a heartbeat within the timeout window."""
        threshold = timeout_secs if timeout_secs is not None else self.dead_timeout
        elapsed = (datetime.now(UTC) - self.last_heartbeat).total_seconds()
        return elapsed < threshold

    @property
    def health(self) -> AgentHealth:
        """Derive health from heartbeat recency.

        Uses two thresholds derived from ``dead_timeout``:
        - < dead_timeout: healthy
        - dead_timeout .. 2*dead_timeout: unhealthy
        - >= 2*dead_timeout: dead
        """
        elapsed = (datetime.now(UTC) - self.last_heartbeat).total_seconds()
        if elapsed < self.dead_timeout:
            return AgentHealth.HEALTHY
        if elapsed < 2 * self.dead_timeout:
            return AgentHealth.UNHEALTHY
        return AgentHealth.DEAD


E = TypeVar("E", bound=AgentRecord)


class AgentTable(Generic[E]):
    """Thread-safe agent entries by id, with the heartbeat health queries."""

    def __init__(self, dead_timeout: float = 90.0) -> None:
        """Initialize the table.

        Args:
            dead_timeout: Seconds without heartbeat before agent is considered dead.
        """
        self._agents: dict[str, E] = {}
        self._lock = threading.RLock()
        self._dead_timeout = dead_timeout

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent. Returns True if it was registered."""
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    def get_healthy(self) -> list[E]:
        """Get agents with recent heartbeats."""
        with self._lock:
            return [a for a in self._agents.values() if a.is_alive(self._dead_timeout)]

    def get_unhealthy(self) -> list[E]:
        """Get agents that have missed heartbeats but aren't dead yet."""
        with self._lock:
            return [a for a in self._agents.values() if a.health == AgentHealth.UNHEALTHY]

    def get_dead(self) -> list[E]:
        """Get agents past the dead threshold."""
        with self._lock:
            return [a for a in self._agents.values() if a.health == AgentHealth.DEAD]

    def cleanup_dead(self) -> list[str]:
        """Remove dead agents. Returns the removed IDs."""
        with self._lock:
            dead = [a.id for a in self._agents.values() if a.health == AgentHealth.DEAD]
            for agent_id in dead:
                del self._agents[agent_id]
            return dead

    def increment_restart(self, agent_id: str) -> int:
        """Increment and return the restart count for an agent (0 if not found)."""
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return 0
            entry.restart_count += 1
            return entry.restart_count

    @property
    def count(self) -> int:
        """Total number of registered agents."""
        with self._lock:
            return len(self._agents)

    @property
    def healthy_count(self) -> int:
        """Number of healthy agents."""
        with self._lock:
            return sum(1 for a in self._agents.values() if a.is_alive(self._dead_timeout))
