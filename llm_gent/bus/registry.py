"""Agent registry for swarm membership and health tracking.

Thread-safe registry tracking which agents are connected to the swarm,
their capabilities, health status, and runtime statistics.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class AgentHealth(StrEnum):
    """Agent health states."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


class AgentType(StrEnum):
    """How the agent is managed by the hub."""

    INJECTED = "injected"
    EXTERNAL = "external"


@dataclass
class AgentStats:
    """Runtime statistics reported by an agent."""

    ticks: int = 0
    errors: int = 0
    llm_tokens_used: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInfo:
    """Registry entry for a connected agent."""

    id: str
    agent_type: AgentType
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    health_url: str | None = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    stats: AgentStats = field(default_factory=AgentStats)
    restart_count: int = 0

    # Timeout for health derivation (seconds without heartbeat before dead).
    dead_timeout: float = 90.0

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
        elif elapsed < 2 * self.dead_timeout:
            return AgentHealth.UNHEALTHY
        return AgentHealth.DEAD


class AgentRegistry:
    """Thread-safe registry of agents connected to the swarm.

    Tracks agent membership, health, and statistics. Used by the hub
    to manage the swarm.
    """

    def __init__(self, dead_timeout: float = 90.0) -> None:
        """Initialize registry.

        Args:
            dead_timeout: Seconds without heartbeat before agent is considered dead.
        """
        self._agents: dict[str, AgentInfo] = {}
        self._lock = threading.RLock()
        self._dead_timeout = dead_timeout

    def register(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.EXTERNAL,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        health_url: str | None = None,
    ) -> AgentInfo:
        """Register an agent with the swarm.

        If the agent is already registered, updates its registration
        (preserves restart_count and stats).

        Args:
            agent_id: Unique agent identifier.
            agent_type: Whether the agent is injected or external.
            capabilities: List of capability strings the agent advertises.
            metadata: Arbitrary metadata about the agent.
            health_url: HTTP endpoint for active health probing.

        Returns:
            The agent's registry entry.
        """
        now = datetime.now(UTC)
        with self._lock:
            existing = self._agents.get(agent_id)
            info = AgentInfo(
                id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities or [],
                metadata=metadata or {},
                health_url=health_url,
                registered_at=now,
                last_heartbeat=now,
                stats=existing.stats if existing else AgentStats(),
                restart_count=existing.restart_count if existing else 0,
                dead_timeout=self._dead_timeout,
            )
            self._agents[agent_id] = info
            return info

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry.

        Args:
            agent_id: Agent to remove.

        Returns:
            True if the agent was found and removed.
        """
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    def heartbeat(self, agent_id: str, stats: AgentStats | None = None) -> bool:
        """Update agent's last heartbeat timestamp and optional stats.

        Args:
            agent_id: Agent sending the heartbeat.
            stats: Updated runtime statistics.

        Returns:
            True if agent was found, False if unknown agent.
        """
        with self._lock:
            info = self._agents.get(agent_id)
            if info is None:
                return False
            info.last_heartbeat = datetime.now(UTC)
            if stats is not None:
                info.stats = stats
            return True

    def get(self, agent_id: str) -> AgentInfo | None:
        """Get a single agent's info (shallow copy).

        Args:
            agent_id: Agent to look up.

        Returns:
            AgentInfo or None if not registered.
        """
        with self._lock:
            entry = self._agents.get(agent_id)
            return replace(entry) if entry is not None else None

    def list_agents(self) -> list[AgentInfo]:
        """Get all registered agents (shallow copies)."""
        with self._lock:
            return [replace(a) for a in self._agents.values()]

    def get_healthy(self) -> list[AgentInfo]:
        """Get agents with recent heartbeats."""
        with self._lock:
            return [a for a in self._agents.values() if a.is_alive(self._dead_timeout)]

    def get_unhealthy(self) -> list[AgentInfo]:
        """Get agents that have missed heartbeats but aren't dead yet."""
        with self._lock:
            return [a for a in self._agents.values() if a.health == AgentHealth.UNHEALTHY]

    def get_dead(self) -> list[AgentInfo]:
        """Get agents past the dead threshold."""
        with self._lock:
            return [a for a in self._agents.values() if a.health == AgentHealth.DEAD]

    def cleanup_dead(self) -> list[str]:
        """Remove dead agents from the registry.

        Returns:
            List of removed agent IDs.
        """
        with self._lock:
            dead = [a.id for a in self._agents.values() if a.health == AgentHealth.DEAD]
            for agent_id in dead:
                del self._agents[agent_id]
            return dead

    def increment_restart(self, agent_id: str) -> int:
        """Increment and return the restart count for an agent.

        Args:
            agent_id: Agent being restarted.

        Returns:
            New restart count, or 0 if agent not found.
        """
        with self._lock:
            info = self._agents.get(agent_id)
            if info is None:
                return 0
            info.restart_count += 1
            return info.restart_count

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
