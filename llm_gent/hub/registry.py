"""Unified agent registry for the swarm hub.

Single source of truth for agent membership, configuration, health,
and lifecycle state. Replaces the separate bus and runtime registries.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from appinfra import DotDict

from ..bus.protocol import AgentStats


class AgentHealth(StrEnum):
    """Agent health states derived from heartbeat recency."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


class AgentType(StrEnum):
    """How the agent is managed by the hub."""

    INJECTED = "injected"
    EXTERNAL = "external"


@dataclass
class AgentEntry:
    """Registry entry for a connected agent.

    Combines membership, configuration, health, and lifecycle state
    in one record.
    """

    id: str
    agent_type: AgentType
    config: DotDict = field(default_factory=DotDict)
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Health tracking
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_run: datetime | None = None
    stats: AgentStats = field(default_factory=lambda: AgentStats())

    # Lifecycle
    restart_count: int = 0
    error: str | None = None

    # Timeout for health derivation (seconds without heartbeat before dead).
    dead_timeout: float = 90.0

    def is_alive(self, timeout_secs: float | None = None) -> bool:
        """Check if agent has heartbeated within the timeout window."""
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

    @property
    def schedule_interval(self) -> float | None:
        """Extract schedule interval from config."""
        schedule = self.config.get("schedule")
        if schedule and isinstance(schedule, dict):
            interval = schedule.get("interval")
            if interval is not None:
                try:
                    return float(interval)
                except (ValueError, TypeError):
                    return None
        return None


class Registry:
    """Unified thread-safe agent registry.

    Tracks all agents in the swarm -- injected and external -- with
    their configuration, health, and lifecycle state.
    """

    def __init__(self, dead_timeout: float = 90.0) -> None:
        self._agents: dict[str, AgentEntry] = {}
        self._lock = threading.RLock()
        self._dead_timeout = dead_timeout

    def register(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.EXTERNAL,
        config: DotDict | None = None,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEntry:
        """Register an agent. Re-registering updates entry (preserves restart_count)."""
        now = datetime.now(UTC)
        with self._lock:
            existing = self._agents.get(agent_id)
            entry = AgentEntry(
                id=agent_id,
                agent_type=agent_type,
                config=config or DotDict(),
                capabilities=capabilities or [],
                metadata=metadata or {},
                registered_at=now,
                last_heartbeat=now,
                stats=existing.stats if existing else AgentStats(),
                restart_count=existing.restart_count if existing else 0,
                dead_timeout=self._dead_timeout,
            )
            self._agents[agent_id] = entry
            return entry

    def register_or_merge(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.EXTERNAL,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEntry:
        """Atomically register or merge an agent.

        If the agent is already registered as INJECTED, merges
        capabilities/metadata without overwriting the type or config.
        Otherwise registers as a new external agent.
        """
        now = datetime.now(UTC)
        with self._lock:
            existing = self._agents.get(agent_id)
            if existing is not None and existing.agent_type == AgentType.INJECTED:
                if capabilities:
                    existing.capabilities = capabilities
                if metadata:
                    existing.metadata.update(metadata)
                existing.last_heartbeat = now
                return existing
            return self.register(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                metadata=metadata,
            )

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent. Returns True if found."""
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    def get(self, agent_id: str) -> AgentEntry | None:
        """Get an agent's entry."""
        with self._lock:
            return self._agents.get(agent_id)

    def heartbeat(self, agent_id: str, stats: AgentStats | None = None) -> bool:
        """Update heartbeat timestamp and optional stats. Returns False if unknown."""
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return False
            now = datetime.now(UTC)
            entry.last_heartbeat = now
            if stats is not None:
                if stats.ticks > entry.stats.ticks:
                    entry.last_run = now
                entry.stats = stats
            return True

    def list_agents(self) -> list[AgentEntry]:
        """Get all registered agents."""
        with self._lock:
            return list(self._agents.values())

    def get_by_type(self, agent_type: AgentType) -> list[AgentEntry]:
        """Get agents of a specific type."""
        with self._lock:
            return [a for a in self._agents.values() if a.agent_type == agent_type]

    def get_healthy(self) -> list[AgentEntry]:
        """Get agents with recent heartbeats."""
        with self._lock:
            return [a for a in self._agents.values() if a.is_alive(self._dead_timeout)]

    def get_unhealthy(self) -> list[AgentEntry]:
        """Get agents that missed heartbeats but aren't dead."""
        with self._lock:
            return [a for a in self._agents.values() if a.health == AgentHealth.UNHEALTHY]

    def get_dead(self) -> list[AgentEntry]:
        """Get agents past the dead threshold."""
        with self._lock:
            return [a for a in self._agents.values() if a.health == AgentHealth.DEAD]

    def cleanup_dead(self) -> list[str]:
        """Remove dead agents. Returns removed IDs."""
        with self._lock:
            dead = [a.id for a in self._agents.values() if a.health == AgentHealth.DEAD]
            for agent_id in dead:
                del self._agents[agent_id]
            return dead

    def increment_restart(self, agent_id: str) -> int:
        """Increment restart count. Returns new count (0 if not found)."""
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return 0
            entry.restart_count += 1
            return entry.restart_count

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._agents)

    @property
    def healthy_count(self) -> int:
        with self._lock:
            return sum(1 for a in self._agents.values() if a.is_alive(self._dead_timeout))
