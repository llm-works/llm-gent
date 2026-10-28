# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent registry for swarm membership and health tracking.

Thread-safe registry tracking which agents are connected to the swarm,
their capabilities, health status, and runtime statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from .health import AgentHealth as AgentHealth
from .health import AgentRecord, HeartbeatRegistry
from .health import AgentType as AgentType


@dataclass
class AgentStats:
    """Runtime statistics reported by an agent."""

    ticks: int = 0
    errors: int = 0
    llm_tokens_used: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInfo(AgentRecord):
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


class AgentRegistry(HeartbeatRegistry[AgentInfo]):
    """Thread-safe registry of agents connected to the swarm.

    Tracks agent membership, health, and statistics. Used by the hub
    to manage the swarm.
    """

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
