# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent registry - manages collection of agent configurations.

The registry is a pure data structure that tracks registered agents.
It does not manage lifecycle (start/stop) - that's handled by Core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra import DotDict
from appinfra.service import State

from .handle import AgentHandle, AgentInfo


if TYPE_CHECKING:
    from appinfra.log import Logger


class AgentRegistry:
    """Manages collection of agent configurations.

    The registry tracks registered agents and their handles.
    It provides a pure data interface:
    - register: Add an agent configuration
    - get: Retrieve an agent handle
    - list_agents: List all registered agents
    - unregister: Remove an agent

    Lifecycle management (start, stop, communicate) is handled by Core.
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize the registry.

        Args:
            lg: Logger instance.
        """
        self._lg = lg
        self._agents: dict[str, AgentHandle] = {}

    def register(self, name: str, config: DotDict) -> AgentHandle:
        """Register an agent configuration.

        The agent is registered in CREATED state.

        Args:
            name: Unique agent name.
            config: Agent configuration (DotDict from appinfra config).

        Returns:
            AgentHandle for the registered agent.

        Raises:
            ValueError: If agent name already registered.
        """
        if name in self._agents:
            raise ValueError(f"Agent already registered: {name}")

        # Extract schedule interval from config
        schedule_interval = None
        schedule = config.get("schedule")
        if schedule and isinstance(schedule, dict):
            schedule_interval = schedule.get("interval")

        handle = AgentHandle(
            name=name,
            config=config,
            state=State.CREATED,
            schedule_interval=schedule_interval,
        )
        self._agents[name] = handle
        self._lg.debug("registered agent", extra={"agent": name})
        return handle

    def get(self, name: str) -> AgentHandle | None:
        """Get agent handle by name.

        Args:
            name: Agent name.

        Returns:
            AgentHandle or None if not found.
        """
        return self._agents.get(name)

    def list_agents(self) -> list[AgentInfo]:
        """List all registered agents.

        Returns:
            List of AgentInfo snapshots.
        """
        return [AgentInfo.from_handle(h) for h in self._agents.values()]

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry.

        The agent must be in CREATED or STOPPED state.

        Args:
            name: Agent name.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent is not in a safe state to unregister.
        """
        handle = self._agents.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        if handle.state not in (State.CREATED, State.STOPPED):
            raise RuntimeError(
                f"Cannot unregister agent '{name}' in state {handle.state.value}. "
                f"Agent must be CREATED or STOPPED."
            )

        del self._agents[name]
        self._lg.info("agent unregistered", extra={"agent": name})

    def handles(self) -> list[AgentHandle]:
        """Get all agent handles.

        Returns:
            List of all AgentHandle instances.
        """
        return list(self._agents.values())
