"""Agent handle - represents a managed agent process or thread."""

from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from appinfra import DotDict
from appinfra.service import State


@dataclass
class AgentHandle:
    """Handle to a managed agent process.

    Contains all state needed to manage an agent's lifecycle:
    - Identity (name, config)
    - Runtime state (state, process)
    - Metrics (cycle_count, last_run, error)

    The handle is owned by the AgentRegistry and should not be
    modified directly - use Core methods instead.
    """

    name: str
    """Agent name (unique identifier)."""

    config: DotDict
    """Agent configuration (DotDict for dot notation access)."""

    state: State = State.CREATED
    """Current lifecycle state."""

    process: mp.Process | threading.Thread | None = None
    """Subprocess or thread running the agent, or None if not started."""

    cycle_count: int = 0
    """Number of scheduled cycles completed."""

    last_run: datetime | None = None
    """Timestamp of last cycle completion."""

    error: str | None = None
    """Error message if state is ERROR."""

    schedule_interval: int | None = None
    """Seconds between scheduled executions, or None if not scheduled."""

    def to_info_dict(self) -> dict[str, Any]:
        """Convert to a serializable info dictionary.

        Returns:
            Dictionary with agent status information.
        """
        return {
            "name": self.name,
            "status": self.state.value,
            "cycle_count": self.cycle_count,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "error": self.error,
            "schedule_interval": self.schedule_interval,
        }


@dataclass
class AgentInfo:
    """Immutable snapshot of agent status for API responses.

    This is a read-only view of AgentHandle for external use.
    """

    name: str
    status: str
    cycle_count: int = 0
    last_run: datetime | None = None
    error: str | None = None
    schedule_interval: int | None = None

    @classmethod
    def from_handle(cls, handle: AgentHandle) -> AgentInfo:
        """Create AgentInfo from an AgentHandle.

        Args:
            handle: The agent handle to snapshot.

        Returns:
            Immutable AgentInfo with current status.
        """
        return cls(
            name=handle.name,
            status=handle.state.value,
            cycle_count=handle.cycle_count,
            last_run=handle.last_run,
            error=handle.error,
            schedule_interval=handle.schedule_interval,
        )
