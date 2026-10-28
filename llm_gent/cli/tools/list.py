# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""List tool - list registered agents."""

from typing import Any

from appinfra.app.tools import ToolConfig

from .gateway import GatewayTool


class ListTool(GatewayTool):
    """List all registered agents and their status."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="list", help_text="List all registered agents")
        super().__init__(parent, config)

    def run(self, **kwargs: Any) -> int:
        data = self._request("GET", "/agents", timeout=10.0)
        if data is None:
            return 1

        agents = data.get("agents", [])
        if not agents:
            print("No agents registered")
            return 0

        self._print_agents_table(agents)
        return 0

    def _print_agents_table(self, agents: list[dict[str, Any]]) -> None:
        """Print agents in tabular format."""
        print(f"{'NAME':<30} {'STATUS':<12} {'CYCLES':<8} {'SCHEDULE'}")
        print("-" * 70)

        for agent in agents:
            name = agent.get("name", "<unknown>")
            status = agent.get("status", "unknown")
            cycles = agent.get("cycle_count", 0)
            interval = agent.get("schedule_interval")
            schedule = f"{interval}s" if interval is not None else "-"
            status_display = self._colorize_status(status)
            print(f"{name:<30} {status_display:<21} {cycles:<8} {schedule}")

    def _colorize_status(self, status: str) -> str:
        """Apply ANSI color to status string."""
        if status == "running":
            return f"\033[32m{status}\033[0m"  # Green
        if status == "error":
            return f"\033[31m{status}\033[0m"  # Red
        return status
