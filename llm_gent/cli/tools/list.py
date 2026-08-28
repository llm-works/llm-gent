# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""List tool - list registered agents."""

import argparse
import json
from typing import Any

import httpx
from appinfra.app.tools import Tool, ToolConfig


class ListTool(Tool):
    """List all registered agents and their status."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="list", help_text="List all registered agents")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--server",
            default="http://localhost:8080",
            help="Gateway server URL (default: http://localhost:8080)",
        )

    def run(self, **kwargs: Any) -> int:
        server = self.args.server.rstrip("/")
        data = self._fetch_agents(server)
        if data is None:
            return 1

        agents = data.get("agents", [])
        if not agents:
            print("No agents registered")
            return 0

        self._print_agents_table(agents)
        return 0

    def _fetch_agents(self, server: str) -> dict[str, Any] | None:
        """Fetch agents from gateway server."""
        try:
            response = httpx.get(f"{server}/agents", timeout=10.0)
            response.raise_for_status()
            return dict(response.json())
        except httpx.RequestError as e:
            self.lg.error("failed to connect to server", extra={"exception": e})
            print(f"Error: Could not connect to gateway at {server}")
            return None
        except httpx.HTTPStatusError as e:
            self.lg.error("server error", extra={"status": e.response.status_code})
            print(f"Error: Server returned {e.response.status_code}")
            return None
        except json.JSONDecodeError:
            self.lg.error("invalid JSON response from server")
            print("Error: Server returned invalid response")
            return None

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
        elif status == "error":
            return f"\033[31m{status}\033[0m"  # Red
        return status
