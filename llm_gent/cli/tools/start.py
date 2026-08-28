# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Start tool - start an agent."""

import argparse
import json
from typing import Any

import httpx
from appinfra.app.tools import Tool, ToolConfig


class StartTool(Tool):
    """Start an agent."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="start", help_text="Start an agent by name")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "name",
            help="Name of the agent to start",
        )
        parser.add_argument(
            "--server",
            default="http://localhost:8080",
            help="Gateway server URL (default: http://localhost:8080)",
        )

    def run(self, **kwargs: Any) -> int:
        server = self.args.server.rstrip("/")
        name = self.args.name

        data = self._start_agent(server, name)
        if data is None:
            return 1

        return self._handle_response(name, data)

    def _start_agent(self, server: str, name: str) -> dict[str, Any] | None:
        """Send start request to gateway."""
        try:
            response = httpx.post(f"{server}/agents/{name}/start", timeout=30.0)
            if response.status_code == 404:
                print(f"Error: Agent '{name}' not found")
                return None
            elif response.status_code == 400:
                print(f"Error: {response.json().get('detail', 'Unknown error')}")
                return None
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

    def _handle_response(self, name: str, data: dict[str, Any]) -> int:
        """Handle start response and print status."""
        status = data.get("status", "unknown")
        if status == "running":
            print(f"Agent '{name}' started successfully")
            if data.get("schedule_interval"):
                print(f"  Running on {data['schedule_interval']}s schedule")
            return 0
        elif status == "error":
            print(f"Agent '{name}' failed to start: {data.get('error', 'Unknown error')}")
            return 1
        print(f"Agent '{name}' status: {status}")
        return 0
