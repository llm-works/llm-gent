# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Stop tool - stop an agent."""

import argparse
import json
from typing import Any

import httpx
from appinfra.app.tools import Tool, ToolConfig


class StopTool(Tool):
    """Stop an agent."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="stop", help_text="Stop an agent by name")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "name",
            help="Name of the agent to stop",
        )
        parser.add_argument(
            "--server",
            default="http://localhost:8080",
            help="Gateway server URL (default: http://localhost:8080)",
        )

    def run(self, **kwargs: Any) -> int:
        server = self.args.server.rstrip("/")
        name = self.args.name

        data = self._stop_agent(server, name)
        if data is None:
            return 1

        return self._handle_response(name, data)

    def _stop_agent(self, server: str, name: str) -> dict[str, Any] | None:
        """Send stop request to gateway."""
        try:
            response = httpx.post(f"{server}/agents/{name}/stop", timeout=30.0)
            if response.status_code == 404:
                print(f"Error: Agent '{name}' not found")
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
        """Handle stop response and print status."""
        status = data.get("status", "unknown")
        if status == "stopped":
            print(f"Agent '{name}' stopped successfully")
            return 0
        if status == "error":
            print(f"Agent '{name}' failed to stop: {data.get('error', 'Unknown error')}")
            return 1
        print(f"Agent '{name}' status: {status}")
        return 0
