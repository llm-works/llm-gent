# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Start tool - start an agent."""

import argparse
from typing import Any

from appinfra.app.tools import ToolConfig

from .gateway import GatewayTool


class StartTool(GatewayTool):
    """Start an agent."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="start", help_text="Start an agent by name")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "name",
            help="Name of the agent to start",
        )
        super().add_args(parser)

    def run(self, **kwargs: Any) -> int:
        name = self.args.name
        data = self._request("POST", f"/agents/{name}/start", timeout=30.0, agent=name)
        if data is None:
            return 1

        return self._handle_response(name, data)

    def _handle_response(self, name: str, data: dict[str, Any]) -> int:
        """Handle start response and print status."""
        status = data.get("status", "unknown")
        if status == "running":
            print(f"Agent '{name}' started successfully")
            if data.get("schedule_interval"):
                print(f"  Running on {data['schedule_interval']}s schedule")
            return 0
        if status == "error":
            print(f"Agent '{name}' failed to start: {data.get('error', 'Unknown error')}")
            return 1
        print(f"Agent '{name}' status: {status}")
        return 0
