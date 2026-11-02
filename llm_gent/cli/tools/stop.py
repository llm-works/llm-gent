# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Stop tool - stop an agent."""

import argparse
from typing import Any

from appinfra.app.tools import ToolConfig

from .gateway import GatewayTool


class StopTool(GatewayTool):
    """Stop an agent."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="stop", help_text="Stop an agent by name")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "name",
            help="Name of the agent to stop",
        )
        super().add_args(parser)

    def run(self, **kwargs: Any) -> int:
        name = self.args.name
        data = self._request("POST", f"/agents/{name}/stop", timeout=30.0, agent=name)
        if data is None:
            return 1

        return self._handle_response(name, data)

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
