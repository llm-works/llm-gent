# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Feedback tool - provide feedback to an agent."""

import argparse
import json
from typing import Any

import httpx
from appinfra.app.tools import Tool, ToolConfig


class FeedbackTool(Tool):
    """Provide feedback to an agent."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="feedback", help_text="Provide feedback to an agent")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "name",
            help="Name of the agent to provide feedback to",
        )
        parser.add_argument(
            "message",
            help="Feedback message",
        )
        parser.add_argument(
            "--server",
            default="http://localhost:8080",
            help="Gateway server URL (default: http://localhost:8080)",
        )

    def run(self, **kwargs: Any) -> int:
        server = self.args.server.rstrip("/")
        name = self.args.name
        message = self.args.message

        data = self._send_feedback(server, name, message)
        if data is None:
            return 1

        if data.get("success"):
            print(f"Feedback recorded for agent '{name}'")
            return 0
        print("Failed to record feedback")
        return 1

    def _send_feedback(self, server: str, name: str, message: str) -> dict[str, Any] | None:
        """Send feedback request to gateway."""
        try:
            response = httpx.post(
                f"{server}/agents/{name}/feedback", json={"message": message}, timeout=10.0
            )
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
