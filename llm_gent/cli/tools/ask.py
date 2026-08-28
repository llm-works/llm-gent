# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Ask tool - ask an agent a question."""

import argparse
import json
from typing import Any

import httpx
from appinfra.app.tools import Tool, ToolConfig


class AskTool(Tool):
    """Ask an agent a conversational question."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="ask", help_text="Ask an agent a question")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "name",
            help="Name of the agent to ask",
        )
        parser.add_argument(
            "question",
            help="Question to ask the agent",
        )
        parser.add_argument(
            "--server",
            default="http://localhost:8080",
            help="Gateway server URL (default: http://localhost:8080)",
        )

    def run(self, **kwargs: Any) -> int:
        server = self.args.server.rstrip("/")
        name = self.args.name
        question = self.args.question

        data = self._ask_agent(server, name, question)
        if data is None:
            return 1

        agent_response = data.get("response", "")
        print(f"\n{name}: {agent_response}\n")
        return 0

    def _ask_agent(self, server: str, name: str, question: str) -> dict[str, Any] | None:
        """Send ask request to gateway."""
        try:
            response = httpx.post(
                f"{server}/agents/{name}/ask",
                json={"question": question},
                timeout=60.0,  # Longer timeout for LLM responses
            )
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
