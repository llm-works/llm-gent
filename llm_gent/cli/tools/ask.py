# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Ask tool - ask an agent a question."""

import argparse
from typing import Any

from appinfra.app.tools import ToolConfig

from .gateway import GatewayTool


class AskTool(GatewayTool):
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
        super().add_args(parser)

    def run(self, **kwargs: Any) -> int:
        name = self.args.name
        data = self._request(
            "POST",
            f"/agents/{name}/ask",
            timeout=60.0,  # Longer timeout for LLM responses
            agent=name,
            body={"question": self.args.question},
        )
        if data is None:
            return 1

        agent_response = data.get("response", "")
        print(f"\n{name}: {agent_response}\n")
        return 0
