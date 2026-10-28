# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Feedback tool - provide feedback to an agent."""

import argparse
from typing import Any

from appinfra.app.tools import ToolConfig

from .gateway import GatewayTool


class FeedbackTool(GatewayTool):
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
        super().add_args(parser)

    def run(self, **kwargs: Any) -> int:
        name = self.args.name
        data = self._request(
            "POST",
            f"/agents/{name}/feedback",
            timeout=10.0,
            agent=name,
            body={"message": self.args.message},
        )
        if data is None:
            return 1

        if data.get("success"):
            print(f"Feedback recorded for agent '{name}'")
            return 0
        print("Failed to record feedback")
        return 1
