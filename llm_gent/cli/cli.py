#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CLI entry point for agent server and management.

Provides commands to:
- serve: Start the agent gateway server
- list: List registered agents
- start: Start an agent
- stop: Stop an agent
- ask: Ask an agent a question
- feedback: Provide feedback to an agent
- rate: Rate agent responses
"""

from appinfra.app import AppBuilder

from .. import __version__
from .tools import (
    AgentTool,
    AskTool,
    FeedbackTool,
    ListTool,
    RateTool,
    ServeTool,
    StartTool,
    StopTool,
)


def main() -> int:
    """Main entry point for the CLI."""
    app = (
        AppBuilder("llm-gent")
        .with_description("LLM agent server and management")
        .version.with_semver(__version__)
        .done()
        .config.with_spec("llm-works", "llm-gent")
        .done()
        .cli.with_all_flags()
        .done()
        .tools.with_tools(
            ServeTool(),
            ListTool(),
            StartTool(),
            StopTool(),
            AskTool(),
            FeedbackTool(),
            RateTool(),
            AgentTool(),
        )
        .done()
        .build()
    )
    result: int = app.main()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
