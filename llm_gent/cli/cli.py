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

from pathlib import Path

from appinfra.app import AppBuilder

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


_BASE_CONFIG = Path(__file__).parent.parent / "etc" / "llm-gent.yaml"


def main() -> int:
    """Main entry point for the CLI."""
    app = (
        AppBuilder("agent")
        .with_description("LLM agent server and management")
        .with_config_spec("llm-works", "llm-gent", _BASE_CONFIG)
        .with_standard_args(etc_dir=True)
        .tools.with_tool(ServeTool())
        .with_tool(ListTool())
        .with_tool(StartTool())
        .with_tool(StopTool())
        .with_tool(AskTool())
        .with_tool(FeedbackTool())
        .with_tool(RateTool())
        .with_tool(AgentTool())
        .done()
        .build()
    )
    result: int = app.main()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
