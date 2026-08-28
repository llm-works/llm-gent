# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for Task models and ToolsTrait."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llm_gent import (
    ShellTool,
    Task,
    TaskResult,
    ToolCallResult,
    ToolResult,
)
from llm_gent.core.tools.base import BaseTool
from llm_gent.core.traits.builtin.tools import ToolsTrait


pytestmark = pytest.mark.unit


class MockCompleteTaskTool(BaseTool):
    """Mock complete_task tool for testing."""

    name = "complete_task"
    description = "Complete the task"
    parameters = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["done", "stuck"]},
            "conclusion": {"type": "string"},
        },
        "required": ["status", "conclusion"],
    }
    terminal = True

    def execute(self, status: str, conclusion: str) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"Task completed with status: {status}",
            terminal=True,
            terminal_data={"status": status, "conclusion": conclusion},
        )


class Answer(BaseModel):
    """Test output schema."""

    answer: str
    confidence: float


class TestTaskModel:
    """Tests for Task model."""

    def test_minimal_task(self):
        task = Task(name="test", description="Do something")

        assert task.name == "test"
        assert task.description == "Do something"
        assert task.context == {}
        assert task.output_schema is None
        assert task.system_prompt is None
        assert task.max_iterations == 10

    def test_full_task(self):
        task = Task(
            name="analyze",
            description="Analyze sentiment",
            context={"text": "Hello"},
            output_schema=Answer,
            system_prompt="Be precise.",
            max_iterations=5,
        )

        assert task.name == "analyze"
        assert task.description == "Analyze sentiment"
        assert task.context == {"text": "Hello"}
        assert task.output_schema is Answer
        assert task.system_prompt == "Be precise."
        assert task.max_iterations == 5


class TestTaskResultModel:
    """Tests for TaskResult model."""

    def test_success_result(self):
        result = TaskResult(
            success=True,
            content="The answer is 4",
            parsed=Answer(answer="4", confidence=0.95),
            iterations=2,
            tokens_used=100,
        )

        assert result.success is True
        assert result.content == "The answer is 4"
        assert result.parsed.answer == "4"
        assert result.parsed.confidence == 0.95
        assert result.tool_calls == []
        assert result.iterations == 2
        assert result.tokens_used == 100
        assert result.error is None

    def test_error_result(self):
        result = TaskResult(
            success=False,
            content="",
            error="LLM failed",
        )

        assert result.success is False
        assert result.content == ""
        assert result.parsed is None
        assert result.error == "LLM failed"

    def test_result_with_tool_calls(self):
        tool_call = ToolCallResult(
            call_id="call-1",
            name="shell",
            result=ToolResult(success=True, output="hello"),
        )
        result = TaskResult(
            success=True,
            content="Executed command",
            tool_calls=[tool_call],
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"


class TestToolsTrait:
    """Tests for ToolsTrait."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        return MagicMock()

    def test_register_tool(self, mock_agent):
        trait = ToolsTrait(mock_agent)
        tool = ShellTool()

        trait.register(tool)

        assert trait.has_tools() is True
        assert trait.registry.get("shell") is tool

    def test_unregister_tool(self, mock_agent):
        trait = ToolsTrait(mock_agent)
        trait.register(ShellTool())

        trait.unregister("shell")

        assert trait.has_tools() is False

    def test_trait_lifecycle(self, mock_agent):
        trait = ToolsTrait(mock_agent)

        trait.on_start()
        trait.on_stop()

        assert trait.agent is mock_agent

    def test_empty_registry(self, mock_agent):
        trait = ToolsTrait(mock_agent)

        assert trait.has_tools() is False
