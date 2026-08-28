# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for SAIATrait."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.core.traits.builtin.saia import (
    SAIAConfig,
    SAIATrait,
    _create_executor,
    _tool_to_tooldef,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# SAIAConfig
# ---------------------------------------------------------------------------


class TestSAIAConfig:
    def test_defaults(self):
        cfg = SAIAConfig()
        assert cfg.terminal_tool == "complete_task"
        assert cfg.max_iterations == 0
        assert cfg.timeout_secs == 0
        assert cfg.system_prompt is None

    def test_custom(self):
        cfg = SAIAConfig(
            terminal_tool="done",
            max_iterations=10,
            timeout_secs=30.0,
            system_prompt="You are helpful.",
        )
        assert cfg.terminal_tool == "done"
        assert cfg.max_iterations == 10
        assert cfg.system_prompt == "You are helpful."


# ---------------------------------------------------------------------------
# SAIATrait init / properties
# ---------------------------------------------------------------------------


class TestSAIATraitInit:
    def test_init(self):
        agent = MagicMock()
        backend = MagicMock()
        trait = SAIATrait(agent, backend)

        assert trait.backend is backend
        assert trait.config.terminal_tool == "complete_task"
        assert trait._saia is None

    def test_custom_config(self):
        agent = MagicMock()
        backend = MagicMock()
        cfg = SAIAConfig(max_iterations=5)
        trait = SAIATrait(agent, backend, config=cfg)

        assert trait.config.max_iterations == 5

    def test_saia_property_raises_before_start(self):
        trait = SAIATrait(MagicMock(), MagicMock())
        with pytest.raises(RuntimeError, match="not started"):
            _ = trait.saia


# ---------------------------------------------------------------------------
# on_start / on_stop
# ---------------------------------------------------------------------------


class TestSAIATraitLifecycle:
    @pytest.fixture
    def mock_saia_builder(self):
        """Create a fluent mock builder that returns itself for all chained calls."""
        with patch("llm_gent.core.traits.builtin.saia.SAIA") as mock_saia_cls:
            mock_builder = MagicMock()
            mock_saia_cls.builder.return_value = mock_builder
            # Fluent API: every method returns the builder itself
            for method in (
                "backend",
                "max_iterations",
                "timeout",
                "logger",
                "system",
                "terminal_tool",
                "tools",
            ):
                getattr(mock_builder, method).return_value = mock_builder
            mock_builder.build.return_value = MagicMock()
            yield mock_builder

    def test_on_start_builds_saia(self, mock_saia_builder):
        agent = MagicMock()
        agent.get_trait.return_value = None  # no ToolsTrait

        backend = MagicMock()
        trait = SAIATrait(agent, backend, config=SAIAConfig(system_prompt="Be helpful"))

        trait.on_start()

        mock_saia_builder.backend.assert_called_once_with(backend)
        mock_saia_builder.system.assert_called_once_with("Be helpful")
        mock_saia_builder.terminal_tool.assert_called_once_with("complete_task")
        assert trait._saia is not None

    def test_on_start_with_tools(self, mock_saia_builder):
        agent = MagicMock()
        tools_trait = MagicMock()
        tools_trait.has_tools.return_value = True
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search the web"
        tool.parameters = {"query": {"type": "string"}}
        tools_trait.registry.list_tools.return_value = [tool]
        agent.get_trait.return_value = tools_trait

        backend = MagicMock()
        trait = SAIATrait(agent, backend)

        trait.on_start()

        mock_saia_builder.tools.assert_called_once()

    def test_on_start_no_system_prompt(self, mock_saia_builder):
        agent = MagicMock()
        agent.get_trait.return_value = None

        trait = SAIATrait(agent, MagicMock(), config=SAIAConfig(system_prompt=None))

        trait.on_start()

        mock_saia_builder.system.assert_not_called()

    def test_on_stop(self):
        agent = MagicMock()
        trait = SAIATrait(agent, MagicMock())
        trait._saia = MagicMock()

        trait.on_stop()

        assert trait._saia is None


# ---------------------------------------------------------------------------
# to_execution_result
# ---------------------------------------------------------------------------


class TestToExecutionResult:
    def test_with_score(self):
        agent = MagicMock()
        trait = SAIATrait(agent, MagicMock())

        saia_result = MagicMock()
        saia_result.completed = True
        saia_result.output = "Task done"
        saia_result.iterations = 3
        saia_result.score.total_tokens = 500
        saia_result.trace.trace_id = "abc123"

        result = trait.to_execution_result(saia_result)

        assert result.success is True
        assert result.content == "Task done"
        assert result.iterations == 3
        assert result.tokens_used == 500
        assert result.trace_id == "abc123"

    def test_without_score(self):
        agent = MagicMock()
        trait = SAIATrait(agent, MagicMock())

        saia_result = MagicMock()
        saia_result.completed = False
        saia_result.output = "Failed"
        saia_result.iterations = 1
        saia_result.score = None
        saia_result.trace.trace_id = "def456"

        result = trait.to_execution_result(saia_result)

        assert result.success is False
        assert result.tokens_used == 0


# ---------------------------------------------------------------------------
# _tool_to_tooldef
# ---------------------------------------------------------------------------


class TestToolToTooldef:
    def test_conversion(self):
        tool = MagicMock()
        tool.name = "web_search"
        tool.description = "Search the web"
        tool.parameters = {"query": {"type": "string"}}

        td = _tool_to_tooldef(tool)

        assert td.name == "web_search"
        assert td.description == "Search the web"
        assert td.parameters == {"query": {"type": "string"}}


# ---------------------------------------------------------------------------
# _create_executor
# ---------------------------------------------------------------------------


class TestCreateExecutor:
    @pytest.mark.asyncio
    async def test_success(self):
        registry = MagicMock()
        tool = MagicMock()
        result_mock = MagicMock()
        result_mock.success = True
        result_mock.output = "search results"
        tool.execute.return_value = result_mock
        registry.get.return_value = tool

        executor = _create_executor(registry, MagicMock())
        result = await executor("search", {"query": "test"})

        assert result == "search results"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        registry = MagicMock()
        registry.get.return_value = None

        executor = _create_executor(registry, MagicMock())
        result = await executor("nonexistent", {})

        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_tool_failure(self):
        registry = MagicMock()
        tool = MagicMock()
        result_mock = MagicMock()
        result_mock.success = False
        result_mock.error = "rate limited"
        tool.execute.return_value = result_mock
        registry.get.return_value = tool

        executor = _create_executor(registry, MagicMock())
        result = await executor("search", {})

        assert "Error" in result
        assert "rate limited" in result

    @pytest.mark.asyncio
    async def test_tool_exception(self):
        registry = MagicMock()
        tool = MagicMock()
        tool.execute.side_effect = RuntimeError("crash")
        registry.get.return_value = tool

        lg = MagicMock()
        executor = _create_executor(registry, lg)
        result = await executor("search", {})

        assert "Error executing" in result
        lg.warning.assert_called_once()
