"""Tests for SimpleToolExecutor."""

from unittest.mock import MagicMock

import pytest

from llm_gent.core.tools.base import ToolCall, ToolResult
from llm_gent.core.tools.executor import SimpleToolExecutor


pytestmark = pytest.mark.unit


def _make_executor(
    registry_tools: dict[str, MagicMock] | None = None,
) -> tuple[SimpleToolExecutor, MagicMock, MagicMock]:
    """Create executor with mock logger and registry.

    Args:
        registry_tools: Map of tool name -> mock tool. Registry.get returns
            the mock if the name matches, None otherwise.

    Returns:
        (executor, mock_logger, mock_registry)
    """
    lg = MagicMock()
    registry = MagicMock()
    registry.get.side_effect = lambda name: (registry_tools or {}).get(name)
    return SimpleToolExecutor(lg, registry), lg, registry


def _make_tool(result: ToolResult) -> MagicMock:
    """Create a mock tool that returns the given result."""
    tool = MagicMock()
    tool.execute.return_value = result
    return tool


class TestExecute:
    """Tests for SimpleToolExecutor.execute()."""

    def test_successful_tool_call(self):
        result = ToolResult(success=True, output="done")
        tool = _make_tool(result)
        executor, _, _ = _make_executor({"my_tool": tool})

        call = ToolCall(id="c1", name="my_tool", arguments={"x": 1})
        results = executor.execute([call], parse_errors={})

        assert len(results) == 1
        assert results[0].call_id == "c1"
        assert results[0].name == "my_tool"
        assert results[0].result.success is True
        assert results[0].result.output == "done"
        tool.execute.assert_called_once_with(x=1)

    def test_multiple_tool_calls(self):
        t1 = _make_tool(ToolResult(success=True, output="a"))
        t2 = _make_tool(ToolResult(success=True, output="b"))
        executor, _, _ = _make_executor({"t1": t1, "t2": t2})

        calls = [
            ToolCall(id="c1", name="t1", arguments={}),
            ToolCall(id="c2", name="t2", arguments={"k": "v"}),
        ]
        results = executor.execute(calls, parse_errors={})

        assert len(results) == 2
        assert results[0].result.output == "a"
        assert results[1].result.output == "b"

    def test_parse_error_returns_error_without_executing(self):
        tool = _make_tool(ToolResult(success=True, output="should not run"))
        executor, _, _ = _make_executor({"my_tool": tool})

        call = ToolCall(id="c1", name="my_tool", arguments={})
        results = executor.execute([call], parse_errors={"c1": "bad JSON"})

        assert len(results) == 1
        assert results[0].result.success is False
        assert results[0].result.error == "bad JSON"
        tool.execute.assert_not_called()

    def test_parse_error_only_affects_matching_call(self):
        t1 = _make_tool(ToolResult(success=True, output="ok"))
        t2 = _make_tool(ToolResult(success=True, output="also ok"))
        executor, _, _ = _make_executor({"t1": t1, "t2": t2})

        calls = [
            ToolCall(id="c1", name="t1", arguments={}),
            ToolCall(id="c2", name="t2", arguments={}),
        ]
        results = executor.execute(calls, parse_errors={"c1": "parse fail"})

        assert results[0].result.success is False
        assert results[0].result.error == "parse fail"
        assert results[1].result.success is True
        assert results[1].result.output == "also ok"
        t1.execute.assert_not_called()
        t2.execute.assert_called_once()


class TestExecuteSingleTool:
    """Tests for _execute_single_tool edge cases."""

    def test_unknown_tool(self):
        executor, lg, _ = _make_executor()

        call = ToolCall(id="c1", name="no_such_tool", arguments={})
        results = executor.execute([call], parse_errors={})

        assert results[0].result.success is False
        assert "Unknown tool" in results[0].result.error
        assert "no_such_tool" in results[0].result.error
        lg.warning.assert_called_once()

    def test_tool_raises_exception(self):
        tool = MagicMock()
        tool.execute.side_effect = RuntimeError("boom")
        executor, lg, _ = _make_executor({"bad_tool": tool})

        call = ToolCall(id="c1", name="bad_tool", arguments={"a": 1})
        results = executor.execute([call], parse_errors={})

        assert results[0].result.success is False
        assert "Tool execution error" in results[0].result.error
        assert "boom" in results[0].result.error
        lg.warning.assert_called_once()

    def test_tool_called_with_arguments(self):
        tool = _make_tool(ToolResult(success=True, output="ok"))
        executor, _, _ = _make_executor({"t": tool})

        call = ToolCall(id="c1", name="t", arguments={"foo": "bar", "n": 42})
        executor.execute([call], parse_errors={})

        tool.execute.assert_called_once_with(foo="bar", n=42)


class TestLogToolResult:
    """Tests for _log_tool_result."""

    def test_logs_success(self):
        executor, lg, _ = _make_executor()
        result = ToolResult(success=True, output="some output")

        executor._log_tool_result("my_tool", result)

        lg.trace.assert_called_once()
        call_kwargs = lg.trace.call_args
        assert call_kwargs[1]["extra"]["success"] is True
        assert "some output" in call_kwargs[1]["extra"]["output"]

    def test_logs_failure(self):
        executor, lg, _ = _make_executor()
        result = ToolResult(success=False, output="", error="failed hard")

        executor._log_tool_result("my_tool", result)

        lg.trace.assert_called_once()
        call_kwargs = lg.trace.call_args
        assert call_kwargs[1]["extra"]["success"] is False
        assert call_kwargs[1]["extra"]["error"] == "failed hard"


class TestTruncateStr:
    """Tests for _truncate_str."""

    def test_none_returns_empty(self):
        executor, _, _ = _make_executor()
        assert executor._truncate_str(None) == ""

    def test_empty_returns_empty(self):
        executor, _, _ = _make_executor()
        assert executor._truncate_str("") == ""

    def test_short_string_unchanged(self):
        executor, _, _ = _make_executor()
        assert executor._truncate_str("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        executor, _, _ = _make_executor()
        assert executor._truncate_str("abcde", 5) == "abcde"

    def test_long_string_truncated(self):
        executor, _, _ = _make_executor()
        result = executor._truncate_str("a" * 300, 10)
        assert result == "a" * 10 + "..."
        assert len(result) == 13


class TestTruncateArgs:
    """Tests for _truncate_args."""

    def test_short_values_unchanged(self):
        executor, _, _ = _make_executor()
        args = {"a": "short", "b": 42, "c": True}
        result = executor._truncate_args(args)
        assert result == args

    def test_long_string_value_truncated(self):
        executor, _, _ = _make_executor()
        long_val = "x" * 200
        result = executor._truncate_args({"key": long_val}, max_value_len=50)
        assert result["key"] == "x" * 50 + "..."

    def test_non_string_values_unchanged(self):
        executor, _, _ = _make_executor()
        args = {"n": 999999, "lst": [1, 2, 3], "obj": {"nested": True}}
        result = executor._truncate_args(args, max_value_len=5)
        assert result == args

    def test_mixed_values(self):
        executor, _, _ = _make_executor()
        args = {"short": "ok", "long": "y" * 200, "num": 7}
        result = executor._truncate_args(args, max_value_len=10)
        assert result["short"] == "ok"
        assert result["long"] == "y" * 10 + "..."
        assert result["num"] == 7
