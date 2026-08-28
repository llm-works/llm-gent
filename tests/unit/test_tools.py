# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for tool use infrastructure."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_gent import (
    BaseTool,
    FileReadTool,
    FileWriteTool,
    HTTPFetchTool,
    Registry,
    ShellTool,
    Tool,
    ToolCall,
    ToolResult,
)


pytestmark = pytest.mark.unit


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        result = ToolResult(success=True, output="hello")

        assert result.success is True
        assert result.output == "hello"
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, output="", error="something went wrong")

        assert result.success is False
        assert result.output == ""
        assert result.error == "something went wrong"


class TestToolCall:
    """Tests for ToolCall."""

    def test_tool_call(self):
        call = ToolCall(id="call-1", name="shell", arguments={"command": "ls"})

        assert call.id == "call-1"
        assert call.name == "shell"
        assert call.arguments == {"command": "ls"}

    def test_tool_call_empty_arguments(self):
        call = ToolCall(id="call-1", name="noop")

        assert call.arguments == {}


class TestBaseTool:
    """Tests for BaseTool."""

    def test_protocol_compliance(self):
        class MyTool(BaseTool):
            name = "my_tool"
            description = "A test tool"
            parameters = {"type": "object", "properties": {}}

            def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")

        tool = MyTool()
        assert isinstance(tool, Tool)

    def test_to_openai_function(self):
        class MyTool(BaseTool):
            name = "my_tool"
            description = "A test tool"
            parameters = {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            }

            def execute(self, arg: str) -> ToolResult:
                return ToolResult(success=True, output=arg)

        tool = MyTool()
        func = tool.to_openai_function()

        assert func == {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                },
            },
        }


class TestRegistry:
    """Tests for Registry."""

    def test_register_and_get(self):
        registry = Registry()
        tool = ShellTool()

        registry.register(tool)

        assert registry.get("shell") is tool
        assert "shell" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self):
        registry = Registry()
        registry.register(ShellTool())

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ShellTool())

    def test_unregister(self):
        registry = Registry()
        registry.register(ShellTool())

        registry.unregister("shell")

        assert registry.get("shell") is None
        assert len(registry) == 0

    def test_unregister_unknown_raises(self):
        registry = Registry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("unknown")

    def test_list_names(self):
        registry = Registry()
        registry.register(ShellTool())

        assert registry.list_names() == ["shell"]

    def test_list_tools(self):
        registry = Registry()
        tool = ShellTool()
        registry.register(tool)

        assert registry.list_tools() == [tool]

    def test_to_openai_tools(self):
        registry = Registry()
        registry.register(ShellTool())

        tools = registry.to_openai_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "shell"


class TestShellTool:
    """Tests for ShellTool."""

    def test_execute_success(self):
        tool = ShellTool()
        result = tool.execute(command="echo hello")

        assert result.success is True
        assert "hello" in result.output

    def test_execute_failure(self):
        tool = ShellTool()
        result = tool.execute(command="exit 1")

        assert result.success is False
        assert result.error is not None
        assert "exit" in result.error.lower() or "code 1" in result.error.lower()

    def test_execute_with_working_dir(self, tmp_path):
        tool = ShellTool(working_dir=str(tmp_path))
        result = tool.execute(command="pwd")

        assert result.success is True
        assert str(tmp_path) in result.output

    def test_execute_timeout(self):
        tool = ShellTool(timeout=0.1)
        result = tool.execute(command="sleep 10")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_execute_allowed_commands(self):
        tool = ShellTool(allowed_commands=["echo", "ls"])

        # Allowed command works
        result = tool.execute(command="echo hello")
        assert result.success is True

        # Disallowed command fails
        result = tool.execute(command="rm -rf /")
        assert result.success is False
        assert "not in allowed list" in result.error

    def test_empty_command_with_allowlist(self):
        """Empty or whitespace-only commands return clear error."""
        tool = ShellTool(allowed_commands=["echo", "ls"])

        # Whitespace-only command gets clear error (empty string caught earlier)
        result = tool.execute(command="   ")
        assert result.success is False
        assert "Empty command" in result.error

    def test_shell_metacharacters_blocked_with_allowlist(self):
        """Shell metacharacters are blocked when allowlist is enabled to prevent bypass."""
        tool = ShellTool(allowed_commands=["echo", "ls"])

        # Command chaining with semicolon
        result = tool.execute(command="echo hello; rm -rf /")
        assert result.success is False
        assert "metacharacters" in result.error

        # Command chaining with &&
        result = tool.execute(command="echo hello && cat /etc/passwd")
        assert result.success is False
        assert "metacharacters" in result.error

        # Pipe
        result = tool.execute(command="ls | grep secret")
        assert result.success is False
        assert "metacharacters" in result.error

        # Command substitution
        result = tool.execute(command="echo $(whoami)")
        assert result.success is False
        assert "metacharacters" in result.error

        # Backticks
        result = tool.execute(command="echo `id`")
        assert result.success is False
        assert "metacharacters" in result.error

        # Variable expansion with braces ${VAR}
        result = tool.execute(command="echo ${PATH}")
        assert result.success is False
        assert "metacharacters" in result.error

        # Variable reference $VAR
        result = tool.execute(command="echo $HOME")
        assert result.success is False
        assert "metacharacters" in result.error

        # Output redirection
        result = tool.execute(command="echo x > /tmp/test")
        assert result.success is False
        assert "metacharacters" in result.error

        # Append redirection
        result = tool.execute(command="echo x >> /tmp/test")
        assert result.success is False
        assert "metacharacters" in result.error

        # Input redirection
        result = tool.execute(command="cat < /etc/passwd")
        assert result.success is False
        assert "metacharacters" in result.error

        # Heredoc
        result = tool.execute(command="cat << EOF")
        assert result.success is False
        assert "metacharacters" in result.error

    def test_invalid_command_type(self):
        """Non-string command argument is rejected."""
        tool = ShellTool()

        # Integer instead of string
        result = tool.execute(command=123)
        assert result.success is False
        assert "invalid" in result.error.lower()

        # None
        result = tool.execute(command=None)
        assert result.success is False
        assert "invalid" in result.error.lower()

        # List
        result = tool.execute(command=["echo", "hello"])
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_shell_metacharacters_allowed_without_allowlist(self):
        """Shell metacharacters work normally when no allowlist is configured."""
        tool = ShellTool()

        # Pipe should work
        result = tool.execute(command="echo hello | cat")
        assert result.success is True
        assert "hello" in result.output

    def test_output_truncation(self):
        tool = ShellTool(max_output_chars=50)
        result = tool.execute(command="echo " + "x" * 100)

        assert "truncated" in result.output
        assert len(result.output) < 150  # Some buffer for truncation message

    def test_tool_properties(self):
        tool = ShellTool()

        assert tool.name == "shell"
        assert "command" in tool.description.lower()
        assert tool.parameters["type"] == "object"
        assert "command" in tool.parameters["properties"]


class TestFileReadTool:
    """Tests for FileReadTool."""

    def test_read_file_success(self, tmp_path):
        """Read a simple file successfully."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt")

        assert result.success is True
        assert "line 1" in result.output
        assert "line 2" in result.output
        assert "line 3" in result.output

    def test_read_file_with_line_numbers(self, tmp_path):
        """Output includes line numbers."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("first\nsecond\nthird")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt")

        assert result.success is True
        assert "1│" in result.output or "1|" in result.output
        assert "2│" in result.output or "2|" in result.output
        assert "3│" in result.output or "3|" in result.output

    def test_read_file_absolute_path(self, tmp_path):
        """Read file using absolute path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content here")

        tool = FileReadTool()
        result = tool.execute(path=str(test_file))

        assert result.success is True
        assert "content here" in result.output

    def test_read_file_not_found(self, tmp_path):
        """Return error when file doesn't exist."""
        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="nonexistent.txt")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_directory_fails(self, tmp_path):
        """Return error when path is a directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="subdir")

        assert result.success is False
        assert "not a file" in result.error.lower()

    def test_read_with_offset(self, tmp_path):
        """Read from specific line offset."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", offset=3)

        assert result.success is True
        assert "line 3" in result.output
        assert "line 4" in result.output
        assert "line 5" in result.output
        assert "line 1" not in result.output
        assert "line 2" not in result.output

    def test_read_with_limit(self, tmp_path):
        """Read limited number of lines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", limit=2)

        assert result.success is True
        assert "line 1" in result.output
        assert "line 2" in result.output
        assert "line 3" not in result.output

    def test_read_with_offset_and_limit(self, tmp_path):
        """Read specific range of lines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", offset=2, limit=2)

        assert result.success is True
        assert "line 2" in result.output
        assert "line 3" in result.output
        assert "line 1" not in result.output
        assert "line 4" not in result.output

    def test_read_offset_past_end(self, tmp_path):
        """Handle offset past end of file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", offset=100)

        assert result.success is True
        assert "past end" in result.output.lower() or "2 lines" in result.output.lower()

    def test_read_empty_file(self, tmp_path):
        """Handle empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="empty.txt")

        assert result.success is True
        assert "Empty file" in result.output

    def test_max_lines_truncation(self, tmp_path):
        """Truncate output when exceeding max_lines."""
        test_file = tmp_path / "big.txt"
        test_file.write_text("\n".join(f"line {i}" for i in range(100)))

        tool = FileReadTool(working_dir=str(tmp_path), max_lines=10)
        result = tool.execute(path="big.txt")

        assert result.success is True
        assert "line 0" in result.output
        assert "line 9" in result.output
        assert "more lines" in result.output.lower()

    def test_max_chars_truncation(self, tmp_path):
        """Truncate output when exceeding max_chars."""
        test_file = tmp_path / "big.txt"
        test_file.write_text("x" * 1000)

        tool = FileReadTool(working_dir=str(tmp_path), max_chars=100)
        result = tool.execute(path="big.txt")

        assert result.success is True
        assert "truncated" in result.output.lower()

    def test_allowed_paths_success(self, tmp_path):
        """Allow reading from allowed paths."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("allowed content")

        tool = FileReadTool(allowed_paths=[str(tmp_path)])
        result = tool.execute(path=str(test_file))

        assert result.success is True
        assert "allowed content" in result.output

    def test_allowed_paths_blocked(self, tmp_path):
        """Block reading from paths outside allowed list."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        blocked_dir = tmp_path / "blocked"
        blocked_dir.mkdir()
        blocked_file = blocked_dir / "secret.txt"
        blocked_file.write_text("secret content")

        tool = FileReadTool(allowed_paths=[str(allowed_dir)])
        result = tool.execute(path=str(blocked_file))

        assert result.success is False
        assert "not under allowed" in result.error.lower()

    def test_symlink_escape_blocked(self, tmp_path):
        """Block symlinks that point outside allowed paths."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        secret_file = secret_dir / "password.txt"
        secret_file.write_text("SECRET")

        # Create symlink inside allowed dir pointing to secret file
        symlink = allowed_dir / "link"
        symlink.symlink_to(secret_file)

        tool = FileReadTool(allowed_paths=[str(allowed_dir)])
        result = tool.execute(path=str(symlink))

        assert result.success is False
        assert "not under allowed" in result.error.lower()

    def test_permission_denied(self, tmp_path):
        """Return error when file is not readable."""
        import os

        test_file = tmp_path / "noperm.txt"
        test_file.write_text("content")
        os.chmod(test_file, 0o000)

        try:
            tool = FileReadTool(working_dir=str(tmp_path))
            result = tool.execute(path="noperm.txt")

            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            os.chmod(test_file, 0o644)  # Restore for cleanup

    def test_binary_file_error(self, tmp_path):
        """Return error when file contains non-UTF-8 content."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x80\x81\x82\xff\xfe")

        tool = FileReadTool(working_dir=str(tmp_path))
        result = tool.execute(path="binary.bin")

        assert result.success is False
        assert "utf-8" in result.error.lower() or "binary" in result.error.lower()

    def test_missing_path_argument(self):
        """Return error when path argument is missing."""
        tool = FileReadTool()

        result = tool.execute()
        assert result.success is False
        assert "missing" in result.error.lower()

        result = tool.execute(path="")
        assert result.success is False
        assert "missing" in result.error.lower() or "invalid" in result.error.lower()

    def test_invalid_path_type(self):
        """Return error when path is not a string."""
        tool = FileReadTool()

        result = tool.execute(path=123)
        assert result.success is False
        assert "invalid" in result.error.lower()

        result = tool.execute(path=None)
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_tool_properties(self):
        """Tool has correct properties."""
        tool = FileReadTool()

        assert tool.name == "read_file"
        assert "file" in tool.description.lower()
        assert tool.parameters["type"] == "object"
        assert "path" in tool.parameters["properties"]
        assert "offset" in tool.parameters["properties"]
        assert "limit" in tool.parameters["properties"]

    def test_to_openai_function(self):
        """Converts to OpenAI function format."""
        tool = FileReadTool()
        func = tool.to_openai_function()

        assert func["type"] == "function"
        assert func["function"]["name"] == "read_file"
        assert "path" in func["function"]["parameters"]["properties"]


class TestFileWriteTool:
    """Tests for FileWriteTool."""

    def test_write_new_file(self, tmp_path):
        """Write a new file successfully."""
        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", content="Hello, world!")

        assert result.success is True
        assert "Created" in result.output
        assert (tmp_path / "test.txt").read_text() == "Hello, world!"

    def test_write_overwrite_existing(self, tmp_path):
        """Overwrite an existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("old content")

        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", content="new content")

        assert result.success is True
        assert "Updated" in result.output
        assert test_file.read_text() == "new content"

    def test_write_append_mode(self, tmp_path):
        """Append to existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\n")

        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", content="line 2\n", mode="append")

        assert result.success is True
        assert "Appended" in result.output
        assert test_file.read_text() == "line 1\nline 2\n"

    def test_write_append_creates_file(self, tmp_path):
        """Append mode creates file if it doesn't exist."""
        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="new.txt", content="content", mode="append")

        assert result.success is True
        assert (tmp_path / "new.txt").read_text() == "content"

    def test_write_creates_parent_dirs(self, tmp_path):
        """Create parent directories when needed."""
        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="a/b/c/test.txt", content="nested")

        assert result.success is True
        assert (tmp_path / "a/b/c/test.txt").read_text() == "nested"

    def test_write_no_create_dirs(self, tmp_path):
        """Fail when parent dirs don't exist and create_dirs=False."""
        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="missing/test.txt", content="content", create_dirs=False)

        assert result.success is False
        assert "does not exist" in result.error.lower()

    def test_write_absolute_path(self, tmp_path):
        """Write using absolute path."""
        tool = FileWriteTool()
        target = tmp_path / "abs.txt"
        result = tool.execute(path=str(target), content="absolute path test")

        assert result.success is True
        assert target.read_text() == "absolute path test"

    def test_write_directory_fails(self, tmp_path):
        """Fail when path is an existing directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="subdir", content="content")

        assert result.success is False
        assert "not a file" in result.error.lower()

    def test_write_allowed_paths_success(self, tmp_path):
        """Allow writing to allowed paths."""
        tool = FileWriteTool(allowed_paths=[str(tmp_path)])
        target = tmp_path / "allowed.txt"
        result = tool.execute(path=str(target), content="allowed")

        assert result.success is True
        assert target.read_text() == "allowed"

    def test_write_allowed_paths_blocked(self, tmp_path):
        """Block writing outside allowed paths."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        blocked_dir = tmp_path / "blocked"
        blocked_dir.mkdir()

        tool = FileWriteTool(allowed_paths=[str(allowed_dir)])
        result = tool.execute(path=str(blocked_dir / "test.txt"), content="blocked")

        assert result.success is False
        assert "not under allowed" in result.error.lower()

    def test_write_symlink_escape_blocked(self, tmp_path):
        """Block symlinks that escape allowed paths."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()

        # Create symlink inside allowed dir pointing to secret dir
        symlink = allowed_dir / "escape"
        symlink.symlink_to(secret_dir)

        tool = FileWriteTool(allowed_paths=[str(allowed_dir)])
        result = tool.execute(path=str(symlink / "test.txt"), content="escaped")

        assert result.success is False
        assert "not under allowed" in result.error.lower()

    def test_write_max_size_exceeded(self, tmp_path):
        """Fail when content exceeds max size."""
        tool = FileWriteTool(working_dir=str(tmp_path), max_write_size=100)
        result = tool.execute(path="big.txt", content="x" * 200)

        assert result.success is False
        assert "exceeds maximum" in result.error.lower()

    def test_write_permission_denied(self, tmp_path):
        """Handle permission denied error."""
        import os

        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, 0o444)

        try:
            tool = FileWriteTool(working_dir=str(tmp_path))
            result = tool.execute(path="readonly/test.txt", content="content", create_dirs=False)

            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            os.chmod(readonly_dir, 0o755)

    def test_write_missing_path_argument(self):
        """Fail when path argument is missing."""
        tool = FileWriteTool()

        result = tool.execute(content="content")
        assert result.success is False
        assert "missing" in result.error.lower()

        result = tool.execute(path="", content="content")
        assert result.success is False

    def test_write_missing_content_argument(self, tmp_path):
        """Fail when content argument is missing."""
        tool = FileWriteTool(working_dir=str(tmp_path))

        result = tool.execute(path="test.txt")
        assert result.success is False
        assert "content" in result.error.lower()

    def test_write_invalid_path_type(self):
        """Fail when path is not a string."""
        tool = FileWriteTool()

        result = tool.execute(path=123, content="content")
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_write_output_shows_stats(self, tmp_path):
        """Output shows line and byte counts."""
        tool = FileWriteTool(working_dir=str(tmp_path))
        result = tool.execute(path="test.txt", content="line 1\nline 2\nline 3")

        assert result.success is True
        assert "3 lines" in result.output
        assert "bytes" in result.output

    def test_tool_properties(self):
        """Tool has correct properties."""
        tool = FileWriteTool()

        assert tool.name == "write_file"
        assert "file" in tool.description.lower()
        assert tool.parameters["type"] == "object"
        assert "path" in tool.parameters["properties"]
        assert "content" in tool.parameters["properties"]
        assert "mode" in tool.parameters["properties"]
        assert "create_dirs" in tool.parameters["properties"]

    def test_to_openai_function(self):
        """Converts to OpenAI function format."""
        tool = FileWriteTool()
        func = tool.to_openai_function()

        assert func["type"] == "function"
        assert func["function"]["name"] == "write_file"
        assert "path" in func["function"]["parameters"]["properties"]
        assert "content" in func["function"]["parameters"]["properties"]

    def test_write_invalid_mode(self, tmp_path):
        """Fail when mode is invalid."""
        tool = FileWriteTool(working_dir=str(tmp_path))

        result = tool.execute(path="test.txt", content="content", mode="invalid")
        assert result.success is False
        assert "invalid mode" in result.error.lower()
        assert "overwrite" in result.error.lower()
        assert "append" in result.error.lower()

    def test_write_invalid_create_dirs(self, tmp_path):
        """Fail when create_dirs is not a boolean."""
        tool = FileWriteTool(working_dir=str(tmp_path))

        result = tool.execute(path="test.txt", content="content", create_dirs="yes")
        assert result.success is False
        assert "create_dirs" in result.error.lower()
        assert "boolean" in result.error.lower()


class TestHTTPFetchTool:
    """Tests for HTTPFetchTool."""

    def _mock_response(
        self,
        status_code: int = 200,
        text: str = "response body",
        headers: dict | None = None,
        reason_phrase: str = "OK",
    ) -> MagicMock:
        """Create a mock httpx.Response."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        response.text = text
        response.headers = headers or {}
        response.reason_phrase = reason_phrase
        response.is_success = 200 <= status_code < 300
        return response

    def _mock_dns_public(self, ip: str = "1.2.3.4"):
        """Create a mock for socket.getaddrinfo returning a public IP."""
        import socket

        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    def _mock_dns_private(self, ip: str = "192.168.1.1"):
        """Create a mock for socket.getaddrinfo returning a private IP."""
        import socket

        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    def test_fetch_success(self):
        """Fetch URL successfully."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(text="Hello, World!")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert "Hello, World!" in result.output

    def test_fetch_with_headers(self):
        """Fetch with custom headers."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(text="authenticated response")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.get.return_value = mock_response

            result = tool.execute(
                url="https://api.example.com/data",
                headers={"Authorization": "Bearer token123"},
            )

            # Verify headers were passed
            call_args = mock_instance.get.call_args
            assert "Authorization" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Authorization"] == "Bearer token123"

        assert result.success is True

    def test_fetch_with_default_headers(self):
        """Default headers are included in requests."""
        tool = HTTPFetchTool(default_headers={"User-Agent": "TestBot/1.0"})
        mock_response = self._mock_response()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.get.return_value = mock_response

            tool.execute(url="https://example.com")

            call_args = mock_instance.get.call_args
            assert call_args.kwargs["headers"]["User-Agent"] == "TestBot/1.0"

    def test_fetch_http_error(self):
        """Handle HTTP error responses."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(
            status_code=404,
            text="Not Found",
            reason_phrase="Not Found",
        )

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com/missing")

        assert result.success is False
        assert "404" in result.error
        assert "Not Found" in result.output

    def test_fetch_timeout(self):
        """Handle request timeout."""
        tool = HTTPFetchTool(timeout=5.0)

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.side_effect = (
                httpx.TimeoutException("timed out")
            )
            result = tool.execute(url="https://slow.example.com")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_fetch_connection_error(self):
        """Handle connection errors."""
        tool = HTTPFetchTool()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            result = tool.execute(url="https://unreachable.example.com")

        assert result.success is False
        assert "connection" in result.error.lower()

    def test_fetch_response_too_large_header(self):
        """Reject responses that exceed max size (from Content-Length header)."""
        tool = HTTPFetchTool(max_response_size=1000)
        mock_response = self._mock_response(
            text="x" * 100,
            headers={"content-length": "2000"},
        )

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com/large")

        assert result.success is False
        assert "too large" in result.error.lower()

    def test_fetch_invalid_content_length_header(self):
        """Handle malformed Content-Length header gracefully."""
        tool = HTTPFetchTool(max_response_size=1000)
        mock_response = self._mock_response(
            text="valid response",
            headers={"content-length": "not-a-number"},
        )

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        # Should succeed - invalid Content-Length is ignored, response is read normally
        assert result.success is True
        assert "valid response" in result.output

    def test_fetch_response_truncated(self):
        """Truncate responses that exceed max size."""
        tool = HTTPFetchTool(max_response_size=100)
        mock_response = self._mock_response(text="x" * 200)

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert len(result.output) < 200
        assert "truncated" in result.output.lower()

    def test_allowed_domains_success(self):
        """Allow requests to allowed domains."""
        tool = HTTPFetchTool(allowed_domains=["api.github.com"])
        mock_response = self._mock_response(text="github data")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://api.github.com/repos")

        assert result.success is True

    def test_allowed_domains_blocked(self):
        """Block requests to non-allowed domains."""
        tool = HTTPFetchTool(allowed_domains=["api.github.com"])

        result = tool.execute(url="https://evil.com/steal-data")

        assert result.success is False
        assert "not in allowed" in result.error.lower()

    def test_allowed_domains_subdomain(self):
        """Subdomains of allowed domains are permitted."""
        tool = HTTPFetchTool(allowed_domains=["github.com"])
        mock_response = self._mock_response()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://api.github.com/repos")

        assert result.success is True

    def test_blocked_domains(self):
        """Block requests to blocked domains."""
        tool = HTTPFetchTool(blocked_domains=["evil.com", "malware.org"])

        result = tool.execute(url="https://evil.com/bad")

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocked_domains_subdomain(self):
        """Subdomains of blocked domains are also blocked."""
        tool = HTTPFetchTool(blocked_domains=["evil.com"])

        result = tool.execute(url="https://api.evil.com/endpoint")

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocked_domains_allowed(self):
        """Non-blocked domains are allowed."""
        tool = HTTPFetchTool(blocked_domains=["evil.com"])
        mock_response = self._mock_response()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://good.com/api")

        assert result.success is True

    def test_invalid_url_scheme(self):
        """Reject non-HTTP schemes."""
        tool = HTTPFetchTool()

        result = tool.execute(url="ftp://files.example.com/data.txt")

        assert result.success is False
        assert "scheme" in result.error.lower()

    def test_invalid_url_missing_domain(self):
        """Reject URLs without domain."""
        tool = HTTPFetchTool()

        result = tool.execute(url="https:///path/only")

        assert result.success is False
        assert "domain" in result.error.lower() or "invalid" in result.error.lower()

    def test_missing_url_argument(self):
        """Fail when URL argument is missing."""
        tool = HTTPFetchTool()

        result = tool.execute()
        assert result.success is False
        assert "missing" in result.error.lower()

        result = tool.execute(url="")
        assert result.success is False

    def test_invalid_url_type(self):
        """Fail when URL is not a string."""
        tool = HTTPFetchTool()

        result = tool.execute(url=123)
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_invalid_headers_type(self):
        """Fail when headers is not a dict."""
        tool = HTTPFetchTool()

        result = tool.execute(url="https://example.com", headers="not-a-dict")
        assert result.success is False
        assert "headers" in result.error.lower()

    def test_invalid_header_values(self):
        """Fail when header values are not strings."""
        tool = HTTPFetchTool()

        result = tool.execute(url="https://example.com", headers={"count": 123})
        assert result.success is False
        assert "string" in result.error.lower()

    def test_tool_properties(self):
        """Tool has correct properties."""
        tool = HTTPFetchTool()

        assert tool.name == "http_fetch"
        assert "url" in tool.description.lower() or "http" in tool.description.lower()
        assert tool.parameters["type"] == "object"
        assert "url" in tool.parameters["properties"]
        assert "headers" in tool.parameters["properties"]

    def test_to_openai_function(self):
        """Converts to OpenAI function format."""
        tool = HTTPFetchTool()
        func = tool.to_openai_function()

        assert func["type"] == "function"
        assert func["function"]["name"] == "http_fetch"
        assert "url" in func["function"]["parameters"]["properties"]

    def test_domain_with_port(self):
        """Handle domains with port numbers."""
        # Use block_private_ips=False since this test is about port handling, not SSRF
        tool = HTTPFetchTool(allowed_domains=["localhost"], block_private_ips=False)
        mock_response = self._mock_response()

        with patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="http://localhost:8080/api")

        assert result.success is True

    # SSRF Protection Tests

    def test_ssrf_blocks_localhost(self):
        """Block requests to localhost (loopback)."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("127.0.0.1")
            result = tool.execute(url="http://localhost/admin")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_private_ip_10(self):
        """Block requests to 10.x.x.x private network."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("10.0.0.1")
            result = tool.execute(url="http://internal-server/api")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_private_ip_172(self):
        """Block requests to 172.16.x.x private network."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("172.16.0.1")
            result = tool.execute(url="http://internal-server/api")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_private_ip_192(self):
        """Block requests to 192.168.x.x private network."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("192.168.1.1")
            result = tool.execute(url="http://router.local/admin")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_link_local(self):
        """Block requests to link-local addresses (169.254.x.x) including cloud metadata."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("169.254.169.254")
            result = tool.execute(url="http://169.254.169.254/latest/meta-data/")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_ipv6_loopback(self):
        """Block requests to IPv6 loopback (::1)."""
        import socket

        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]
            result = tool.execute(url="http://localhost/admin")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_allows_public_ip(self):
        """Allow requests to public IP addresses."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(text="public content")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        assert result.success is True

    def test_ssrf_protection_disabled(self):
        """Allow private IPs when block_private_ips=False."""
        tool = HTTPFetchTool(block_private_ips=False)
        mock_response = self._mock_response(text="internal content")

        with patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="http://localhost/internal")

        assert result.success is True

    def test_ssrf_dns_resolution_failure(self):
        """Handle DNS resolution failures gracefully."""
        import socket

        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror(8, "Name does not resolve")
            result = tool.execute(url="http://nonexistent.invalid/api")

        assert result.success is False
        assert "dns" in result.error.lower() or "resolution" in result.error.lower()


# Note: Tool execution loop tests have been moved to llm-saia.
# SAIA now handles the LLM ↔ tool interaction loop.


class TestRememberTool:
    """Tests for RememberTool."""

    def test_remember_success(self):
        """Store a fact successfully."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.remember.return_value = 42

        tool = RememberTool(mock_memory_trait)
        result = tool.execute(fact="User prefers Python")

        assert result.success is True
        assert "42" in result.output
        assert "User prefers Python" in result.output
        mock_memory_trait.remember.assert_called_once_with(
            fact="User prefers Python",
            category="general",
            source="inferred",
        )

    def test_remember_with_category(self):
        """Store a fact with category."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.remember.return_value = 123

        tool = RememberTool(mock_memory_trait)
        result = tool.execute(fact="Uses vim", category="preferences")

        assert result.success is True
        mock_memory_trait.remember.assert_called_once_with(
            fact="Uses vim",
            category="preferences",
            source="inferred",
        )

    def test_remember_failure(self):
        """Handle store failure gracefully."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.remember.side_effect = RuntimeError("Database error")

        tool = RememberTool(mock_memory_trait)
        result = tool.execute(fact="Some fact")

        assert result.success is False
        assert "Database error" in result.error

    def test_tool_properties(self):
        """Tool has correct properties."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        tool = RememberTool(mock_memory_trait)

        assert tool.name == "remember"
        assert "fact" in tool.parameters["properties"]
        assert "category" in tool.parameters["properties"]
        assert "fact" in tool.parameters["required"]


class TestRecallTool:
    """Tests for RecallTool."""

    def test_recall_with_embedder(self):
        """Search facts using semantic search."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_fact = MagicMock()
        mock_fact.content = "User prefers Python"
        mock_fact.category = "preferences"

        mock_scored_entity = MagicMock()
        mock_scored_entity.entity = mock_fact
        mock_scored_entity.score = 0.85

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.return_value = [mock_scored_entity]

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="programming language")

        assert result.success is True
        assert "User prefers Python" in result.output
        assert "0.85" in result.output
        mock_memory_trait.recall.assert_called_once()

    def test_recall_without_embedder(self):
        """Fall back to listing facts without semantic search."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_fact = MagicMock()
        mock_fact.content = "Uses vim"
        mock_fact.category = "tools"

        # Build mock chain avoiding 'assertions' conflict with MagicMock
        mock_assertions = MagicMock()
        mock_assertions.list.return_value = [mock_fact]
        mock_atomic = MagicMock()
        mock_atomic.assertions = mock_assertions
        mock_learn = MagicMock()
        mock_learn.atomic = mock_atomic

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = False
        mock_memory_trait.kelt = mock_learn

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="editor")

        assert result.success is True
        assert "Uses vim" in result.output

    def test_recall_no_results(self):
        """Handle no matching facts."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.return_value = []

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="nonexistent topic")

        assert result.success is True
        assert "No matching facts" in result.output

    def test_recall_with_category_filter(self):
        """Filter recall by category."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.return_value = []

        tool = RecallTool(mock_memory_trait)
        tool.execute(query="test", category="preferences")

        mock_memory_trait.recall.assert_called_once_with(
            query="test",
            top_k=5,
            min_similarity=0.3,
            categories=["preferences"],
        )

    def test_recall_failure(self):
        """Handle recall failure gracefully."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.side_effect = RuntimeError("Embedder error")

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="test")

        assert result.success is False
        assert "Embedder error" in result.error

    def test_tool_properties(self):
        """Tool has correct properties."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        tool = RecallTool(mock_memory_trait)

        assert tool.name == "recall"
        assert "query" in tool.parameters["properties"]
        assert "limit" in tool.parameters["properties"]
        assert "category" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]


class TestCompleteTaskTool:
    """Tests for CompleteTaskTool."""

    def test_complete_done(self):
        """Complete with done status."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="The answer is 42.")

        assert result.success is True
        assert result.terminal is True
        assert result.terminal_data == {
            "status": "done",
            "conclusion": "The answer is 42.",
        }
        assert "done" in result.output

    def test_complete_stuck(self):
        """Complete with stuck status."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="stuck", conclusion="Need database access.")

        assert result.success is True
        assert result.terminal is True
        assert result.terminal_data == {
            "status": "stuck",
            "conclusion": "Need database access.",
        }
        assert "stuck" in result.output

    def test_invalid_status(self):
        """Invalid status returns error."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="invalid", conclusion="test")

        assert result.success is False
        assert result.terminal is False
        assert "Invalid status" in result.error

    def test_missing_conclusion(self):
        """Missing conclusion returns error."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="")

        assert result.success is False
        assert "required" in result.error.lower()

    def test_whitespace_only_conclusion(self):
        """Whitespace-only conclusion is rejected."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="   \n\t  ")

        assert result.success is False
        assert "required" in result.error.lower()

    def test_conclusion_is_trimmed(self):
        """Conclusion whitespace is trimmed."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="  The answer is 42.  ")

        assert result.success is True
        assert result.terminal_data["conclusion"] == "The answer is 42."

    def test_tool_properties(self):
        """Tool has correct properties."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()

        assert tool.name == "complete_task"
        assert "status" in tool.parameters["properties"]
        assert "conclusion" in tool.parameters["properties"]
        assert "status" in tool.parameters["required"]
        assert "conclusion" in tool.parameters["required"]
        assert tool.parameters["properties"]["status"]["enum"] == ["done", "stuck"]
