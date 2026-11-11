# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The shell and file tools against real processes and the file system."""

import pytest

from llm_gent import FileReadTool, FileWriteTool, ShellTool


pytestmark = pytest.mark.integration


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

    @pytest.mark.parametrize(
        "syntax",
        [
            "; touch {m}",
            "&& touch {m}",
            "|| touch {m}",
            "| touch {m}",
            "& touch {m}",
            "$(touch {m})",
            "`touch {m}`",
            "> {m}",
            ">> {m}",
        ],
    )
    def test_shell_syntax_is_plain_arguments_with_allowlist(self, tmp_path, syntax):
        """With an allowlist nothing runs in a shell: chaining, substitution and redirects run nothing."""
        marker = tmp_path / "marker"
        tool = ShellTool(allowed_commands=["echo"])

        result = tool.execute(command=f"echo hello {syntax.format(m=marker)}")

        assert result.success is True
        assert result.output.startswith("hello ")
        assert not marker.exists()

    def test_variables_are_not_expanded_with_allowlist(self):
        tool = ShellTool(allowed_commands=["echo"])

        result = tool.execute(command="echo $HOME ${PATH}")

        assert result.output.strip() == "$HOME ${PATH}"

    def test_quoted_arguments_with_allowlist(self, tmp_path):
        (tmp_path / "a file.txt").write_text("needle here\n")
        tool = ShellTool(working_dir=str(tmp_path), allowed_commands=["grep"])

        result = tool.execute(command="grep -n 'needle here' 'a file.txt'")

        assert result.success is True
        assert result.output.strip() == "1:needle here"

    def test_unbalanced_quotes_are_rejected_with_allowlist(self):
        tool = ShellTool(allowed_commands=["echo"])

        result = tool.execute(command="echo 'unterminated")

        assert result.success is False
        assert "Cannot parse command" in result.error

    def test_allowlist_description_says_no_shell(self):
        assert "without a shell" in ShellTool(allowed_commands=["ls"]).description
        assert "without a shell" not in ShellTool().description

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
