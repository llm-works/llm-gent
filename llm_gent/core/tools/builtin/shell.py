# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shell command execution tool."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from ..base import BaseTool, ToolResult


# Shell metacharacters that enable command chaining/injection
_SHELL_METACHAR_PATTERN = re.compile(
    r"""
    ;           |  # command separator
    &&          |  # AND operator
    \|\|        |  # OR operator
    \|          |  # pipe
    \$[\(\{]    |  # command/variable substitution $() or ${}
    \$[A-Za-z_] |  # variable reference $VAR
    `           |  # backtick command substitution
    \n          |  # newline (command separator)
    &\s*$       |  # background execution at end
    >           |  # output redirection (includes >>)
    <              # input redirection (includes << heredoc)
    """,
    re.VERBOSE,
)


class ShellTool(BaseTool):
    """Tool for executing shell commands.

    Allows the agent to run shell commands and receive their output.
    Useful for git operations, file system exploration, running tests, etc.

    Security Note:
        This tool executes arbitrary shell commands. Only use in trusted
        environments where the LLM's actions are acceptable. Consider
        sandboxing or command allowlists for production use.

    Example:
        tool = ShellTool(working_dir="/path/to/repo")
        result = tool.execute(command="git status")
        print(result.output)
    """

    name = "shell"
    description = (
        "Execute a shell command and return its output. "
        "Use for: git operations, file listing, grep searches, running scripts. "
        "Commands run in a bash shell with the configured working directory."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        working_dir: str | None = None,
        timeout: float = 30.0,
        max_output_chars: int = 50000,
        allowed_commands: list[str] | None = None,
    ) -> None:
        """Initialize shell tool.

        Args:
            working_dir: Working directory for commands. Defaults to current dir.
            timeout: Command timeout in seconds. Defaults to 30.
            max_output_chars: Maximum output characters to return. Defaults to 50000.
            allowed_commands: If set, only these command prefixes are allowed.
                              E.g., ["git", "ls", "grep"] allows git/ls/grep commands.
        """
        self._working_dir = working_dir
        self._timeout = timeout
        self._max_output_chars = max_output_chars
        self._allowed_commands = allowed_commands

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a shell command.

        Args:
            **kwargs: Must contain 'command' key with the shell command to execute.

        Returns:
            ToolResult with command output or error.
        """
        command = kwargs.get("command")
        if not isinstance(command, str) or not command:
            return ToolResult(
                success=False, output="", error="Missing or invalid 'command' argument"
            )

        if error := self._check_allowed(command):
            return error

        return self._run_command(command)

    def _check_allowed(self, command: str) -> ToolResult | None:
        """Check if command is in allowed list. Returns error result if not allowed."""
        if not self._allowed_commands:
            return None

        # Reject commands with shell metacharacters that could bypass allowlist
        if _SHELL_METACHAR_PATTERN.search(command):
            return ToolResult(
                success=False,
                output="",
                error="Command contains shell metacharacters (;, &&, ||, |, $(), `) "
                "which are not allowed when command allowlist is enabled",
            )

        parts = command.split()
        if not parts:
            return ToolResult(success=False, output="", error="Empty command")
        cmd_prefix = parts[0]
        if cmd_prefix not in self._allowed_commands:
            return ToolResult(
                success=False,
                output="",
                error=f"Command '{cmd_prefix}' not in allowed list: {self._allowed_commands}",
            )
        return None

    def _run_command(self, command: str) -> ToolResult:
        """Run the shell command and return result."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self._working_dir,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return self._build_result(result)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {self._timeout} seconds",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to execute command: {e}")

    def _build_result(self, result: subprocess.CompletedProcess[str]) -> ToolResult:
        """Build ToolResult from subprocess result."""
        output = self._combine_output(result.stdout, result.stderr)
        output = self._truncate_output(output)

        if result.returncode == 0:
            return ToolResult(success=True, output=output)
        return ToolResult(
            success=False,
            output=output,
            error=f"Command exited with code {result.returncode}",
        )

    def _combine_output(self, stdout: str, stderr: str) -> str:
        """Combine stdout and stderr into single output string."""
        if not stderr:
            return stdout
        if not stdout:
            return stderr
        return f"{stdout}\n--- stderr ---\n{stderr}"

    def _truncate_output(self, output: str) -> str:
        """Truncate output if too long."""
        if len(output) <= self._max_output_chars:
            return output
        return output[: self._max_output_chars] + f"\n... (truncated, {len(output)} total chars)"
