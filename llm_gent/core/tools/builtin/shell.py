# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shell command execution tool."""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

from ..base import BaseTool, ToolResult


_ALLOWLIST_DESCRIPTION = (
    "Run a command and return its output. "
    "The command runs without a shell, in the configured working directory: "
    "no pipes, redirects, globs, variables or command chaining; quote "
    "arguments that contain spaces. Only allowed commands run."
)


class ShellTool(BaseTool):
    """Tool for executing shell commands.

    Allows the agent to run shell commands and receive their output.
    Useful for git operations, file system exploration, running tests, etc.

    Without ``allowed_commands`` the command runs in a shell: the agent can
    do anything the shell can. With it, the command is split into
    arguments (:func:`shlex.split`) and its program runs without a shell,
    so pipes, redirects, globs, variables and chaining are plain arguments;
    the program must be in the list. Each listed program is fully trusted:
    the agent can do whatever it can (``find -exec`` and git aliases run
    other programs). ``working_dir`` is where commands start, not a
    boundary; confining the agent to it is the deployment's job.

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
            allowed_commands: If set, only these programs run, without a shell.
                              E.g., ["ls", "grep"] allows ls and grep.
        """
        self._working_dir = working_dir
        self._timeout = timeout
        self._max_output_chars = max_output_chars
        self._allowed_commands = allowed_commands
        if allowed_commands:
            self.description = _ALLOWLIST_DESCRIPTION

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

        if not self._allowed_commands:
            return self._run_command(command)
        argv = self._allowed_argv(self._allowed_commands, command)
        if isinstance(argv, ToolResult):
            return argv
        return self._run_command(argv)

    def _allowed_argv(self, allowed: list[str], command: str) -> list[str] | ToolResult:
        """The command's arguments when its program is allowed, else the error."""
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return ToolResult(success=False, output="", error=f"Cannot parse command: {e}")
        if not argv:
            return ToolResult(success=False, output="", error="Empty command")
        if argv[0] not in allowed:
            return ToolResult(
                success=False,
                output="",
                error=f"Command '{argv[0]}' not in allowed list: {allowed}",
            )
        return argv

    def _run_command(self, command: str | list[str]) -> ToolResult:
        """Run ``command``: a string in a shell, an argument list without one."""
        try:
            result = subprocess.run(
                command,
                shell=isinstance(command, str),
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
