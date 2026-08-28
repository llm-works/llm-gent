# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""File operations tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..base import BaseTool, ToolResult


class _FileToolBase(BaseTool):
    """Base class for file tools with shared path handling logic."""

    # Subclasses must define these
    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(
        self,
        working_dir: str | None = None,
        allowed_paths: list[str] | None = None,
    ) -> None:
        """Initialize file tool base.

        Args:
            working_dir: Base directory for relative paths. Defaults to current dir.
            allowed_paths: If set, only files under these paths can be accessed.
                          Paths are resolved to absolute paths for comparison.
                          Symlinks are resolved before checking, so a symlink inside
                          an allowed directory pointing outside will be blocked.
        """
        self._working_dir = Path(working_dir) if working_dir else Path.cwd()
        self._allowed_paths = [Path(p).resolve() for p in allowed_paths] if allowed_paths else None

    def _resolve_path(self, path_arg: str) -> Path:
        """Resolve path argument to absolute path."""
        path = Path(path_arg)
        if not path.is_absolute():
            path = self._working_dir / path
        return path.resolve()

    def _check_allowed(self, path: Path) -> ToolResult | None:
        """Check if path is within allowed paths. Returns error if not allowed."""
        if not self._allowed_paths:
            return None

        for allowed in self._allowed_paths:
            try:
                path.relative_to(allowed)
                return None  # Path is under an allowed path
            except ValueError:
                continue

        return ToolResult(
            success=False,
            output="",
            error=f"Path '{path}' is not under allowed paths: {self._allowed_paths}",
        )


class FileReadTool(_FileToolBase):
    """Tool for reading file contents.

    Allows the agent to read files from the filesystem. Supports reading
    entire files or specific line ranges for large files.

    Security Note:
        This tool reads files from the filesystem. Consider restricting
        access to specific directories in production environments using
        the allowed_paths parameter.

    Example:
        tool = FileReadTool(allowed_paths=["/path/to/repo"])
        result = tool.execute(path="/path/to/repo/src/main.py")
        print(result.output)
    """

    name = "read_file"
    description = (
        "Read the contents of a file. "
        "Use for: examining source code, reading configuration files, checking file contents. "
        "Returns the file content with line numbers. "
        "For large files, use offset and limit to read specific line ranges."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (absolute or relative to working directory)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed, default: 1)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default: read all)",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        working_dir: str | None = None,
        max_chars: int = 100000,
        max_lines: int = 2000,
        allowed_paths: list[str] | None = None,
    ) -> None:
        """Initialize file read tool.

        Args:
            working_dir: Base directory for relative paths. Defaults to current dir.
            max_chars: Maximum characters to return. Defaults to 100000.
            max_lines: Maximum lines to return when no limit specified. Defaults to 2000.
            allowed_paths: If set, only files under these paths can be read.
                          Paths are resolved to absolute paths for comparison.
                          Symlinks are resolved before checking, so a symlink inside
                          an allowed directory pointing outside will be blocked.
        """
        super().__init__(working_dir=working_dir, allowed_paths=allowed_paths)
        self._max_chars = max_chars
        self._max_lines = max_lines

    def execute(self, **kwargs: Any) -> ToolResult:
        """Read a file's contents.

        Args:
            **kwargs: Must contain 'path'. Optional: 'offset', 'limit'.

        Returns:
            ToolResult with file contents (with line numbers) or error.
        """
        path_arg = kwargs.get("path")
        if not isinstance(path_arg, str) or not path_arg:
            return ToolResult(success=False, output="", error="Missing or invalid 'path' argument")

        offset = kwargs.get("offset", 1)
        if not isinstance(offset, int) or offset < 1:
            offset = 1

        limit = kwargs.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            limit = None

        # Resolve path
        path = self._resolve_path(path_arg)

        # Security check
        if error := self._check_allowed(path):
            return error

        # Read file
        return self._read_file(path, offset, limit)

    def _read_file(self, path: Path, offset: int, limit: int | None) -> ToolResult:
        """Read file contents with line numbers."""
        if error := self._validate_path(path):
            return error

        content_or_error = self._read_content(path)
        if isinstance(content_or_error, ToolResult):
            return content_or_error

        return self._select_and_format_lines(content_or_error, offset, limit)

    def _validate_path(self, path: Path) -> ToolResult | None:
        """Validate path exists and is a file. Returns error or None."""
        if not path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        if not path.is_file():
            return ToolResult(success=False, output="", error=f"Not a file: {path}")
        return None

    def _read_content(self, path: Path) -> str | ToolResult:
        """Read file content. Returns content string or error ToolResult."""
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {path}")
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot read file as UTF-8 (binary file?): {path}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read file: {e}")

    def _select_and_format_lines(self, content: str, offset: int, limit: int | None) -> ToolResult:
        """Select lines by offset/limit and format with line numbers."""
        lines = content.splitlines()
        total_lines = len(lines)

        # Handle empty file
        if total_lines == 0:
            return ToolResult(success=True, output="(Empty file)")

        # Apply offset (1-indexed)
        start_idx = offset - 1
        if start_idx >= total_lines:
            return ToolResult(
                success=True,
                output=f"(File has {total_lines} lines, offset {offset} is past end)",
            )

        # Apply limit and select lines
        effective_limit = limit if limit else self._max_lines
        end_idx = min(start_idx + effective_limit, total_lines)
        selected_lines = lines[start_idx:end_idx]

        # Format output
        output = self._format_with_line_numbers(selected_lines, offset)
        if end_idx < total_lines and limit is None:
            output += f"\n... ({total_lines - end_idx} more lines, use offset/limit to read more)"

        return ToolResult(success=True, output=self._truncate_output(output))

    def _format_with_line_numbers(self, lines: list[str], start_line: int) -> str:
        """Format lines with line numbers."""
        # Calculate width for line numbers
        max_line_num = start_line + len(lines) - 1
        width = len(str(max_line_num))

        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            numbered_lines.append(f"{line_num:>{width}}│ {line}")

        return "\n".join(numbered_lines)

    def _truncate_output(self, output: str) -> str:
        """Truncate output if too long."""
        if len(output) <= self._max_chars:
            return output
        return output[: self._max_chars] + f"\n... (truncated, {len(output)} total chars)"


class FileWriteTool(_FileToolBase):
    """Tool for writing file contents.

    Allows the agent to write files to the filesystem. Supports both
    overwriting existing files and appending to them.

    Security Note:
        This tool writes files to the filesystem. Consider restricting
        access to specific directories in production environments using
        the allowed_paths parameter.

    Example:
        tool = FileWriteTool(allowed_paths=["/path/to/repo"])
        result = tool.execute(
            path="/path/to/repo/output.txt",
            content="Hello, world!",
        )
        print(result.output)
    """

    name = "write_file"
    description = (
        "Write content to a file. "
        "Use for: creating new files, updating existing files, saving output. "
        "By default overwrites the file. Use mode='append' to append instead. "
        "Parent directories are created automatically if create_dirs is true."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (absolute or relative to working directory)",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "Write mode: 'overwrite' replaces file contents, 'append' adds to end (default: overwrite)",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Create parent directories if they don't exist (default: true)",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(
        self,
        working_dir: str | None = None,
        allowed_paths: list[str] | None = None,
        max_write_size: int = 1000000,
    ) -> None:
        """Initialize file write tool.

        Args:
            working_dir: Base directory for relative paths. Defaults to current dir.
            allowed_paths: If set, only files under these paths can be written.
                          Paths are resolved to absolute paths for comparison.
                          Symlinks are resolved before checking, so a symlink inside
                          an allowed directory pointing outside will be blocked.
            max_write_size: Maximum bytes to write in a single operation. Defaults to 1MB.
        """
        super().__init__(working_dir=working_dir, allowed_paths=allowed_paths)
        self._max_write_size = max_write_size

    def execute(self, **kwargs: Any) -> ToolResult:
        """Write content to a file.

        Args:
            **kwargs: Must contain 'path' and 'content'.
                      Optional: 'mode' (overwrite/append), 'create_dirs' (bool).

        Returns:
            ToolResult with success message or error.
        """
        # Parse and validate arguments
        args_or_error = self._parse_args(kwargs)
        if isinstance(args_or_error, ToolResult):
            return args_or_error
        path_arg, content, mode, create_dirs = args_or_error

        # Check content size
        if len(content.encode("utf-8")) > self._max_write_size:
            return ToolResult(
                success=False,
                output="",
                error=f"Content exceeds maximum write size of {self._max_write_size} bytes",
            )

        # Resolve path and check security
        path = self._resolve_path(path_arg)
        if error := self._check_allowed(path):
            return error

        return self._write_file(path, content, mode, create_dirs)

    def _parse_args(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, str, Literal["overwrite", "append"], bool] | ToolResult:
        """Parse and validate execute() arguments. Returns tuple or error."""
        path_arg = kwargs.get("path")
        if not isinstance(path_arg, str) or not path_arg:
            return ToolResult(success=False, output="", error="Missing or invalid 'path' argument")

        content = kwargs.get("content")
        if not isinstance(content, str):
            return ToolResult(
                success=False, output="", error="Missing or invalid 'content' argument"
            )

        mode_arg = kwargs.get("mode", "overwrite")
        if mode_arg not in ("overwrite", "append"):
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid mode '{mode_arg}'. Must be 'overwrite' or 'append'.",
            )
        mode: Literal["overwrite", "append"] = mode_arg

        create_dirs = kwargs.get("create_dirs", True)
        if not isinstance(create_dirs, bool):
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid create_dirs '{create_dirs}'. Must be a boolean.",
            )

        return path_arg, content, mode, create_dirs

    def _write_file(
        self,
        path: Path,
        content: str,
        mode: Literal["overwrite", "append"],
        create_dirs: bool,
    ) -> ToolResult:
        """Write content to file."""
        # Validate path is not a directory
        try:
            if path.exists() and not path.is_file():
                return ToolResult(
                    success=False, output="", error=f"Path exists but is not a file: {path}"
                )
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {path}")

        # Create parent directories if needed
        if create_dirs:
            if error := self._ensure_parent_dirs(path):
                return error
        elif not path.parent.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Parent directory does not exist: {path.parent}",
            )

        # Write the file
        return self._do_write(path, content, mode)

    def _ensure_parent_dirs(self, path: Path) -> ToolResult | None:
        """Create parent directories if they don't exist. Returns error or None."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return None
        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied creating directories: {path.parent}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create directories: {e}",
            )

    def _do_write(
        self,
        path: Path,
        content: str,
        mode: Literal["overwrite", "append"],
    ) -> ToolResult:
        """Perform the actual file write."""
        try:
            file_existed = path.exists()
            if mode == "append":
                with path.open("a", encoding="utf-8") as f:
                    f.write(content)
                action = "Appended to"
            else:
                path.write_text(content, encoding="utf-8")
                action = "Updated" if file_existed else "Created"

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            byte_count = len(content.encode("utf-8"))
            return ToolResult(
                success=True,
                output=f"{action} {path} ({line_count} lines, {byte_count} bytes)",
            )
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to write file: {e}")
