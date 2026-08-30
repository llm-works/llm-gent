# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Task completion tool for signaling task status."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class CompleteTaskTool(BaseTool):
    """Tool for signaling task completion.

    When the agent believes it has finished working on a task (either
    successfully or because it's stuck), it calls this tool to signal
    completion with a structured status and conclusion.

    This is a terminal tool - calling it ends the tool execution loop
    immediately and captures the completion data.

    Example:
        # Agent reached a conclusion
        complete_task(status="done", conclusion="The answer is 42.")

        # Agent is stuck and needs help
        complete_task(status="stuck", conclusion="Need access to the database.")
    """

    name = "complete_task"
    terminal = True  # Calling this tool ends the execution loop
    description = (
        "Signal that you have finished working on the current task. "
        "Call this when you have reached a conclusion OR when you cannot make further progress. "
        "status='done' means you reached a conclusion (success, failure, or 'impossible'). "
        "status='stuck' means you need external help to continue. "
        "The conclusion should explain what you determined or why you're stuck."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["done", "stuck"],
                "description": (
                    "'done' = reached a conclusion (success, impossible, etc.), "
                    "'stuck' = cannot proceed without external help"
                ),
            },
            "conclusion": {
                "type": "string",
                "description": (
                    "What you determined (if done) or why you're stuck. "
                    "Be specific and include your reasoning."
                ),
            },
        },
        "required": ["status", "conclusion"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute task completion.

        Returns a terminal ToolResult that signals the executor to stop
        and captures the completion data.
        """
        status = kwargs.get("status", "")
        conclusion = kwargs.get("conclusion", "").strip()

        if status not in ("done", "stuck"):
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid status '{status}'. Must be 'done' or 'stuck'.",
            )

        if not conclusion:
            return ToolResult(
                success=False,
                output="",
                error="Conclusion is required.",
            )

        return ToolResult(
            success=True,
            output=f"Task marked as {status}.",
            terminal=True,
            terminal_data={
                "status": status,
                "conclusion": conclusion,
            },
        )
