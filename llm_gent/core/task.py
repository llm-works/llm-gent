# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Task execution models for agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .tools.base import ToolCallResult


class TaskStatus(StrEnum):
    """Status of a completed task.

    The agent uses this to signal whether it reached a conclusion
    or needs external help to continue.
    """

    DONE = "done"
    """Agent reached a conclusion (success, impossible, or any definitive answer)."""

    STUCK = "stuck"
    """Agent cannot make progress and needs external help."""


class TaskCompletion(BaseModel):
    """Structured completion data from agent.

    Captured when agent calls complete_task() to signal it's done
    working on the current task.
    """

    status: TaskStatus
    """Whether agent reached a conclusion or is stuck."""

    conclusion: str
    """What was determined, or why stuck."""


class Task(BaseModel):
    """A task for an agent to execute.

    Tasks define what an agent should accomplish, optionally with structured
    output and iteration limits.

    Example:
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str
            confidence: float

        task = Task(
            name="analyze",
            description="Analyze the sentiment of the text",
            context={"text": "I love this product!"},
            output_schema=Answer,
        )
    """

    name: str
    """Task identifier."""

    description: str
    """What the agent should do."""

    context: dict[str, Any] = Field(default_factory=dict)
    """Additional context injected into the prompt."""

    output_schema: type[BaseModel] | None = None
    """Pydantic model for structured output validation."""

    system_prompt: str | None = None
    """Override agent's default system prompt."""

    max_iterations: int = 10
    """Maximum LLM round-trips before stopping. 0 means no limit."""

    timeout_secs: float = 0
    """Maximum seconds for task execution. 0 means no limit."""

    model_config = {"arbitrary_types_allowed": True}


class TaskResult(BaseModel):
    """Result of task execution.

    Contains the final response, any parsed structured output, and
    execution metadata.
    """

    success: bool
    """Whether the task completed successfully."""

    content: str
    """Final LLM response text."""

    parsed: Any | None = None
    """Validated object when output_schema was provided."""

    completion: TaskCompletion | None = None
    """Structured completion when agent called complete_task()."""

    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    """All tool calls made during execution."""

    iterations: int = 1
    """Number of LLM round-trips."""

    tokens_used: int = 0
    """Total tokens consumed."""

    error: str | None = None
    """Error message if success=False."""

    model_config = {"arbitrary_types_allowed": True}
