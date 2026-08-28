# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""LLM message and result types."""

from __future__ import annotations

from typing import Any, Literal

from llm_infer.client.types import AdapterInfo
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation.

    Supports standard chat messages and tool-related messages:
    - system/user/assistant: Standard chat messages
    - assistant with tool_calls: LLM requesting tool execution
    - tool: Result of a tool execution
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    # For assistant messages that include tool calls
    tool_calls: list[dict[str, Any]] | None = None

    # For tool result messages
    tool_call_id: str | None = None


class CompletionResult(BaseModel):
    """Result from LLM completion."""

    id: str = Field(description="Unique response ID for feedback tracking")
    content: str = Field(description="Generated text")
    model: str = Field(description="Model used")
    tokens_used: int = Field(description="Total tokens (prompt + completion)")
    latency_ms: int = Field(description="Response time in milliseconds")

    # Tool calls requested by the LLM (if any)
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls requested by LLM"
    )

    # Parsed structured output (when output_schema is provided)
    parsed: Any | None = Field(
        default=None, description="Validated object when output_schema is used"
    )

    # LoRA adapter info (None if not requested)
    adapter: AdapterInfo | None = Field(
        default=None, description="LoRA adapter info with requested/actual/fallback status"
    )
