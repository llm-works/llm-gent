# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Protocol v1 message definitions for learning agent API."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from ..base import Request, Response


# =============================================================================
# Health
# =============================================================================


class HealthRequest(Request):
    """Health check request."""

    message_type: ClassVar[str] = "health_request"


class HealthResponse(Response):
    """Health check response."""

    message_type: ClassVar[str] = "health_response"

    status: str = Field(description="Server status")
    agent_name: str = Field(description="Name of the agent")


# =============================================================================
# Complete
# =============================================================================


class CompleteRequest(Request):
    """LLM completion request."""

    message_type: ClassVar[str] = "complete_request"

    query: str = Field(description="User query to complete")
    system_prompt: str | None = Field(default=None, description="Optional system prompt override")


class CompleteResponse(Response):
    """LLM completion response."""

    message_type: ClassVar[str] = "complete_response"

    response_id: str = Field(description="Response ID for feedback")
    content: str = Field(description="LLM response content")
    model: str = Field(description="Model used for completion")
    tokens_used: int = Field(description="Total tokens used")


# =============================================================================
# Remember
# =============================================================================


class RememberRequest(Request):
    """Store a fact request."""

    message_type: ClassVar[str] = "remember_request"

    fact: str = Field(description="Fact to store")
    category: str = Field(default="general", description="Category for organization")


class RememberResponse(Response):
    """Store a fact response."""

    message_type: ClassVar[str] = "remember_response"

    fact_id: int = Field(description="ID of stored fact")


# =============================================================================
# Forget
# =============================================================================


class ForgetRequest(Request):
    """Delete a fact request."""

    message_type: ClassVar[str] = "forget_request"

    fact_id: int = Field(description="ID of the fact to delete")


class ForgetResponse(Response):
    """Delete a fact response."""

    message_type: ClassVar[str] = "forget_response"


# =============================================================================
# Recall
# =============================================================================


class RecallRequest(Request):
    """Semantic search request."""

    message_type: ClassVar[str] = "recall_request"

    query: str = Field(description="Query for semantic search")
    top_k: int | None = Field(default=None, description="Max results to return")
    min_similarity: float | None = Field(default=None, description="Minimum similarity threshold")
    categories: list[str] | None = Field(default=None, description="Filter to these categories")


class RecallResponse(Response):
    """Semantic search response."""

    message_type: ClassVar[str] = "recall_response"

    facts: list[dict[str, Any]] = Field(default_factory=list, description="Matching facts")


# =============================================================================
# Feedback
# =============================================================================


class FeedbackRequest(Request):
    """Feedback on a response request."""

    message_type: ClassVar[str] = "feedback_request"

    response_id: str = Field(description="ID from CompleteResponse")
    signal: Literal["positive", "negative"] = Field(description="Feedback signal")
    correction: str | None = Field(default=None, description="Corrected response if negative")


class FeedbackResponse(Response):
    """Feedback response."""

    message_type: ClassVar[str] = "feedback_response"


# =============================================================================
# Message Registry
# =============================================================================

MESSAGE_REGISTRY: dict[str, type[Request | Response]] = {
    HealthRequest.message_type: HealthRequest,
    HealthResponse.message_type: HealthResponse,
    CompleteRequest.message_type: CompleteRequest,
    CompleteResponse.message_type: CompleteResponse,
    RememberRequest.message_type: RememberRequest,
    RememberResponse.message_type: RememberResponse,
    ForgetRequest.message_type: ForgetRequest,
    ForgetResponse.message_type: ForgetResponse,
    RecallRequest.message_type: RecallRequest,
    RecallResponse.message_type: RecallResponse,
    FeedbackRequest.message_type: FeedbackRequest,
    FeedbackResponse.message_type: FeedbackResponse,
}
