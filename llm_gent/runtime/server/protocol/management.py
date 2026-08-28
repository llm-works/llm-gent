# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Protocol messages for agent management API.

These messages enable IPC between the FastAPI subprocess and main process
where Core runs.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from .base import Request, Response


# =============================================================================
# Health
# =============================================================================


class MgmtHealthRequest(Request):
    """Management health check request."""

    message_type: ClassVar[str] = "mgmt_health_request"


class MgmtHealthResponse(Response):
    """Management health check response."""

    message_type: ClassVar[str] = "mgmt_health_response"

    status: str = Field(description="Server status")
    agent_count: int = Field(description="Number of registered agents")


# =============================================================================
# List Agents
# =============================================================================


class ListAgentsRequest(Request):
    """List all agents request."""

    message_type: ClassVar[str] = "list_agents_request"


class AgentInfoData(Response):
    """Agent information data (used in responses)."""

    message_type: ClassVar[str] = "agent_info_data"

    name: str
    status: str
    cycle_count: int
    last_run: str | None
    error: str | None
    schedule_interval: int | None


class ListAgentsResponse(Response):
    """List all agents response."""

    message_type: ClassVar[str] = "list_agents_response"

    agents: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Get Agent
# =============================================================================


class GetAgentRequest(Request):
    """Get agent information request."""

    message_type: ClassVar[str] = "get_agent_request"

    agent_name: str = Field(description="Name of the agent")


class GetAgentResponse(Response):
    """Get agent information response."""

    message_type: ClassVar[str] = "get_agent_response"

    name: str = ""
    status: str = ""
    cycle_count: int = 0
    last_run: str | None = None
    error: str | None = None
    schedule_interval: int | None = None


# =============================================================================
# Start Agent
# =============================================================================


class StartAgentRequest(Request):
    """Start agent request."""

    message_type: ClassVar[str] = "start_agent_request"

    agent_name: str = Field(description="Name of the agent to start")


class StartAgentResponse(Response):
    """Start agent response."""

    message_type: ClassVar[str] = "start_agent_response"

    name: str = ""
    status: str = ""
    cycle_count: int = 0
    last_run: str | None = None
    error: str | None = None
    schedule_interval: int | None = None


# =============================================================================
# Stop Agent
# =============================================================================


class StopAgentRequest(Request):
    """Stop agent request."""

    message_type: ClassVar[str] = "stop_agent_request"

    agent_name: str = Field(description="Name of the agent to stop")


class StopAgentResponse(Response):
    """Stop agent response."""

    message_type: ClassVar[str] = "stop_agent_response"

    name: str = ""
    status: str = ""
    cycle_count: int = 0
    last_run: str | None = None
    error: str | None = None
    schedule_interval: int | None = None


# =============================================================================
# Ask Agent
# =============================================================================


class AskAgentRequest(Request):
    """Ask agent a question request."""

    message_type: ClassVar[str] = "ask_agent_request"

    agent_name: str = Field(description="Name of the agent")
    question: str = Field(description="Question to ask")


class AskAgentResponse(Response):
    """Ask agent response."""

    message_type: ClassVar[str] = "ask_agent_response"

    response: str = ""


# =============================================================================
# Feedback
# =============================================================================


class FeedbackAgentRequest(Request):
    """Send feedback to agent request."""

    message_type: ClassVar[str] = "feedback_agent_request"

    agent_name: str = Field(description="Name of the agent")
    message: str = Field(description="Feedback message")


class FeedbackAgentResponse(Response):
    """Feedback response."""

    message_type: ClassVar[str] = "feedback_agent_response"


# =============================================================================
# Get Insights
# =============================================================================


class GetInsightsRequest(Request):
    """Get agent insights request."""

    message_type: ClassVar[str] = "get_insights_request"

    agent_name: str = Field(description="Name of the agent")
    limit: int = Field(default=10, description="Maximum number of insights")


class InsightData(Response):
    """Single insight data."""

    message_type: ClassVar[str] = "insight_data"

    success: bool = True
    content: str = ""
    parsed: dict[str, Any] | None = None
    iterations: int = 0


class GetInsightsResponse(Response):
    """Get insights response."""

    message_type: ClassVar[str] = "get_insights_response"

    insights: list[dict[str, Any]] = Field(default_factory=list)
