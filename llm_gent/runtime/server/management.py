# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent management routes.

REST API for managing agent lifecycle via Core.
Routes use IPC to communicate with main process where Core runs.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .protocol.management import (
    AskAgentRequest,
    AskAgentResponse,
    FeedbackAgentRequest,
    FeedbackAgentResponse,
    GetAgentRequest,
    GetAgentResponse,
    GetInsightsRequest,
    GetInsightsResponse,
    ListAgentsRequest,
    ListAgentsResponse,
    MgmtHealthRequest,
    MgmtHealthResponse,
    StartAgentRequest,
    StartAgentResponse,
    StopAgentRequest,
    StopAgentResponse,
)


# ============================================================================
# Response Models (for OpenAPI docs)
# ============================================================================


class AgentInfoResponse(BaseModel):
    """Agent information response."""

    name: str
    status: str
    cycle_count: int
    last_run: str | None
    error: str | None
    schedule_interval: int | None


class AgentListResponse(BaseModel):
    """List of agents response."""

    agents: list[AgentInfoResponse]


class AskRequest(BaseModel):
    """Ask question request."""

    question: str


class AskResponse(BaseModel):
    """Ask question response."""

    response: str


class FeedbackRequest(BaseModel):
    """Feedback request."""

    message: str


class FeedbackResponse(BaseModel):
    """Feedback response."""

    success: bool


class InsightResponse(BaseModel):
    """Single insight from agent."""

    success: bool
    content: str
    parsed: dict[str, Any] | None
    iterations: int


class InsightsResponse(BaseModel):
    """List of insights response."""

    insights: list[InsightResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_count: int


# ============================================================================
# Helpers
# ============================================================================


def _check_response(resp: Any, agent_name: str | None = None) -> None:
    """Check IPC response and raise HTTP exceptions as needed."""
    if not resp.success:
        error = resp.error or "Unknown error"
        if agent_name and "not found" in error.lower():
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")
        # Internal IPC errors (from exception handling) are server errors
        if "internal server error" in error.lower():
            raise HTTPException(status_code=500, detail=error)
        # Business logic errors (e.g., "not running", "invalid transition") are client errors
        raise HTTPException(status_code=400, detail=error)


# ============================================================================
# Route Handlers
# ============================================================================


async def _health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    req = MgmtHealthRequest(id=str(uuid4()))
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(MgmtHealthResponse, resp)
    _check_response(resp)
    return HealthResponse(status=resp.status, agent_count=resp.agent_count)


async def _list_agents(request: Request) -> AgentListResponse:
    """List all registered agents."""
    req = ListAgentsRequest(id=str(uuid4()))
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(ListAgentsResponse, resp)
    _check_response(resp)
    return AgentListResponse(agents=[AgentInfoResponse(**a) for a in resp.agents])


async def _get_agent(name: str, request: Request) -> AgentInfoResponse:
    """Get agent information by name."""
    req = GetAgentRequest(id=str(uuid4()), agent_name=name)
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(GetAgentResponse, resp)
    _check_response(resp, name)
    return AgentInfoResponse(
        name=resp.name,
        status=resp.status,
        cycle_count=resp.cycle_count,
        last_run=resp.last_run,
        error=resp.error,
        schedule_interval=resp.schedule_interval,
    )


async def _start_agent(name: str, request: Request) -> AgentInfoResponse:
    """Start an agent process."""
    req = StartAgentRequest(id=str(uuid4()), agent_name=name)
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(StartAgentResponse, resp)
    _check_response(resp, name)
    return AgentInfoResponse(
        name=resp.name,
        status=resp.status,
        cycle_count=resp.cycle_count,
        last_run=resp.last_run,
        error=resp.error,
        schedule_interval=resp.schedule_interval,
    )


async def _stop_agent(name: str, request: Request) -> AgentInfoResponse:
    """Stop an agent process."""
    req = StopAgentRequest(id=str(uuid4()), agent_name=name)
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(StopAgentResponse, resp)
    _check_response(resp, name)
    return AgentInfoResponse(
        name=resp.name,
        status=resp.status,
        cycle_count=resp.cycle_count,
        last_run=resp.last_run,
        error=resp.error,
        schedule_interval=resp.schedule_interval,
    )


async def _ask_agent(name: str, body: AskRequest, request: Request) -> AskResponse:
    """Ask an agent a question."""
    req = AskAgentRequest(id=str(uuid4()), agent_name=name, question=body.question)
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(AskAgentResponse, resp)
    _check_response(resp, name)
    return AskResponse(response=resp.response)


async def _feedback_agent(name: str, body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Send feedback to an agent."""
    req = FeedbackAgentRequest(id=str(uuid4()), agent_name=name, message=body.message)
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(FeedbackAgentResponse, resp)
    _check_response(resp, name)
    return FeedbackResponse(success=True)


async def _get_insights(name: str, request: Request, limit: int = 10) -> InsightsResponse:
    """Get recent insights from an agent."""
    req = GetInsightsRequest(id=str(uuid4()), agent_name=name, limit=limit)
    resp = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(GetInsightsResponse, resp)
    _check_response(resp, name)
    return InsightsResponse(insights=[InsightResponse(**i) for i in resp.insights])


# ============================================================================
# Route Factory
# ============================================================================


def create_management_routes() -> APIRouter:
    """Create a fresh API router for agent management endpoints.

    Returns a new router instance each time, allowing safe mounting
    to multiple apps or test isolation.

    Routes use IPC to communicate with main process where Core runs.
    """
    router = APIRouter(tags=["Agent Management"])

    router.add_api_route("/health", _health, methods=["GET"], response_model=HealthResponse)
    router.add_api_route("/agents", _list_agents, methods=["GET"], response_model=AgentListResponse)
    router.add_api_route(
        "/agents/{name}", _get_agent, methods=["GET"], response_model=AgentInfoResponse
    )
    router.add_api_route(
        "/agents/{name}/start", _start_agent, methods=["POST"], response_model=AgentInfoResponse
    )
    router.add_api_route(
        "/agents/{name}/stop", _stop_agent, methods=["POST"], response_model=AgentInfoResponse
    )
    router.add_api_route(
        "/agents/{name}/ask", _ask_agent, methods=["POST"], response_model=AskResponse
    )
    router.add_api_route(
        "/agents/{name}/feedback",
        _feedback_agent,
        methods=["POST"],
        response_model=FeedbackResponse,
    )
    router.add_api_route(
        "/agents/{name}/insights", _get_insights, methods=["GET"], response_model=InsightsResponse
    )

    return router
