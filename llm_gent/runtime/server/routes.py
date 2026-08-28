# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Route handlers for learning agent API.

Creates an APIRouter with IPC-based handlers for protocol v1 endpoints.
"""

from __future__ import annotations

from typing import cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path, Request

from .protocol.v1 import (
    CompleteRequest,
    CompleteResponse,
    FeedbackRequest,
    FeedbackResponse,
    ForgetRequest,
    ForgetResponse,
    HealthRequest,
    HealthResponse,
    RecallRequest,
    RecallResponse,
    RememberRequest,
    RememberResponse,
)


async def _health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    req = HealthRequest(id=str(uuid4()))
    response = await request.app.state.ipc_channel.submit(req.id, req)
    return cast(HealthResponse, response)


async def _complete(body: CompleteRequest, request: Request) -> CompleteResponse:
    """Generate a completion from the agent."""
    response = await request.app.state.ipc_channel.submit(body.id, body)
    resp = cast(CompleteResponse, response)
    if not resp.success:
        raise HTTPException(status_code=500, detail=resp.error)
    return resp


async def _remember(body: RememberRequest, request: Request) -> RememberResponse:
    """Store a fact in agent memory."""
    response = await request.app.state.ipc_channel.submit(body.id, body)
    resp = cast(RememberResponse, response)
    if not resp.success:
        raise HTTPException(status_code=500, detail=resp.error)
    return resp


async def _forget(
    request: Request,
    fact_id: int = Path(..., description="ID of the fact to delete"),
) -> ForgetResponse:
    """Delete a fact from agent memory."""
    req = ForgetRequest(id=str(uuid4()), fact_id=fact_id)
    response = await request.app.state.ipc_channel.submit(req.id, req)
    resp = cast(ForgetResponse, response)
    if not resp.success:
        if resp.error and "not found" in resp.error.lower():
            raise HTTPException(status_code=404, detail=resp.error)
        raise HTTPException(status_code=500, detail=resp.error)
    return resp


async def _recall(body: RecallRequest, request: Request) -> RecallResponse:
    """Search facts by semantic similarity."""
    response = await request.app.state.ipc_channel.submit(body.id, body)
    resp = cast(RecallResponse, response)
    if not resp.success:
        if resp.error and "embedder" in resp.error.lower():
            raise HTTPException(status_code=400, detail=resp.error)
        raise HTTPException(status_code=500, detail=resp.error)
    return resp


async def _feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Record feedback on a response."""
    response = await request.app.state.ipc_channel.submit(body.id, body)
    resp = cast(FeedbackResponse, response)
    if not resp.success:
        if resp.error and "unknown response_id" in resp.error.lower():
            raise HTTPException(status_code=404, detail=resp.error)
        raise HTTPException(status_code=400, detail=resp.error)
    return resp


def create_agent_routes() -> APIRouter:
    """Create router with learning agent v1 endpoints."""
    router = APIRouter(prefix="/v1", tags=["Agent API v1"])

    router.add_api_route("/health", _health, methods=["GET"], response_model=HealthResponse)
    router.add_api_route("/complete", _complete, methods=["POST"], response_model=CompleteResponse)
    router.add_api_route("/remember", _remember, methods=["POST"], response_model=RememberResponse)
    router.add_api_route(
        "/facts/{fact_id}", _forget, methods=["DELETE"], response_model=ForgetResponse
    )
    router.add_api_route("/recall", _recall, methods=["POST"], response_model=RecallResponse)
    router.add_api_route("/feedback", _feedback, methods=["POST"], response_model=FeedbackResponse)

    return router
