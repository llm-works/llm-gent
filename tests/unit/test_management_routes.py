# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for management routes.

Requests go through httpx's ASGI transport on the test's own event loop: every handler is
``async def``, so no worker thread runs.
"""

from collections.abc import AsyncIterator
from unittest.mock import DEFAULT, AsyncMock, MagicMock

import httpx
import pytest
from appinfra.app.fastapi.runtime.ipc import IPCChannel
from fastapi import FastAPI

from llm_gent.bus.protocol import AskProgress, AskResponse
from llm_gent.core.agent import Identity, SessionConfig, Sessions
from llm_gent.core.errors import SessionNotFoundError
from llm_gent.runtime.server import create_app
from llm_gent.runtime.server.management import create_management_routes
from llm_gent.runtime.server.protocol.management import (
    FeedbackAgentResponse,
    GetAgentResponse,
    GetInsightsResponse,
    ListAgentsResponse,
    MgmtHealthResponse,
    StartAgentResponse,
    StopAgentResponse,
)


pytestmark = pytest.mark.unit


def _client(app: FastAPI) -> httpx.AsyncClient:
    """An HTTP client calling ``app`` in-process."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _session_rejection(session: str) -> str:
    """The error an agent raises for ``session``, as it travels back to the gateway."""
    sessions = Sessions(MagicMock(), Identity("test"), SessionConfig(), None)
    with pytest.raises(SessionNotFoundError) as rejected, sessions.use(session):
        pass
    return str(rejected.value)


class TestCreateApp:
    """Tests for create_app factory function."""

    async def test_creates_fastapi_app(self):
        """create_app returns a FastAPI application."""
        mock_core = MagicMock()
        mock_core.registry.list_agents.return_value = []

        app = create_app(mock_core)

        assert isinstance(app, FastAPI)
        assert app.title == "Agent Gateway"

    async def test_stores_core_in_state(self):
        """create_app stores core in app.state."""
        mock_core = MagicMock()

        app = create_app(mock_core)

        assert app.state.core is mock_core

    async def test_includes_management_routes(self):
        """create_app includes management routes."""
        mock_core = MagicMock()
        mock_core.registry.list_agents.return_value = []

        app = create_app(mock_core)
        # Add mock ipc_channel for IPC-based routes
        mock_channel = AsyncMock()
        mock_channel.submit.return_value = MgmtHealthResponse(id="1", status="ok", agent_count=0)
        app.state.ipc_channel = mock_channel

        async with _client(app) as client:
            response = await client.get("/health")

        assert response.status_code == 200

    async def test_custom_title(self):
        """create_app accepts custom title."""
        mock_core = MagicMock()

        app = create_app(mock_core, title="Custom Gateway")

        assert app.title == "Custom Gateway"


def _submit_contract(request, timeout=None):
    """``IPCChannel.submit``'s contract: the request first, carrying the id its reply gets."""
    if not hasattr(request, "id"):
        raise ValueError("Request must have an 'id' attribute")
    return DEFAULT


@pytest.fixture
def mock_ipc_channel():
    """An IPC channel that takes requests as appinfra's does and replies its return_value."""
    channel = AsyncMock(spec=IPCChannel)
    channel.submit.side_effect = _submit_contract
    return channel


@pytest.fixture
def app(mock_ipc_channel):
    """Create FastAPI app with routes and mock IPC channel."""
    app = FastAPI()
    app.state.ipc_channel = mock_ipc_channel
    app.include_router(create_management_routes())
    return app


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    """Create test client."""
    async with _client(app) as client:
        yield client


class TestHealthRoute:
    """Tests for /health endpoint."""

    async def test_health_returns_ok(self, client, mock_ipc_channel):
        """Health endpoint returns ok status."""
        mock_ipc_channel.submit.return_value = MgmtHealthResponse(
            id="1", status="ok", agent_count=0
        )

        response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "agent_count": 0}

    async def test_health_includes_agent_count(self, client, mock_ipc_channel):
        """Health endpoint includes agent count."""
        mock_ipc_channel.submit.return_value = MgmtHealthResponse(
            id="1", status="ok", agent_count=3
        )

        response = await client.get("/health")

        assert response.json()["agent_count"] == 3


class TestListAgentsRoute:
    """Tests for GET /agents endpoint."""

    async def test_list_empty(self, client, mock_ipc_channel):
        """List returns empty when no agents."""
        mock_ipc_channel.submit.return_value = ListAgentsResponse(id="1", agents=[])

        response = await client.get("/agents")

        assert response.status_code == 200
        assert response.json() == {"agents": []}

    async def test_list_returns_agents(self, client, mock_ipc_channel):
        """List returns all agents."""
        mock_ipc_channel.submit.return_value = ListAgentsResponse(
            id="1",
            agents=[
                {
                    "name": "agent1",
                    "status": "idle",
                    "cycle_count": 0,
                    "last_run": None,
                    "error": None,
                    "schedule_interval": None,
                },
                {
                    "name": "agent2",
                    "status": "running",
                    "cycle_count": 5,
                    "last_run": None,
                    "error": None,
                    "schedule_interval": None,
                },
            ],
        )

        response = await client.get("/agents")

        data = response.json()
        assert len(data["agents"]) == 2
        assert data["agents"][0]["name"] == "agent1"
        assert data["agents"][1]["name"] == "agent2"


class TestGetAgentRoute:
    """Tests for GET /agents/{name} endpoint."""

    async def test_get_existing_agent(self, client, mock_ipc_channel):
        """Get returns agent info."""
        mock_ipc_channel.submit.return_value = GetAgentResponse(
            id="1",
            name="test-agent",
            status="running",
            cycle_count=10,
            last_run=None,
            error=None,
            schedule_interval=None,
        )

        response = await client.get("/agents/test-agent")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-agent"
        assert data["status"] == "running"

    async def test_get_nonexistent_agent(self, client, mock_ipc_channel):
        """Get returns 404 for nonexistent agent."""
        mock_ipc_channel.submit.return_value = GetAgentResponse(
            id="1", success=False, error="Agent not found: nonexistent"
        )

        response = await client.get("/agents/nonexistent")

        assert response.status_code == 404


class TestStartAgentRoute:
    """Tests for POST /agents/{name}/start endpoint."""

    async def test_start_agent(self, client, mock_ipc_channel):
        """Start transitions agent to running."""
        mock_ipc_channel.submit.return_value = StartAgentResponse(
            id="1",
            name="test",
            status="running",
            cycle_count=0,
            last_run=None,
            error=None,
            schedule_interval=None,
        )

        response = await client.post("/agents/test/start")

        assert response.status_code == 200
        assert response.json()["status"] == "running"

    async def test_start_nonexistent(self, client, mock_ipc_channel):
        """Start returns 404 for nonexistent agent."""
        mock_ipc_channel.submit.return_value = StartAgentResponse(
            id="1", success=False, error="Agent not found: nonexistent"
        )

        response = await client.post("/agents/nonexistent/start")

        assert response.status_code == 404

    async def test_start_invalid_state(self, client, mock_ipc_channel):
        """Start returns 400 for invalid state transition."""
        mock_ipc_channel.submit.return_value = StartAgentResponse(
            id="1", success=False, error="Invalid transition"
        )

        response = await client.post("/agents/test/start")

        assert response.status_code == 400


class TestStopAgentRoute:
    """Tests for POST /agents/{name}/stop endpoint."""

    async def test_stop_agent(self, client, mock_ipc_channel):
        """Stop transitions agent to stopped."""
        mock_ipc_channel.submit.return_value = StopAgentResponse(
            id="1",
            name="test",
            status="stopped",
            cycle_count=0,
            last_run=None,
            error=None,
            schedule_interval=None,
        )

        response = await client.post("/agents/test/stop")

        assert response.status_code == 200
        assert response.json()["status"] == "stopped"

    async def test_stop_nonexistent(self, client, mock_ipc_channel):
        """Stop returns 404 for nonexistent agent."""
        mock_ipc_channel.submit.return_value = StopAgentResponse(
            id="1", success=False, error="Agent not found: nonexistent"
        )

        response = await client.post("/agents/nonexistent/stop")

        assert response.status_code == 404


class TestAskAgentRoute:
    """Tests for POST /agents/{name}/ask endpoint."""

    async def test_ask_agent(self, client, mock_ipc_channel):
        """Ask returns agent response."""
        mock_ipc_channel.submit.return_value = AskResponse(id="1", response="Agent response")

        response = await client.post("/agents/test/ask", json={"question": "What time is it?"})

        assert response.status_code == 200
        assert response.json()["response"] == "Agent response"

    async def test_ask_carries_the_session(self, client, mock_ipc_channel):
        """The session goes to the agent, and the one the answer ran in comes back."""
        mock_ipc_channel.submit.return_value = AskResponse(
            id="1", response="Agent response", session="s-1"
        )

        response = await client.post(
            "/agents/test/ask", json={"question": "q", "session": "s-1", "kwargs": {"job": "j-1"}}
        )

        assert response.json()["session"] == "s-1"
        sent = mock_ipc_channel.submit.call_args[0][0]
        assert (sent.agent_name, sent.id) == ("test", sent.ask.id)
        assert (sent.ask.session, sent.ask.kwargs, sent.ask.stream) == (
            "s-1",
            {"job": "j-1"},
            False,
        )

    async def test_ask_of_an_agent_without_chat_is_a_client_error(self, client, mock_ipc_channel):
        mock_ipc_channel.submit.return_value = AskResponse(
            id="1", success=False, error="agent test has no chat"
        )

        response = await client.post("/agents/test/ask", json={"question": "q"})

        assert response.status_code == 400

    async def test_ask_in_an_unknown_session_is_a_client_error(self, client, mock_ipc_channel):
        """An unknown session is a 400, not a missing agent."""
        mock_ipc_channel.submit.return_value = AskResponse(
            id="1", success=False, error="no session s-x"
        )

        response = await client.post("/agents/test/ask", json={"question": "q", "session": "s-x"})

        assert response.status_code == 400

    @pytest.mark.parametrize("session", ["not found", "internal server error"])
    async def test_a_rejected_session_is_a_client_error_whatever_it_says(
        self, client, mock_ipc_channel, session
    ):
        """The session's own text does not pick the status: no 404, no 500."""
        rejection = _session_rejection(session)
        mock_ipc_channel.submit.return_value = AskResponse(id="1", success=False, error=rejection)

        response = await client.post("/agents/test/ask", json={"question": "q", "session": session})

        assert response.status_code == 400

    async def test_ask_nonexistent(self, client, mock_ipc_channel):
        """Ask returns 404 for nonexistent agent."""
        mock_ipc_channel.submit.return_value = AskResponse(
            id="1", success=False, error="Agent not found: nonexistent"
        )

        response = await client.post("/agents/nonexistent/ask", json={"question": "test"})

        assert response.status_code == 404

    async def test_ask_not_running(self, client, mock_ipc_channel):
        """Ask returns 400 for non-running agent."""
        mock_ipc_channel.submit.return_value = AskResponse(
            id="1", success=False, error="Agent not running"
        )

        response = await client.post("/agents/test/ask", json={"question": "test"})

        assert response.status_code == 400


def _stream(*chunks):
    """An IPC channel's ``stream`` replaying ``chunks`` under the request's id; its requests."""
    sent: list = []

    async def stream(request, timeout=None):
        sent.append(request)
        for chunk in chunks:
            yield chunk.model_copy(update={"id": request.id})

    return stream, sent


class TestAskStreamRoute:
    """Tests for POST /agents/{name}/ask/stream: the ask's steps as server-sent events."""

    async def test_the_steps_then_the_answer_stream_as_sse(self, client, mock_ipc_channel):
        mock_ipc_channel.stream, sent = _stream(
            AskProgress(id="", name="model_call", data={"iteration": 1}),
            AskResponse(id="", response="hello", session="s-1"),
        )

        response = await client.post("/agents/test/ask/stream", json={"question": "q"})

        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text == (
            'event: step\ndata: {"name": "model_call", "data": {"iteration": 1}}\n\n'
            'event: answer\ndata: {"response": "hello", "session": "s-1"}\n\n'
        )
        assert sent[0].ask.stream is True

    async def test_a_failure_before_any_step_is_an_http_error(self, client, mock_ipc_channel):
        mock_ipc_channel.stream, _ = _stream(
            AskResponse(id="", success=False, error="Agent not found: ghost")
        )

        response = await client.post("/agents/ghost/ask/stream", json={"question": "q"})

        assert response.status_code == 404

    async def test_a_failure_after_a_step_is_an_error_event(self, client, mock_ipc_channel):
        mock_ipc_channel.stream, _ = _stream(
            AskProgress(id="", name="model_call"),
            AskResponse(id="", success=False, error="oops"),
        )

        response = await client.post("/agents/test/ask/stream", json={"question": "q"})

        assert response.text.endswith('event: error\ndata: {"detail": "oops"}\n\n')


class TestFeedbackRoute:
    """Tests for POST /agents/{name}/feedback endpoint."""

    async def test_feedback(self, client, mock_ipc_channel):
        """Feedback returns success."""
        mock_ipc_channel.submit.return_value = FeedbackAgentResponse(id="1")

        response = await client.post("/agents/test/feedback", json={"message": "Good job!"})

        assert response.status_code == 200
        assert response.json()["success"] is True

    async def test_feedback_nonexistent(self, client, mock_ipc_channel):
        """Feedback returns 404 for nonexistent agent."""
        mock_ipc_channel.submit.return_value = FeedbackAgentResponse(
            id="1", success=False, error="Agent not found: nonexistent"
        )

        response = await client.post("/agents/nonexistent/feedback", json={"message": "test"})

        assert response.status_code == 404


class TestGetInsightsRoute:
    """Tests for GET /agents/{name}/insights endpoint."""

    async def test_get_insights(self, client, mock_ipc_channel):
        """Get insights returns list."""
        mock_ipc_channel.submit.return_value = GetInsightsResponse(
            id="1",
            insights=[
                {"success": True, "content": "Insight 1", "parsed": None, "iterations": 1},
                {
                    "success": True,
                    "content": "Insight 2",
                    "parsed": {"key": "value"},
                    "iterations": 2,
                },
            ],
        )

        response = await client.get("/agents/test/insights")

        assert response.status_code == 200
        data = response.json()
        assert len(data["insights"]) == 2

    async def test_get_insights_with_limit(self, client, mock_ipc_channel):
        """Get insights respects limit parameter."""
        mock_ipc_channel.submit.return_value = GetInsightsResponse(id="1", insights=[])

        response = await client.get("/agents/test/insights?limit=5")

        assert response.status_code == 200
        # Verify the request was made (limit is passed via IPC request)
        mock_ipc_channel.submit.assert_called_once()
        call_args = mock_ipc_channel.submit.call_args
        request = call_args[0][0]
        assert request.limit == 5

    async def test_get_insights_nonexistent(self, client, mock_ipc_channel):
        """Get insights returns 404 for nonexistent agent."""
        mock_ipc_channel.submit.return_value = GetInsightsResponse(
            id="1", success=False, error="Agent not found: nonexistent"
        )

        response = await client.get("/agents/nonexistent/insights")

        assert response.status_code == 404
