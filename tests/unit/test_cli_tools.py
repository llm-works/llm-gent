# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for CLI tools."""

import argparse
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_gent.cli.tools.ask import AskTool
from llm_gent.cli.tools.feedback import FeedbackTool
from llm_gent.cli.tools.gateway import GatewayTool
from llm_gent.cli.tools.list import ListTool
from llm_gent.cli.tools.start import StartTool
from llm_gent.cli.tools.stop import StopTool


pytestmark = pytest.mark.unit

SERVER = "http://localhost:8080"


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MagicMock()


@pytest.fixture
def make_tool(
    monkeypatch: pytest.MonkeyPatch, mock_logger: MagicMock
) -> Callable[..., GatewayTool]:
    """Build a tool with parsed ``args`` (``--server`` with a trailing slash) and ``mock_logger``."""

    def make(cls: type[GatewayTool], **args: Any) -> GatewayTool:
        monkeypatch.setattr(cls, "lg", property(lambda _self: mock_logger))
        tool = cls()
        tool._parsed_args = argparse.Namespace(server=f"{SERVER}/", **args)
        return tool

    return make


@pytest.mark.parametrize(
    ("cls", "args", "method", "path", "body"),
    [
        (AskTool, {"name": "a", "question": "q"}, "POST", "/agents/a/ask", {"question": "q"}),
        (StartTool, {"name": "a"}, "POST", "/agents/a/start", None),
        (StopTool, {"name": "a"}, "POST", "/agents/a/stop", None),
        (
            FeedbackTool,
            {"name": "a", "message": "m"},
            "POST",
            "/agents/a/feedback",
            {"message": "m"},
        ),
        (ListTool, {}, "GET", "/agents", None),
    ],
)
def test_each_tool_sends_its_request(
    make_tool: Callable[..., GatewayTool],
    cls: type[GatewayTool],
    args: dict[str, Any],
    method: str,
    path: str,
    body: dict[str, Any] | None,
) -> None:
    tool = make_tool(cls, **args)
    with patch.object(httpx, "request") as request:
        request.return_value.status_code = 200
        request.return_value.json.return_value = {}
        tool.run()
    request.assert_called_once()
    assert request.call_args.args == (method, f"{SERVER}{path}")
    assert request.call_args.kwargs["json"] == body


@pytest.mark.parametrize("cls", [AskTool, StartTool, StopTool, FeedbackTool, ListTool])
def test_http_status_error_is_reported(
    make_tool: Callable[..., GatewayTool], mock_logger: MagicMock, cls: type[GatewayTool]
) -> None:
    """HTTPStatusError is handled gracefully."""
    error_response = MagicMock()
    error_response.status_code = 500
    tool = make_tool(cls)
    with patch.object(httpx, "request") as request:
        request.return_value.status_code = 500
        request.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=error_response
        )
        assert tool._request("POST", "/x", timeout=1.0) is None
    mock_logger.error.assert_called_once()


@pytest.mark.parametrize(
    ("response", "printed"),
    [
        (
            httpx.Response(500, json={"detail": "agent crashed"}),
            "Server returned 500: agent crashed",
        ),
        (httpx.Response(502, text="Bad Gateway"), "Server returned 502: Bad Gateway"),
    ],
)
def test_http_status_error_prints_the_detail(
    make_tool: Callable[..., GatewayTool],
    capsys: pytest.CaptureFixture[str],
    response: httpx.Response,
    printed: str,
) -> None:
    tool = make_tool(AskTool)
    with patch.object(httpx, "request") as request:
        request.return_value.status_code = response.status_code
        request.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
        assert tool._request("POST", "/x", timeout=1.0) is None
    assert printed in capsys.readouterr().out


def test_404_names_the_agent(
    make_tool: Callable[..., GatewayTool], capsys: pytest.CaptureFixture[str]
) -> None:
    tool = make_tool(StartTool)
    with patch.object(httpx, "request") as request:
        request.return_value.status_code = 404
        assert tool._request("POST", "/agents/a/start", timeout=1.0, agent="a") is None
    assert "Agent 'a' not found" in capsys.readouterr().out


def test_400_prints_the_gateway_detail(
    make_tool: Callable[..., GatewayTool], capsys: pytest.CaptureFixture[str]
) -> None:
    tool = make_tool(StopTool)
    with patch.object(httpx, "request") as request:
        request.return_value.status_code = 400
        request.return_value.json.return_value = {"detail": "already stopped"}
        assert tool._request("POST", "/agents/a/stop", timeout=1.0, agent="a") is None
    assert "Error: already stopped" in capsys.readouterr().out


def test_connect_error_names_the_server(
    make_tool: Callable[..., GatewayTool],
    mock_logger: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = make_tool(ListTool)
    with patch.object(httpx, "request", side_effect=httpx.ConnectError("refused")):
        assert tool._request("GET", "/agents", timeout=1.0) is None
    assert f"Could not connect to gateway at {SERVER}" in capsys.readouterr().out
    mock_logger.error.assert_called_once()
