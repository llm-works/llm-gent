# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Base for the CLI tools that call the agent gateway's HTTP API."""

import argparse
import json
from typing import Any

import httpx
from appinfra.app.tools import Tool


class GatewayTool(Tool):
    """A CLI tool calling the agent gateway: the ``--server`` flag and request handling."""

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--server",
            default="http://localhost:8080",
            help="Gateway server URL (default: http://localhost:8080)",
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        agent: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Send ``method`` ``path`` to the gateway: its JSON reply, or ``None`` once the error is printed.

        A 404 reports ``agent`` as not found; other errors print the gateway's ``detail``.
        """
        server = self.args.server.rstrip("/")
        try:
            response = httpx.request(method, f"{server}{path}", json=body, timeout=timeout)
            if response.status_code == 404 and agent is not None:
                print(f"Error: Agent '{agent}' not found")
                return None
            if response.status_code == 400:
                print(f"Error: {response.json().get('detail', 'Unknown error')}")
                return None
            response.raise_for_status()
            return dict(response.json())
        except httpx.RequestError as e:
            self.lg.error("failed to connect to server", extra={"exception": e})
            print(f"Error: Could not connect to gateway at {server}")
            return None
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = _detail(e.response)
            self.lg.error("server error", extra={"status": status, "detail": detail})
            print(f"Error: Server returned {status}" + (f": {detail}" if detail else ""))
            return None
        except json.JSONDecodeError:
            self.lg.error("invalid JSON response from server")
            print("Error: Server returned invalid response")
            return None


def _detail(response: httpx.Response) -> str:
    """The gateway's ``detail`` field, or the raw body when the reply is not a JSON object."""
    try:
        return str(response.json()["detail"])
    except (ValueError, TypeError, KeyError):
        return response.text
