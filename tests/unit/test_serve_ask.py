# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The gateway's main process answering asks from the HTTP process: the agent's reply, as is."""

import queue
from unittest.mock import MagicMock

import pytest

from llm_gent.bus.protocol import AskProgress, AskRequest, AskResponse
from llm_gent.cli.tools.serve import ServeTool
from llm_gent.runtime.server.protocol.management import AskAgentRequest


pytestmark = pytest.mark.unit


def _serve(hub, request) -> list:
    """The replies the main process puts on the response queue for ``request``."""
    request_q: queue.Queue = queue.Queue()
    response_q: queue.Queue = queue.Queue()
    request_q.put(request)
    ServeTool()._ipc_loop(hub, request_q, response_q, is_shutdown=request_q.empty)
    return [response_q.get_nowait() for _ in range(response_q.qsize())]


def _ask() -> AskAgentRequest:
    return AskAgentRequest(agent_name="a", ask=AskRequest(id="r-1", question="q", stream=True))


class TestServeAsk:
    def test_every_chunk_of_the_agents_reply_goes_on_the_response_queue(self):
        reply = [
            AskProgress(id="r-1", name="model_call"),
            AskResponse(id="r-1", response="hello", session="s-1"),
        ]
        hub = MagicMock()
        hub.forward.return_value = iter(reply)
        req = _ask()

        assert _serve(hub, req) == reply
        hub.forward.assert_called_once_with("a", req.ask)

    def test_a_failure_to_get_the_reply_is_its_last_chunk(self):
        hub = MagicMock()
        hub.forward.side_effect = KeyError("No channel for agent: a")

        [reply] = _serve(hub, _ask())

        assert (reply.id, reply.success) == ("r-1", False)
