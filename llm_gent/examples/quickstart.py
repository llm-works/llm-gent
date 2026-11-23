#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run: --smoke

"""Quickstart for llm-gent.

- :class:`PlatformFactory` builds the platform: the model router every agent
  shares, from the packaged ``etc/llm.yaml`` with ``--base-url`` / ``--model``
  on its ``local`` backend.
- The prompt-agent factory (:class:`llm_gent.agents.default.Factory`) builds
  an agent on that platform from a config: its ``directive`` becomes the
  system prompt of the agent's SAIA, which runs over the platform's router.
- ``agent.ask()`` runs one task; the agent answers by calling its
  ``complete_task`` tool.

Pass ``--smoke`` to run on llm-infer's scripted client instead of a backend.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from appinfra.app import AppBuilder
from llm_infer.client import ChatRequest, ChatResponse
from llm_infer.client import Factory as LLMInferFactory
from llm_infer.schemas.openai import FunctionCall, ToolCall

from llm_gent import PlatformFactory
from llm_gent.agents.default import Factory as PromptAgentFactory


app = (
    AppBuilder("quickstart")
    .with_description("llm-gent quick-start hello-agent")
    .cli.with_flags(log_level=True)
    .done()
    .tools.with_main("run")
    .done()
    .build()
)


def _smoke_answer(request: ChatRequest) -> ChatResponse:
    """The ``--smoke`` model's reply: it completes the task with a greeting."""
    arguments = {"status": "done", "conclusion": "Hello from the scripted model!"}
    call = ToolCall(
        id="call-1",
        type="function",
        function=FunctionCall(name="complete_task", arguments=json.dumps(arguments)),
    )
    return ChatResponse(content="", tool_calls=[call])


def _llm_config(base_url: str | None, model: str | None) -> dict[str, Any]:
    """``--base-url`` / ``--model`` on the packaged ``local`` backend."""
    local = {k: v for k, v in (("base_url", base_url), ("model", model)) if v}
    return {"backends": {"local": local}} if local else {}


@app.tool(name="run", help="Run the hello-agent example")
@app.argument("--base-url", help="OpenAI-compatible endpoint (default: etc/llm.yaml)")
@app.argument("--model", help="Model name (default: etc/llm.yaml)")
@app.argument("--smoke", action="store_true", help="Scripted model; no backend contacted")
def run(self: Any) -> int:
    factory = PlatformFactory(self.lg)
    if self.args.smoke:
        factory = factory.with_llm(LLMInferFactory(self.lg).scripted(_smoke_answer))
    with factory.from_config({"llm": _llm_config(self.args.base_url, self.args.model)}) as platform:
        agent = PromptAgentFactory(platform).from_config(
            {
                "identity": {"name": "hello-agent"},
                "directive": "You are a concise assistant.",
                "tools": {"complete_task": {}},
            }
        )
        agent.start()
        try:
            answer = agent.ask("Say hello.")
        finally:
            agent.stop()
    print(f"agent said: {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(app.main())
