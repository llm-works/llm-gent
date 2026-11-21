#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run: --smoke

"""Quickstart for llm-gent.

- :class:`PlatformFactory` builds the platform: the model router every agent
  shares, from the packaged ``etc/llm.yaml`` with ``--base-url`` / ``--model``
  on its ``local`` backend.
- :meth:`AgentFactory.from_config` builds a plain :class:`Agent` on that
  platform, with the traits ``traits.required`` names.
- ``DirectiveTrait`` prepends its directive as a system message on
  ``LLMTrait.complete()``.

Pass ``--smoke`` to run on llm-infer's scripted client instead of a backend.
"""

from __future__ import annotations

import sys
from typing import Any

from appinfra.app import AppBuilder
from llm_infer.client import ChatRequest, ChatResponse
from llm_infer.client import Factory as LLMInferFactory

from llm_gent import AgentFactory, LLMTrait, PlatformFactory


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
    """The ``--smoke`` model's reply."""
    return ChatResponse(content="Hello from the scripted model!")


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
        agent = AgentFactory(platform).from_config(
            {
                "identity": {"name": "hello-agent"},
                "directive": "You are a concise assistant.",
                "traits": {"required": ["llm", "directive"]},
            }
        )
        agent.start()
        try:
            result = agent.require_trait(LLMTrait).complete(
                [{"role": "user", "content": "Say hello."}]
            )
        finally:
            agent.stop()
    print(f"agent said: {result.content}")
    return 0


if __name__ == "__main__":
    sys.exit(app.main())
