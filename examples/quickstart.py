#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Quickstart for llm-gent.

Tutorial-shape agent construction:

- :class:`AgentFactory` takes a bare :class:`~appinfra.log.Logger`; no
  :class:`~llm_gent.core.platform.PlatformContext` boilerplate.
- :meth:`AgentFactory.from_config` builds and returns a plain concrete
  :class:`Agent` with traits attached — no subclass required.
- Config is a plain dict with a nested identity block plus per-trait keys and
  ``traits.required`` selecting which traits to attach.
- ``DirectiveTrait`` auto-prepends its directive as a system message on
  ``LLMTrait.complete()`` — no manual wiring.

Set ``LLM_GENT_SMOKE=1`` to swap in a stub router that returns a canned
response without contacting a real backend — used by CI's wheel-smoke job.
"""

from __future__ import annotations

import os
from typing import Any

from appinfra.log import create_lg
from llm_infer.client import ChatResponse

from llm_gent import AgentFactory, LLMTrait


SMOKE = os.getenv("LLM_GENT_SMOKE") == "1"


class _StubRouter:
    """Stub ChatClient used when LLM_GENT_SMOKE=1."""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content="Hello from smoke-mode stub!")

    def close(self) -> None:
        pass


def _llm_config() -> dict[str, Any]:
    """Point LLMTrait at a local OpenAI-compatible endpoint by default.

    Override the endpoint and model with ``LLM_GENT_BASE_URL`` and
    ``LLM_GENT_MODEL`` env vars; the openai_compatible backend picks up
    ``OPENAI_API_KEY`` from the environment automatically.
    """
    return {
        "default": "local",
        "backends": {
            "local": {
                "type": "openai_compatible",
                "base_url": os.getenv("LLM_GENT_BASE_URL", "http://localhost:8000/v1"),
                "model": os.getenv("LLM_GENT_MODEL", "default"),
            }
        },
    }


def main() -> None:
    lg = create_lg("hello-agent", "info")

    agent = AgentFactory(lg).from_config(
        {
            "identity": {"name": "hello-agent"},
            "llm": _llm_config(),
            "directive": "You are a concise assistant.",
            "traits": {"required": ["llm", "directive"]},
        }
    )
    agent.start()

    llm = agent.require_trait(LLMTrait)
    if SMOKE:
        if llm._router is not None:
            llm._router.close()
        llm._router = _StubRouter()  # type: ignore[assignment]

    result = llm.complete([{"role": "user", "content": "Say hello."}])
    print(f"agent said: {result.content}")

    agent.stop()


if __name__ == "__main__":
    main()
