#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Quickstart for llm-gent 0.3.1.

Demonstrates the working shape of the shipped API:

- Logger via ``appinfra.log.create_lg``.
- Agent as a small concrete subclass (the public ``Agent`` class is abstract
  and ships without a ready-to-instantiate default at 0.3.x).
- Config as a plain dict with a nested identity block
  (``{"identity": {"name": ...}}``) — the exported ``Config`` model does not
  carry the identity contract the base Agent reads internally.
- System message injected manually into the message list — ``DirectiveTrait``
  does not auto-inject its directive into ``LLMTrait.complete()`` calls yet.
- Dict-form messages passed to ``LLMTrait.complete()``.

Set ``LLM_GENT_SMOKE=1`` to swap in a stub router that returns a canned
response without contacting a real backend — used by CI's wheel-smoke job.
"""

from __future__ import annotations

import os
from typing import Any

from appinfra import DotDict
from appinfra.log import create_lg
from llm_infer.client import ChatResponse

from llm_gent import Agent, LLMTrait
from llm_gent.core.agent.types import ExecutionResult


SMOKE = os.getenv("LLM_GENT_SMOKE") == "1"


class _StubRouter:
    """Stub ChatClient used when LLM_GENT_SMOKE=1."""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content="Hello from smoke-mode stub!")

    def close(self) -> None:
        pass


class HelloAgent(Agent):
    """Minimal concrete Agent.

    The public ``Agent`` class is abstract in 0.3.x. Real applications give the
    lifecycle and execution methods meaningful bodies; this quickstart calls
    ``LLMTrait.complete()`` directly, so the abstract methods are trivial stubs.
    """

    def start(self) -> None:
        self._start_traits()

    def stop(self) -> None:
        self._stop_traits()

    def run_once(self) -> ExecutionResult:
        return ExecutionResult(success=True, content="")

    def ask(self, question: str) -> str:
        return ""

    def record_feedback(self, message: str) -> None:
        pass

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        return []


def _build_llm_config() -> DotDict:
    """Point LLMTrait at a local OpenAI-compatible endpoint by default.

    Override the endpoint and model with ``LLM_GENT_BASE_URL`` and
    ``LLM_GENT_MODEL`` env vars; the openai_compatible backend picks up
    ``OPENAI_API_KEY`` from the environment automatically.
    """
    return DotDict(
        {
            "default": "local",
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": os.getenv("LLM_GENT_BASE_URL", "http://localhost:8000/v1"),
                    "model": os.getenv("LLM_GENT_MODEL", "default"),
                }
            },
        }
    )


def main() -> None:
    lg = create_lg("hello-agent", "info")

    # Agent reads config.identity.name internally.
    agent = HelloAgent(lg, {"identity": {"name": "hello-agent"}})
    agent.add_trait(LLMTrait(agent, _build_llm_config()))
    agent.start()

    llm = agent.require_trait(LLMTrait)
    if SMOKE:
        # Duck-typed stub — matches the .chat() surface LLMTrait actually calls.
        if llm._router is not None:
            llm._router.close()  # Close real client before replacing
        llm._router = _StubRouter()  # type: ignore[assignment]

    result = llm.complete(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello."},
        ]
    )
    print(f"agent said: {result.content}")

    agent.stop()


if __name__ == "__main__":
    main()
