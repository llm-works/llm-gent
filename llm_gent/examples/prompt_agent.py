#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run: codebase-qna --smoke
# ci-run: jokester --smoke
# ci-timeout: 30

"""Run one cycle of a shipped prompt agent (``etc/agents/<name>.yaml``).

The agent is built the way ``llm-gent serve`` builds it — its YAML parsed by
the server config, then the default agent factory — and ``run_once()`` runs
its task once. The LLM backends come from the shipped ``etc/llm.yaml``;
``--base-url`` / ``--model`` override its ``local`` backend.

``--smoke`` replaces the model with llm-infer's scripted client: it calls
the agent's ``shell`` tool once, if it has one, then ``complete_task``. The
real client, tool loop, tools and terminal tool run; only the model's
answers are scripted.

The memory tools (``remember`` / ``recall``) need the learning database,
which this example does not connect.

Usage:
    python -m llm_gent.examples.prompt_agent codebase-qna --base-url http://localhost:8000
    python -m llm_gent.examples.prompt_agent jokester --smoke

Flags:
    --base-url URL    OpenAI-compatible server for the local backend (/v1 added if missing)
    --model NAME      Model for the local backend
    --codebase PATH   {{CODEBASE_PATH}} for agents that read a codebase (default: .)
    --smoke           Scripted model; no backend contacted
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from typing import Any

import yaml
from appinfra.log import Logger, create_lg
from llm_infer.client import BackendError, ChatRequest, ChatResponse
from llm_infer.client import Factory as LLMInferFactory
from llm_infer.schemas.openai import FunctionCall, ToolCall

import llm_gent
from llm_gent.agents.default import Factory
from llm_gent.core.platform import Context
from llm_gent.core.platform import Factory as PlatformFactory
from llm_gent.runtime.server.config import AgentServerConfig


ETC = Path(llm_gent.__file__).parent / "etc"


def _smoke_answer(request: ChatRequest) -> ChatResponse:
    """The ``--smoke`` script: ``shell`` once if offered, then ``complete_task``.

    The answer depends on the request alone: ``shell`` while no tool was
    called yet, otherwise ``complete_task``, quoting the shell's result
    (the first tool result) when the agent has a shell.
    """
    offered = {t["function"]["name"] for t in request.tools or []}
    if "shell" not in offered:
        return _tool_call("complete_task", {"status": "done", "conclusion": "smoke run"})
    called = any(m.get("role") == "assistant" and m.get("tool_calls") for m in request.messages)
    if not called:
        return _tool_call("shell", {"command": "ls"})
    output = _first_tool_output(request.messages)
    conclusion = "smoke run" + (f"; shell said: {output!r}" if output else "")
    return _tool_call("complete_task", {"status": "done", "conclusion": conclusion})


def _tool_call(name: str, arguments: dict[str, Any]) -> ChatResponse:
    """A response that makes one tool call."""
    call = ToolCall(
        id=f"call-{name}", function=FunctionCall(name=name, arguments=json.dumps(arguments))
    )
    return ChatResponse(content="", tool_calls=[call], finish_reason="tool_calls")


def _first_tool_output(messages: list[dict[str, Any]]) -> str:
    """The first line of the first tool result in ``messages``, or ``""``."""
    results = [m for m in messages if m.get("role") == "tool"]
    lines = str(results[0].get("content", "")).splitlines() if results else []
    return lines[0] if lines else ""


def _parse_args() -> argparse.Namespace:
    agents = sorted(p.stem for p in (ETC / "agents").glob("*.yaml"))
    parser = argparse.ArgumentParser(description="Run one cycle of a shipped prompt agent")
    parser.add_argument("agent", choices=agents, help="Agent config under etc/agents")
    parser.add_argument("--base-url", help="Endpoint for the local backend (default: etc/llm.yaml)")
    parser.add_argument("--model", help="Model for the local backend (default: etc/llm.yaml)")
    parser.add_argument("--codebase", default=".", help="{{CODEBASE_PATH}} (default: .)")
    parser.add_argument("--smoke", action="store_true", help="Scripted model; no backend contacted")
    return parser.parse_args()


def _llm_config(args: argparse.Namespace) -> dict[str, Any]:
    """``--base-url`` / ``--model`` on the shipped ``local`` backend; the rest stays as shipped."""
    local: dict[str, Any] = {}
    if args.base_url:
        # The OpenAI-compatible backend appends /chat/completions to
        # base_url verbatim, so a bare server root would 404.
        base_url = args.base_url.rstrip("/")
        local["base_url"] = base_url if base_url.endswith("/v1") else base_url + "/v1"
    if args.model:
        local["model"] = args.model
    return {"backends": {"local": local}} if local else {}


def _platform(lg: Logger, args: argparse.Namespace) -> Context:
    """The platform: the shipped backends, or llm-infer's scripted client with ``--smoke``."""
    factory = PlatformFactory(lg)
    if args.smoke:
        factory = factory.with_llm(LLMInferFactory(lg).scripted(_smoke_answer))
        print("(smoke: llm-infer's scripted client, no backend contacted)\n")
    return factory.from_config({"llm": _llm_config(args)})


def _build(platform: Context, args: argparse.Namespace) -> Any:
    """Agent ``args.agent``, built as ``llm-gent serve`` builds it."""
    raw = yaml.safe_load((ETC / "agents" / f"{args.agent}.yaml").read_text())
    agent_config = AgentServerConfig.from_dict({"agents": {args.agent: raw}}).agents[args.agent]
    variables = {"CODEBASE_PATH": str(Path(args.codebase).resolve())}
    return Factory(platform=platform).create(agent_config.factory_config(args.agent), variables)


def main() -> int:
    args = _parse_args()
    lg = create_lg("prompt_agent", "warning")
    with _platform(lg, args) as platform:
        agent = _build(platform, args)
        agent.start()
        try:
            result = agent.run_once()
        except BackendError as e:
            print(f"backend error: {e}", file=sys.stderr)
            return 1
        finally:
            agent.stop()
    print(f"success: {result.success}  iterations: {result.iterations}")
    print(result.content)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
