# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Framework-level LLM wrapper with logging and dry-run support.

Sits between llm-infer client and the rest of the framework:
    llm-infer/ChatClient -> llm-gent/LLMCaller -> llm-gent/LLMTrait (agents)
                                                -> RatingService (CLI)

Provides:
- Trace logging of full request/response
- Dry-run mode (log without sending)
- Single place for framework-level LLM concerns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_infer.client.types import AdapterInfo


if TYPE_CHECKING:
    from appinfra.log import Logger
    from llm_infer.client import ChatClient, ChatResponse


@dataclass
class CallResult:
    """Result of an LLM call."""

    content: str
    model: str
    usage: dict[str, int] | None
    tool_calls: list[dict[str, Any]] | None
    adapter: AdapterInfo | None  # LoRA adapter info (None if not requested)
    dry_run: bool  # True if this was a dry-run (no actual LLM call)
    raw_response: ChatResponse | None  # None in dry-run mode


class LLMCaller:
    """Framework wrapper for LLM calls with logging and dry-run.

    Wraps llm-infer's ChatClient to add framework-level concerns:
    - Trace logging of full request/response (for debugging)
    - Dry-run mode (log what would be sent without calling LLM)
    - Consistent interface for both agent and CLI contexts

    Example:
        from llm_infer.client import Factory as LLMClientFactory

        router = LLMClientFactory(lg).from_config(config)
        caller = LLMCaller(lg, router)

        # Normal call
        result = caller.chat(messages=[{"role": "user", "content": "Hello"}])

        # Dry-run (logs but doesn't call LLM)
        caller_dry = LLMCaller(lg, router, dry_run=True)
        result = caller_dry.chat(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        lg: Logger,
        router: ChatClient,
        dry_run: bool = False,
    ) -> None:
        """Initialize LLM caller.

        Args:
            lg: Logger instance.
            router: LLM router from llm-infer.
            dry_run: If True, log requests but don't send to LLM.
        """
        self._lg = lg
        self._router = router
        self._dry_run = dry_run

    @property
    def router(self) -> ChatClient:
        """Access underlying router (for compatibility)."""
        return self._router

    @property
    def dry_run(self) -> bool:
        """Whether dry-run mode is enabled."""
        return self._dry_run

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        backend: str | None = None,
        adapter: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CallResult:
        """Call LLM with logging and optional dry-run.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model to use (None = router default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions in OpenAI format.
            backend: Specific backend to use.
            adapter: LoRA adapter.
            extra_body: Extra request body parameters.

        Returns:
            CallResult with response content and metadata.
        """
        self._log_request(messages, model, backend, temperature, max_tokens, adapter, tools)

        if self._dry_run:
            return self._dry_run_result(model)

        # Build kwargs for optional parameters
        kwargs: dict[str, Any] = {}
        if backend is not None:
            kwargs["backend"] = backend
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        response = self._router.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            adapter=adapter,
            **kwargs,
        )

        self._log_response(response)
        return self._to_result(response)

    def _dry_run_result(self, model: str | None) -> CallResult:
        """Return stub result for dry-run mode."""
        self._lg.info("dry-run: skipping LLM call")
        return CallResult(
            content="[dry-run: no response]",
            model=model or "dry-run",
            usage=None,
            tool_calls=None,
            adapter=None,
            dry_run=True,
            raw_response=None,
        )

    def _to_result(self, response: ChatResponse) -> CallResult:
        """Convert ChatResponse to CallResult."""
        tool_calls = None
        if response.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response.tool_calls
            ]

        # Convert usage object to dict to match declared type
        raw_usage = getattr(response, "usage", None)
        usage_dict = None
        if raw_usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(raw_usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
            }

        return CallResult(
            content=response.content or "",
            model=response.model or "unknown",
            usage=usage_dict,
            tool_calls=tool_calls,
            adapter=response.adapter,
            dry_run=False,
            raw_response=response,
        )

    def _log_request(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        backend: str | None,
        temperature: float,
        max_tokens: int | None,
        adapter: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Log full LLM request at trace level."""
        self._lg.trace(
            "llm request",
            extra={
                "backend": backend,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "adapter": adapter,
                "messages": messages,
                "tools": [t.get("function", {}).get("name") for t in (tools or [])],
                "dry_run": self._dry_run,
            },
        )

    def _log_response(self, response: ChatResponse) -> None:
        """Log full LLM response at trace level."""
        self._lg.trace(
            "llm response",
            extra={
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
                "tool_calls": [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (response.tool_calls or [])
                ],
                "adapter": response.adapter,
            },
        )

    def close(self) -> None:
        """Close the underlying router."""
        self._router.close()
