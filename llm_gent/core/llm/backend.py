# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""LLM backend abstraction for agent framework."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any, Protocol, cast

import httpx

from .types import CompletionResult, Message


class LLMBackend(Protocol):
    """Protocol for LLM backends.

    Implementations must provide completion and adapter management.
    """

    def complete(
        self,
        messages: Sequence[Message | dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Generate a completion.

        Args:
            messages: Conversation messages. Accepts Message instances or
                OpenAI-style dicts (``{"role": ..., "content": ...}``).
            model: Model identifier (uses default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions in OpenAI format (optional).

        Returns:
            CompletionResult with generated content and metadata.
            If tools were provided and LLM wants to call them,
            result.tool_calls will be populated.
        """
        ...

    def load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter."""
        ...

    def unload_adapter(self) -> None:
        """Unload current adapter, revert to base model."""
        ...


class LLMError(Exception):
    """Error from LLM backend."""

    pass


class StructuredOutputError(Exception):
    """Raised when structured output parsing or validation fails."""

    pass


class HTTPBackend:
    """Backend that calls an OpenAI-compatible HTTP API.

    Uses httpx for synchronous HTTP requests. Compatible with any
    OpenAI-compatible API at /v1/chat/completions endpoint.

    Note: No API key support - external services go through a proxy.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        default_model: str = "default",
    ) -> None:
        """Initialize HTTP backend.

        Args:
            base_url: Base URL for API (e.g., "http://localhost:8000/v1").
            timeout: Request timeout in seconds.
            default_model: Model to use when not specified.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_model = default_model
        self._adapter_path: str | None = None
        self._client = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()

    def complete(
        self,
        messages: Sequence[Message | dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Generate a completion using the HTTP API."""
        start_time = time.monotonic()

        # Normalize dict messages to Message objects
        normalized: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in messages]
        payload = self._build_payload(normalized, model, temperature, max_tokens, tools)
        response = self._send_request(payload)
        result = self._parse_response(response)

        latency_ms = int((time.monotonic() - start_time) * 1000)
        return CompletionResult(
            id=result["id"],
            content=result["content"],
            model=result["model"],
            tokens_used=result["tokens_used"],
            latency_ms=latency_ms,
            tool_calls=result.get("tool_calls"),
        )

    def load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter.

        Note: Stores adapter path locally. Actual loading depends on
        server support (Phase 5).
        """
        self._adapter_path = adapter_path

    def unload_adapter(self) -> None:
        """Unload current adapter."""
        self._adapter_path = None

    def _build_payload(
        self,
        messages: list[Message],
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the request payload."""
        # Convert messages to API format
        api_messages = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            api_messages.append(msg)

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": api_messages,
            "temperature": temperature,
            "stream": False,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools

        return payload

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send completion request to the API."""
        url = f"{self._base_url}/chat/completions"

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response else ""
            raise LLMError(f"LLM request failed ({e.response.status_code}): {body}") from e
        except httpx.RequestError as e:
            raise LLMError(f"LLM connection failed: {e}") from e

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse the API response."""
        try:
            choice = response["choices"][0]
            message = choice["message"]
            usage = response.get("usage", {})

            # Generate ID if not provided (some APIs don't include it)
            response_id = response.get("id") or str(uuid.uuid4())

            # Extract tool calls if present
            tool_calls = message.get("tool_calls")

            return {
                "id": response_id,
                "content": message.get("content") or "",
                "model": response.get("model", self._default_model),
                "tokens_used": usage.get("total_tokens", 0)
                or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
                "tool_calls": tool_calls,
            }
        except (KeyError, IndexError) as e:
            raise LLMError(f"Invalid API response: {e}") from e
