# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""LLM trait for agent completion capability."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

from llm_infer.client import ChatClient, ChatResponse
from pydantic import BaseModel, ValidationError

from ...llm.backend import StructuredOutputError
from ...llm.types import CompletionResult, Message
from ..base import BaseTrait
from .directive import DirectiveTrait


if TYPE_CHECKING:
    from ...agent import Agent


from appinfra import DotDict


# Type alias for LLM configuration
LLMConfig = DotDict
"""LLM configuration as DotDict.

Supports multi-backend format (see llm-infer LLMClient.from_config):

    default: local
    backends:
      local:
        type: openai_compatible
        base_url: http://localhost:8000/v1
        model: qwen2.5-72b
      anthropic:
        type: anthropic
        model: claude-sonnet-4-20250514
"""


def _normalize_messages(
    messages: Sequence[Message | dict[str, Any]],
) -> list[Message]:
    """Accept both Message instances and OpenAI-style dicts.

    A dict must have the shape {"role": ..., "content": ..., ...} matching
    Message fields. Conversion uses Message(**d) so pydantic validates.
    """
    return [m if isinstance(m, Message) else Message(**m) for m in messages]


def _resolve_llm_defaults(config: LLMConfig) -> dict[str, Any]:
    """Extract default values from LLM config.

    Returns dict with model, temperature, max_tokens, adapter from the selected backend.
    """
    backends = config.get("backends", {})
    default_name = config.get("default")

    if not backends:
        # Single backend config (no "backends" wrapper)
        return {
            "model": config.get("model", "default"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens"),
            "adapter": config.get("adapter"),
        }

    if not default_name:
        default_name = next(iter(backends.keys()))

    backend_config = backends.get(default_name, {})
    return {
        "model": backend_config.get("model", "default"),
        "temperature": backend_config.get("temperature", 0.7),
        "max_tokens": backend_config.get("max_tokens"),
        "adapter": backend_config.get("adapter"),
    }


class LLMTrait(BaseTrait):
    """LLM capability trait.

    Wraps ``llm_infer.client.ChatClient`` to provide completion capability
    to agents with multi-backend support. The router is injected at
    construction: ``TraitFactory.create_llm_trait`` builds one from config,
    or tests / smoke paths can pass a stub directly.

    Dependency ownership: the trait imports no factory. Router construction
    lives in ``TraitFactory._create_llm``. Two seams:

    - Config-time (test / smoke / declarative): ``TraitFactory`` builds the
      router from ``llm_config``. Factory-built routers set
      ``owns_router=True`` so the trait closes them on ``on_stop``.
    - Direct injection (advanced): pass ``router=`` yourself with
      ``owns_router=False``; caller retains close responsibility. See
      ``.with_router()`` for the immutable-view fluent form.

    Lifecycle:
        - ``__init__``: takes an already-built router.
        - ``on_start()``: resolves defaults from config (no router build).
        - ``on_stop()``: closes the router iff ``owns_router`` is True.
    """

    def __init__(
        self,
        agent: Agent,
        router: ChatClient,
        config: LLMConfig | None = None,
        *,
        owns_router: bool = False,
    ) -> None:
        """Initialize LLM trait with an injected router.

        Args:
            agent: The agent this trait belongs to.
            router: Live ``ChatClient``. Built by ``TraitFactory`` in the
                standard path; may be a stub for tests / smoke.
            config: LLM configuration dict. Used only for defaults
                (model / temperature / adapter) — construction of the
                router itself is the factory's job.
            owns_router: If True, ``on_stop`` closes the router. Set by
                ``TraitFactory`` for factory-built routers. Callers that
                inject their own router keep this False and clean up
                themselves.
        """
        super().__init__(agent)
        self.config: LLMConfig = config or DotDict()
        self._router: ChatClient = router
        self._owns_router = owns_router
        self._defaults: dict[str, Any] = {}
        self._last_adapter_fallback_warning: float = 0.0

    def on_start(self) -> None:
        """Resolve config-derived defaults. Router is already injected."""
        self.agent.lg.trace(
            "starting LLM trait...",
            extra={"agent": self.agent.name, "owns_router": self._owns_router},
        )
        self._defaults = _resolve_llm_defaults(self.config)
        self.agent.lg.trace(
            "LLM trait started",
            extra={
                "agent": self.agent.name,
                "owns_router": self._owns_router,
                "adapter": self._defaults.get("adapter"),
            },
        )

    def on_stop(self) -> None:
        """Close the router iff this trait owns its lifecycle."""
        self.agent.lg.trace(
            "stopping LLM trait...",
            extra={"agent": self.agent.name, "owns_router": self._owns_router},
        )
        if self._owns_router:
            self._router.close()
        self.agent.lg.trace(
            "LLM trait stopped",
            extra={"agent": self.agent.name, "closed_router": self._owns_router},
        )

    @property
    def router(self) -> ChatClient:
        """Access the injected LLM router."""
        return self._router

    def with_router(self, router: ChatClient) -> Self:
        """Return a new trait bound to ``router``, detached from the registry.

        Immutable-view fluent (mirrors ``llm_infer.client.BoundChatClient``):
        ``self`` remains canonical for ``agent.get_trait(LLMTrait)`` and its
        router is unchanged. The returned instance shares this trait's
        ``agent`` and ``config`` but is not registered — it exists for the
        caller's immediate use (e.g. one-shot swap for a specific call).

        Ownership: ``owns_router`` on the returned trait is False. The caller
        owns ``router``'s lifecycle; the returned trait will not close it.
        For a persistent swap, write the returned instance back into the
        agent's trait registry explicitly.

        Concurrency: safe to construct while ``self.complete`` is in flight,
        but the caller must not call ``.complete()`` on the returned instance
        from another thread mid-call; there are no locks.
        """
        new = type(self)(self.agent, router, self.config, owns_router=False)
        new._defaults = _resolve_llm_defaults(new.config)
        self.agent.lg.debug(
            "LLM trait detached with new router",
            extra={"agent": self.agent.name, "detached_from_registry": True},
        )
        return new

    @property
    def adapter(self) -> str | None:
        """Get the default adapter from config, if any."""
        return self._defaults.get("adapter")

    def complete(
        self,
        messages: Sequence[Message | dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        output_schema: type[BaseModel] | None = None,
        backend: str | None = None,
        adapter: str | None = None,
    ) -> CompletionResult:
        """Generate a completion.

        Args:
            messages: Conversation messages. Accepts Message instances or
                OpenAI-style dicts (``{"role": ..., "content": ...}``); dicts
                are converted to Message via pydantic validation.
            model: Model override (uses config default if None). Also used for
                model-based routing if model is in the routing table.
            temperature: Temperature override (uses config default if None).
            max_tokens: Max tokens override (uses config default if None).
            tools: Tool definitions for function calling.
            output_schema: Pydantic model class for structured output. When provided,
                the LLM is instructed to return JSON matching the schema, and the
                response is validated. Result.parsed will contain the validated object.
            backend: Backend to route to. If None, uses model-based routing or default.
            adapter: LoRA adapter to use (uses config default if None).

        Returns:
            CompletionResult with content and metadata. If output_schema was provided,
            result.parsed contains the validated Pydantic object.

        Raises:
            ValueError: If both tools and output_schema are provided.
            StructuredOutputError: If JSON parsing or schema validation fails.
        """
        if tools and output_schema:
            raise ValueError("Cannot use both tools and output_schema")

        normalized = self._apply_directive(_normalize_messages(messages))
        api_messages, extra_body = self._prepare_messages(normalized, output_schema)
        params = self._resolve_params(model, temperature, max_tokens, adapter)

        self._log_request(
            api_messages, params["model"], params["temp"], params["max_tokens"], params["adapter"]
        )

        response = self.router.chat(
            messages=api_messages,
            model=params["model"],
            temperature=params["temp"],
            max_tokens=params["max_tokens"],
            tools=tools,
            backend=backend,
            adapter=params["adapter"],
            extra_body=extra_body,
        )

        self._log_response(response)
        result = self._response_to_result(response)

        if output_schema:
            result.parsed = self._parse_structured_output(result.content, output_schema)

        return result

    async def complete_async(
        self,
        messages: Sequence[Message | dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        output_schema: type[BaseModel] | None = None,
        backend: str | None = None,
        adapter: str | None = None,
    ) -> CompletionResult:
        """Generate a completion asynchronously.

        Same as complete() but uses async I/O for concurrent requests.
        Use with asyncio.gather() for parallel LLM calls.

        Args:
            messages: Conversation messages. Accepts Message instances or
                OpenAI-style dicts (``{"role": ..., "content": ...}``); dicts
                are converted to Message via pydantic validation.
            model: Model override (uses config default if None).
            temperature: Temperature override (uses config default if None).
            max_tokens: Max tokens override (uses config default if None).
            tools: Tool definitions for function calling.
            output_schema: Pydantic model class for structured output.
            backend: Backend to route to.
            adapter: LoRA adapter to use.

        Returns:
            CompletionResult with content and metadata.
        """
        if tools and output_schema:
            raise ValueError("Cannot use both tools and output_schema")

        normalized = self._apply_directive(_normalize_messages(messages))
        api_messages, extra_body = self._prepare_messages(normalized, output_schema)
        params = self._resolve_params(model, temperature, max_tokens, adapter)

        self._log_request(
            api_messages, params["model"], params["temp"], params["max_tokens"], params["adapter"]
        )

        response = await self.router.chat_async(
            messages=api_messages,
            model=params["model"],
            temperature=params["temp"],
            max_tokens=params["max_tokens"],
            tools=tools,
            backend=backend,
            adapter=params["adapter"],
            extra_body=extra_body,
        )

        self._log_response(response)
        result = self._response_to_result(response)

        if output_schema:
            result.parsed = self._parse_structured_output(result.content, output_schema)

        return result

    def _apply_directive(self, messages: list[Message]) -> list[Message]:
        """Prepend DirectiveTrait's prompt as system message when attached.

        Directive comes first: if the caller already supplied a system message
        at index 0, the directive is merged as "<directive>\\n\\n<caller-system>"
        so the agent's core purpose wins ordering while caller context is
        preserved. Without an attached DirectiveTrait, messages pass through
        unchanged.

        Contract: a system message, if present, must be at index 0. This matches
        Anthropic's Messages API (which forbids ``role: "system"`` inside
        ``messages`` — adapters hoist index-0 out to the top-level ``system``
        parameter) and OpenAI convention. A ``role="system"`` at index >0
        raises ``ValueError`` — silent misplacement would produce two system
        messages after this merge and adapter-dependent behavior downstream.
        """
        if any(m.role == "system" for m in messages[1:]):
            raise ValueError(
                "system message must be at index 0; found role='system' at a non-zero position"
            )

        directive_trait = self.agent.get_trait(DirectiveTrait)
        if directive_trait is None:
            return messages

        directive_prompt = directive_trait.directive.prompt
        if messages and messages[0].role == "system":
            merged = Message(
                role="system",
                content=f"{directive_prompt}\n\n{messages[0].content}",
            )
            return [merged, *messages[1:]]
        return [Message(role="system", content=directive_prompt), *messages]

    def _prepare_messages(
        self, messages: list[Message], output_schema: type[BaseModel] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Prepare messages and extra_body for API call."""
        api_messages = self._messages_to_dicts(messages)
        extra_body = None
        if output_schema:
            api_messages = self._inject_schema_prompt(api_messages, output_schema)
            extra_body = {"response_format": {"type": "json_object"}}
        return api_messages, extra_body

    def _resolve_params(
        self,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        adapter: str | None,
    ) -> dict[str, Any]:
        """Resolve parameters with defaults."""
        return {
            "model": model or self._defaults.get("model"),
            "temp": temperature
            if temperature is not None
            else self._defaults.get("temperature", 0.7),
            "max_tokens": max_tokens
            if max_tokens is not None
            else self._defaults.get("max_tokens"),
            "adapter": adapter or self._defaults.get("adapter"),
        }

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to API format."""
        result = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            result.append(msg)
        return result

    def _response_to_result(self, response: ChatResponse) -> CompletionResult:
        """Convert ChatResponse to CompletionResult."""
        import uuid

        self._check_adapter_fallback(response)
        tokens_used = self._extract_tokens(response)
        tool_calls = self._extract_tool_calls(response)

        return CompletionResult(
            id=str(uuid.uuid4()),
            content=response.content,
            model=response.model or self._defaults.get("model", "unknown"),
            tokens_used=tokens_used,
            latency_ms=0,
            tool_calls=tool_calls,
            adapter=response.adapter,
        )

    def _extract_tokens(self, response: ChatResponse) -> int:
        """Extract token count from response usage."""
        if not response.usage:
            return 0
        return response.usage.total_tokens or (
            (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
        )

    def _extract_tool_calls(self, response: ChatResponse) -> list[dict[str, Any]] | None:
        """Extract and convert tool calls from response."""
        if not response.tool_calls:
            return None
        return [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in response.tool_calls
        ]

    def _check_adapter_fallback(self, response: ChatResponse) -> None:
        """Log if adapter fallback occurred (warning every 10 min, else debug)."""
        if not response.adapter or not response.adapter.fallback:
            return

        extra = {"adapter": response.adapter}

        now = time.monotonic()
        if now - self._last_adapter_fallback_warning >= 600:
            self.agent.lg.warning("adapter not available, using base model", extra=extra)
            self._last_adapter_fallback_warning = now
        else:
            self.agent.lg.debug("adapter not available, using base model", extra=extra)

    def _log_request(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        adapter: str | None,
    ) -> None:
        """Log full LLM request at trace level."""
        self.agent.lg.trace(
            "llm request",
            extra={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "adapter": adapter,
                "messages": messages,
            },
        )

    def _log_response(self, response: ChatResponse) -> None:
        """Log full LLM response at trace level."""
        self.agent.lg.trace(
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

    def _build_schema_prompt(self, schema: type[BaseModel]) -> str:
        """Generate prompt instructing LLM to output JSON matching schema."""
        json_schema = schema.model_json_schema()
        return (
            "You must respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
            "Respond ONLY with the JSON object, no other text."
        )

    def _inject_schema_prompt(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Inject schema instruction into messages.

        Appends schema prompt to existing system message at index 0, or creates
        one. See ``_apply_directive`` for the position-0-only contract; a
        ``role="system"`` at index >0 raises ``ValueError``.
        """
        schema_prompt = self._build_schema_prompt(schema)
        result = list(messages)  # shallow copy

        if any(m.get("role") == "system" for m in result[1:]):
            raise ValueError(
                "system message must be at index 0; found role='system' at a non-zero position"
            )

        if result and result[0].get("role") == "system":
            # Append to existing system message
            result[0] = {
                **result[0],
                "content": f"{result[0]['content']}\n\n{schema_prompt}",
            }
        else:
            # Insert new system message at the start
            result.insert(0, {"role": "system", "content": schema_prompt})

        return result

    def _parse_structured_output(
        self,
        content: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """Parse JSON content and validate against schema.

        Args:
            content: Raw JSON string from LLM response.
            schema: Pydantic model class to validate against.

        Returns:
            Validated Pydantic model instance.

        Raises:
            StructuredOutputError: If JSON is invalid or doesn't match schema.
        """
        data = self._clean_and_parse_json(content, schema)

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            self._agent._lg.debug(
                "schema validation failed",
                extra={"parsed_data": data, "expected_fields": list(schema.model_fields.keys())},
            )
            raise StructuredOutputError(f"Response doesn't match schema: {e}") from e

    def _clean_and_parse_json(self, content: str, schema: type[BaseModel]) -> object:
        """Clean and parse JSON content, handling LLM quirks."""
        from ...llm.json_cleaner import JSONCleaner

        cleaner = JSONCleaner()

        try:
            cleaned = cleaner.clean(content)
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            self._agent._lg.warning(
                "failed to parse LLM JSON output",
                extra={"raw_content": content, "error": str(e)},
            )
            raise StructuredOutputError(f"Invalid JSON in response: {e}") from e

        if isinstance(data, dict):
            self._reject_if_schema(data, cleaner, content)
            data = cleaner.clean_parsed(data, set(schema.model_fields.keys()))
        return data

    def _reject_if_schema(self, data: dict[str, Any], cleaner: Any, content: str) -> None:
        """Raise error if LLM returned schema definition instead of data."""
        if cleaner._looks_like_schema(data):
            self._agent._lg.warning(
                "LLM returned schema definition instead of data", extra={"raw_content": content}
            )
            raise StructuredOutputError(
                "LLM returned JSON schema definition instead of actual data. "
                "The model may not understand the structured output instruction."
            )


class LLMTraitBackend:
    """Adapter that wraps LLMTrait to satisfy LLMBackend protocol.

    Provides an LLMBackend-compatible interface for code that needs it.

    Example:
        llm_trait = agent.require_trait(LLMTrait)
        backend = LLMTraitBackend(llm_trait)
    """

    def __init__(self, llm_trait: LLMTrait) -> None:
        self._trait = llm_trait

    def complete(
        self,
        messages: Sequence[Message | dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        adapter: str | None = None,
    ) -> CompletionResult:
        """Delegate to LLMTrait.complete()."""
        return self._trait.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            adapter=adapter,
        )

    def load_adapter(self, adapter_path: str) -> None:
        """Not supported through trait adapter."""
        raise NotImplementedError("Adapter loading not supported via trait")

    def unload_adapter(self) -> None:
        """Not supported through trait adapter."""
        raise NotImplementedError("Adapter unloading not supported via trait")
