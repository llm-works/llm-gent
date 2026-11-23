# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""HTTP protocol handler for default agent."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal, cast

from ...core.llm.caller import Caller
from ...core.llm.types import CompletionResult


if TYPE_CHECKING:
    from ...core.agent import Agent


class HTTPHandler:
    """HTTP protocol implementation for default agent.

    Handles HTTP API methods (complete, remember, forget, recall, feedback)
    by delegating to the appropriate traits.

    This class knows about the HTTP protocol API surface but not about
    HTTP server infrastructure (that's HTTPTrait's job).
    """

    def __init__(self, agent: Agent, caller: Caller) -> None:
        """Initialize handler.

        Args:
            agent: The agent instance this handler operates on.
            caller: The model calls ``complete`` makes (over the platform's router).
        """
        self.agent = agent
        self._caller = caller
        # Track response IDs for feedback validation (bounded FIFO to prevent memory leaks)
        self._response_ids: OrderedDict[str, None] = OrderedDict()
        self._max_response_ids = 1000

    def complete(self, query: str, system_prompt: str | None = None) -> CompletionResult:
        """Complete ``query`` with one model call, as the agent.

        The agent's directive, when it has one, leads the system message;
        ``system_prompt`` follows it.

        Args:
            query: The query to complete.
            system_prompt: Optional system prompt added after the directive.

        Returns:
            CompletionResult with response data.
        """
        messages: list[dict[str, Any]] = []
        system = self._system_message(system_prompt)
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": query})

        started = time.monotonic()
        call = self._caller.chat(messages=messages)
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = call.usage or {}
        result = CompletionResult(
            id=str(uuid.uuid4()),
            content=call.content,
            model=call.model,
            tokens_used=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            tool_calls=call.tool_calls,
            adapter=call.adapter,
        )

        # Track response_id for feedback validation (bounded FIFO)
        self._response_ids[result.id] = None
        while len(self._response_ids) > self._max_response_ids:
            self._response_ids.popitem(last=False)
        return result

    def _system_message(self, system_prompt: str | None) -> str | None:
        """The agent's directive followed by ``system_prompt``; ``None`` when neither is set."""
        from ...core.traits.builtin.directive import DirectiveTrait

        directive = self.agent.get_trait(DirectiveTrait)
        parts = [directive.directive.prompt] if directive is not None else []
        if system_prompt:
            parts.append(system_prompt)
        return "\n\n".join(parts) or None

    def remember(self, fact: str, category: str | None = None) -> int:
        """Store a fact using MemoryTrait.

        Args:
            fact: The fact to remember.
            category: Optional category for the fact.

        Returns:
            The fact ID.

        Raises:
            RuntimeError: If MemoryTrait is not attached.
        """
        from ...core.traits.builtin.memory import MemoryTrait

        memory_trait = self.agent.get_trait(MemoryTrait)
        if memory_trait is None:
            raise RuntimeError("MemoryTrait not attached")
        # HTTP protocol allows None category, trait requires str
        return memory_trait.remember(fact=fact, category=category or "general")

    def forget(self, fact_id: int) -> None:
        """Remove a fact using MemoryTrait.

        Args:
            fact_id: The ID of the fact to forget.

        Raises:
            RuntimeError: If MemoryTrait is not attached.
        """
        from ...core.traits.builtin.memory import MemoryTrait

        memory_trait = self.agent.get_trait(MemoryTrait)
        if memory_trait is None:
            raise RuntimeError("MemoryTrait not attached")
        memory_trait.forget(fact_id=fact_id)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5,
        categories: list[str] | None = None,
    ) -> list[Any]:
        """Recall facts similar to query using MemoryTrait.

        Args:
            query: The query to search for.
            top_k: Maximum number of results.
            min_similarity: Minimum similarity threshold.
            categories: Optional category filter.

        Returns:
            List of scored facts.

        Raises:
            RuntimeError: If MemoryTrait is not attached.
        """
        from ...core.traits.builtin.memory import MemoryTrait

        memory_trait = self.agent.get_trait(MemoryTrait)
        if memory_trait is None:
            raise RuntimeError("MemoryTrait not attached")
        return memory_trait.recall(
            query=query, top_k=top_k, min_similarity=min_similarity, categories=categories
        )

    def feedback(self, response_id: str, signal: str, correction: str | None = None) -> None:
        """Record feedback using MemoryTrait.

        Args:
            response_id: ID of the response being rated.
            signal: Feedback signal ("positive" or "negative").
            correction: Optional correction text.

        Raises:
            ValueError: If response_id is not recognized or signal is invalid.
            RuntimeError: If MemoryTrait is not attached.
        """
        if response_id not in self._response_ids:
            raise ValueError(f"Unknown response_id: {response_id}")

        if signal not in ("positive", "negative"):
            raise ValueError(f"Invalid signal: {signal}. Must be 'positive' or 'negative'")

        from ...core.traits.builtin.memory import MemoryTrait

        memory_trait = self.agent.get_trait(MemoryTrait)
        if memory_trait is None:
            raise RuntimeError("MemoryTrait not attached")

        # Convert HTTP API format to MemoryTrait.record_feedback format
        context: dict[str, Any] = {"response_id": response_id}
        if correction:
            context["correction"] = correction

        memory_trait.record_feedback(
            content=correction or "",
            signal=cast(Literal["positive", "negative"], signal),
            context=context,
        )
