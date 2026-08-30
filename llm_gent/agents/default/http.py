# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""HTTP protocol handler for default agent."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal, cast

from ...core.llm.types import Message


if TYPE_CHECKING:
    from ...core.agent import Agent
    from ...core.llm.types import CompletionResult


class HTTPHandler:
    """HTTP protocol implementation for default agent.

    Handles HTTP API methods (complete, remember, forget, recall, feedback)
    by delegating to the appropriate traits.

    This class knows about the HTTP protocol API surface but not about
    HTTP server infrastructure (that's HTTPTrait's job).
    """

    def __init__(self, agent: Agent) -> None:
        """Initialize handler.

        Args:
            agent: The agent instance this handler operates on.
        """
        self.agent = agent
        # Track response IDs for feedback validation (bounded FIFO to prevent memory leaks)
        self._response_ids: OrderedDict[str, None] = OrderedDict()
        self._max_response_ids = 1000

    def complete(self, query: str, system_prompt: str | None = None) -> CompletionResult:
        """Complete a query using LLMTrait.

        Args:
            query: The query to complete.
            system_prompt: Optional system prompt override.

        Returns:
            CompletionResult with response data.

        Raises:
            RuntimeError: If LLMTrait is not attached.
        """
        from ...core.traits.builtin.llm import LLMTrait

        llm_trait = self.agent.get_trait(LLMTrait)
        if llm_trait is None:
            raise RuntimeError("LLMTrait not attached")

        # Build messages from HTTP API format
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=query))

        result = llm_trait.complete(messages=messages)

        # Track response_id for feedback validation (bounded FIFO)
        if hasattr(result, "id") and result.id:
            self._response_ids[result.id] = None
            # Evict oldest entries if over limit
            while len(self._response_ids) > self._max_response_ids:
                self._response_ids.popitem(last=False)

        return result

    def remember(self, fact: str, category: str | None = None) -> int:
        """Store a fact using LearnTrait.

        Args:
            fact: The fact to remember.
            category: Optional category for the fact.

        Returns:
            The fact ID.

        Raises:
            RuntimeError: If LearnTrait is not attached.
        """
        from ...core.traits.builtin.learn import LearnTrait

        learn_trait = self.agent.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")
        # HTTP protocol allows None category, trait requires str
        return learn_trait.remember(fact=fact, category=category or "general")

    def forget(self, fact_id: int) -> None:
        """Remove a fact using LearnTrait.

        Args:
            fact_id: The ID of the fact to forget.

        Raises:
            RuntimeError: If LearnTrait is not attached.
        """
        from ...core.traits.builtin.learn import LearnTrait

        learn_trait = self.agent.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")
        learn_trait.forget(fact_id=fact_id)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5,
        categories: list[str] | None = None,
    ) -> list[Any]:
        """Recall facts similar to query using LearnTrait.

        Args:
            query: The query to search for.
            top_k: Maximum number of results.
            min_similarity: Minimum similarity threshold.
            categories: Optional category filter.

        Returns:
            List of scored facts.

        Raises:
            RuntimeError: If LearnTrait is not attached.
        """
        from ...core.traits.builtin.learn import LearnTrait

        learn_trait = self.agent.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")
        return learn_trait.recall(
            query=query, top_k=top_k, min_similarity=min_similarity, categories=categories
        )

    def feedback(self, response_id: str, signal: str, correction: str | None = None) -> None:
        """Record feedback using LearnTrait.

        Args:
            response_id: ID of the response being rated.
            signal: Feedback signal ("positive" or "negative").
            correction: Optional correction text.

        Raises:
            ValueError: If response_id is not recognized or signal is invalid.
            RuntimeError: If LearnTrait is not attached.
        """
        if response_id not in self._response_ids:
            raise ValueError(f"Unknown response_id: {response_id}")

        if signal not in ("positive", "negative"):
            raise ValueError(f"Invalid signal: {signal}. Must be 'positive' or 'negative'")

        from ...core.traits.builtin.learn import LearnTrait

        learn_trait = self.agent.get_trait(LearnTrait)
        if learn_trait is None:
            raise RuntimeError("LearnTrait not attached")

        # Convert HTTP API format to LearnTrait.record_feedback format
        context: dict[str, Any] = {"response_id": response_id}
        if correction:
            context["correction"] = correction

        learn_trait.record_feedback(
            content=correction or "",
            signal=cast(Literal["positive", "negative"], signal),
            context=context,
        )
