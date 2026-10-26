# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Memory tools for agent kelt-backed fact operations.

These tools enable agents to store and retrieve facts via MemoryTrait.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import BaseTool, ToolResult


if TYPE_CHECKING:
    from ...traits.builtin.memory import MemoryTrait


class RememberTool(BaseTool):
    """Store a fact in long-term memory.

    Uses MemoryTrait to persist facts for future reference.
    Facts are stored with optional category for organization.

    Example:
        from llm_gent.core.traits.builtin.memory import MemoryTrait, MemoryConfig
        from llm_gent.core.tools.builtin.memory import RememberTool

        memory_trait = MemoryTrait(agent, MemoryConfig(...))
        tool = RememberTool(memory_trait)
        result = tool.execute(fact="User prefers Python", category="preferences")
    """

    name = "remember"
    description = (
        "Store a fact in long-term memory for future reference. Use for important discoveries."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact to remember",
            },
            "category": {
                "type": "string",
                "description": "Category for organization (e.g., 'preference', 'insight', 'discovery')",
            },
        },
        "required": ["fact"],
    }

    def __init__(self, memory_trait: MemoryTrait) -> None:
        """Initialize with MemoryTrait for persistence.

        Args:
            memory_trait: The MemoryTrait instance for storing facts.
        """
        self._memory_trait = memory_trait

    def execute(self, **kwargs: Any) -> ToolResult:
        """Store a fact in memory.

        Args:
            **kwargs: Must include 'fact', optionally 'category'.

        Returns:
            ToolResult with fact ID on success.
        """
        fact: str = kwargs["fact"]
        category: str = kwargs.get("category", "general")

        try:
            fact_id = self._memory_trait.remember(fact=fact, category=category, source="inferred")
            return ToolResult(success=True, output=f"Stored fact (id={fact_id}): {fact}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to store fact: {e}")


class RecallTool(BaseTool):
    """Search memory for relevant facts.

    Uses MemoryTrait to search stored facts. Supports semantic search
    when embedder is configured, otherwise returns recent facts.

    Example:
        from llm_gent.core.traits.builtin.memory import MemoryTrait, MemoryConfig
        from llm_gent.core.tools.builtin.memory import RecallTool

        memory_trait = MemoryTrait(agent, MemoryConfig(...))
        tool = RecallTool(memory_trait)
        result = tool.execute(query="user preferences", limit=5)
    """

    name = "recall"
    description = "Search memory for facts related to a query. Returns relevant stored information."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 5)",
            },
            "category": {
                "type": "string",
                "description": "Filter by category (optional)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory_trait: MemoryTrait) -> None:
        """Initialize with MemoryTrait for retrieval.

        Args:
            memory_trait: The MemoryTrait instance for retrieving facts.
        """
        self._memory_trait = memory_trait

    def execute(self, **kwargs: Any) -> ToolResult:
        """Search for facts matching the query.

        Args:
            **kwargs: Must include 'query', optionally 'limit' and 'category'.

        Returns:
            ToolResult with matching facts.
        """
        query: str = kwargs["query"]
        limit: int = kwargs.get("limit", 5)
        category: str | None = kwargs.get("category")

        try:
            categories = [category] if category else None
            if self._memory_trait.has_embedder:
                return self._recall_semantic(query, limit, categories)
            return self._recall_list(limit, categories)
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to recall facts: {e}")

    def _recall_semantic(self, query: str, limit: int, categories: list[str] | None) -> ToolResult:
        """Recall facts using semantic search."""
        scored_facts = self._memory_trait.recall(
            query=query, top_k=limit, min_similarity=0.3, categories=categories
        )
        if not scored_facts:
            return ToolResult(success=True, output="No matching facts found.")

        lines = [
            f"- [{sf.entity.category}] {sf.entity.content} (similarity: {sf.score:.2f})"
            for sf in scored_facts
        ]
        return ToolResult(
            success=True, output=f"Found {len(scored_facts)} fact(s):\n" + "\n".join(lines)
        )

    def _recall_list(self, limit: int, categories: list[str] | None) -> ToolResult:
        """Recall facts by listing recent entries (no embedder)."""
        facts = self._memory_trait.kelt.atomic.assertions.list(limit=limit)
        # Filter by category in memory if specified (list() doesn't support category filter)
        if categories:
            facts = [f for f in facts if f.category in categories]
        if not facts:
            return ToolResult(success=True, output="No facts stored yet.")

        lines = [f"- [{fact.category}] {fact.content}" for fact in facts]
        return ToolResult(success=True, output=f"Recent facts ({len(facts)}):\n" + "\n".join(lines))
