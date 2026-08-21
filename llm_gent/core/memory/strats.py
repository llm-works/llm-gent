"""Memory strategies for agent context retrieval.

Provides functions for recalling past solutions using different strategies:
- Chronological: Most recent N solutions (for repetitive tasks)
- Semantic: Semantically similar solutions via search (for varied tasks)
"""

from __future__ import annotations

from typing import Any


def recall_chronological(memory_trait: Any, agent_name: str, limit: int = 5) -> list[Any]:
    """Recall last N solutions chronologically.

    Use for repetitive tasks where recency matters (e.g., scheduled execution).

    Args:
        memory_trait: MemoryTrait instance providing access to solutions client.
        agent_name: Agent name to filter solutions by.
        limit: Maximum number of recent solutions to recall.

    Returns:
        List of solution facts, ordered by most recent first.
    """
    try:
        solutions = memory_trait.kelt.atomic.solutions.list_by_agent(
            agent_name=agent_name,
            limit=limit,
            active_only=True,
        )
        return solutions  # type: ignore[no-any-return]
    except Exception as e:
        # Log failure for debugging, but don't fail (graceful degradation)
        if hasattr(memory_trait, "_lg"):
            memory_trait._lg.debug(
                "recall_chronological failed", extra={"agent_name": agent_name, "exception": e}
            )
        return []


def recall_semantic(
    memory_trait: Any, query: str, limit: int = 5, agent_name: str | None = None
) -> list[Any]:
    """Recall semantically similar solutions using search.

    Use for varied tasks where relevance matters (e.g., ad-hoc questions).
    Falls back to chronological recall if search fails.

    Args:
        memory_trait: MemoryTrait instance providing access to solutions client.
        query: Query string to search for similar solutions.
        limit: Maximum number of solutions to recall.
        agent_name: Optional agent name to filter by (if search supports it).

    Returns:
        List of solution facts, ordered by relevance.
    """
    try:
        # Search uses substring matching on problem text
        solutions = memory_trait.kelt.atomic.solutions.search(
            query=query,
            limit=limit,
            active_only=True,
        )

        # Filter by agent_name if specified (search doesn't have agent filter)
        if agent_name and solutions:
            solutions = [
                s
                for s in solutions
                if s.solution_details and s.solution_details.agent_name == agent_name
            ][:limit]

        return solutions  # type: ignore[no-any-return]
    except Exception as e:
        # Log failure for debugging, then fall back to chronological
        if hasattr(memory_trait, "_lg"):
            memory_trait._lg.debug(
                "recall_semantic failed", extra={"query": query[:100], "exception": e}
            )
        # Fall back to chronological on any error
        if agent_name:
            return recall_chronological(memory_trait, agent_name, limit)
        return []


def format_solutions_context(solutions: list[Any]) -> str:
    """Format solutions into clean context string.

    Extracts just the output, removes metadata (tool calls, system messages).
    Generic formatting that works for any agent.

    Args:
        solutions: List of solution facts from recall functions.

    Returns:
        Formatted string ready for prompt injection, or empty string if no solutions.
    """
    if not solutions:
        return ""

    lines = ["## Previously Completed Tasks\n"]
    for i, sol in enumerate(solutions, 1):
        if not sol.solution_details:
            continue

        # Use answer_text if available (summarized output)
        output = sol.solution_details.answer_text
        if not output:
            # Fall back to raw output from answer dict
            output = sol.solution_details.answer.get("output", "")

        if output:
            # Clean up output - just take first 200 chars to keep context concise
            cleaned = output.strip()[:200]
            lines.append(f"{i}. {cleaned}")

    return "\n".join(lines) if len(lines) > 1 else ""
