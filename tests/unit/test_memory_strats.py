# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for memory recall strategies."""

from unittest.mock import MagicMock

import pytest

from llm_gent.core.memory.strats import (
    format_solutions_context,
    recall_chronological,
    recall_semantic,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# recall_chronological
# ---------------------------------------------------------------------------


class TestRecallChronological:
    def test_success(self):
        trait = MagicMock()
        trait.kelt.atomic.solutions.list_by_agent.return_value = ["sol1", "sol2"]

        result = recall_chronological(trait, "agent-1", limit=5)

        assert result == ["sol1", "sol2"]
        trait.kelt.atomic.solutions.list_by_agent.assert_called_once_with(
            agent_name="agent-1", limit=5, active_only=True
        )

    def test_exception_returns_empty(self):
        trait = MagicMock()
        trait.kelt.atomic.solutions.list_by_agent.side_effect = RuntimeError("db down")
        trait._lg = MagicMock()

        result = recall_chronological(trait, "agent-1")

        assert result == []
        trait._lg.debug.assert_called_once()

    def test_exception_no_logger(self):
        """Graceful degradation when trait has no _lg attribute."""
        trait = MagicMock(spec=[])  # no attributes
        trait.kelt = MagicMock()
        trait.kelt.atomic.solutions.list_by_agent.side_effect = RuntimeError("fail")

        result = recall_chronological(trait, "agent-1")
        assert result == []

    def test_default_limit(self):
        trait = MagicMock()
        trait.kelt.atomic.solutions.list_by_agent.return_value = []

        recall_chronological(trait, "agent-1")

        trait.kelt.atomic.solutions.list_by_agent.assert_called_once_with(
            agent_name="agent-1", limit=5, active_only=True
        )


# ---------------------------------------------------------------------------
# recall_semantic
# ---------------------------------------------------------------------------


class TestRecallSemantic:
    def test_success(self):
        trait = MagicMock()
        trait.kelt.atomic.solutions.search.return_value = ["r1", "r2"]

        result = recall_semantic(trait, "how to sort", limit=3)

        assert result == ["r1", "r2"]
        trait.kelt.atomic.solutions.search.assert_called_once_with(
            query="how to sort", limit=3, active_only=True
        )

    def test_agent_filter(self):
        sol1 = MagicMock()
        sol1.solution_details.agent_name = "agent-1"
        sol2 = MagicMock()
        sol2.solution_details.agent_name = "agent-2"

        trait = MagicMock()
        trait.kelt.atomic.solutions.search.return_value = [sol1, sol2]

        result = recall_semantic(trait, "query", agent_name="agent-1")

        assert result == [sol1]

    def test_agent_filter_respects_limit(self):
        sols = []
        for _i in range(10):
            s = MagicMock()
            s.solution_details.agent_name = "agent-1"
            sols.append(s)

        trait = MagicMock()
        trait.kelt.atomic.solutions.search.return_value = sols

        result = recall_semantic(trait, "query", limit=3, agent_name="agent-1")
        assert len(result) == 3

    def test_fallback_on_error_with_agent(self):
        """Falls back to chronological when search fails and agent_name given."""
        trait = MagicMock()
        trait.kelt.atomic.solutions.search.side_effect = RuntimeError("search down")
        trait.kelt.atomic.solutions.list_by_agent.return_value = ["fallback"]
        trait._lg = MagicMock()

        result = recall_semantic(trait, "query", agent_name="agent-1")

        assert result == ["fallback"]

    def test_fallback_on_error_no_agent(self):
        """Returns empty when search fails and no agent_name."""
        trait = MagicMock()
        trait.kelt.atomic.solutions.search.side_effect = RuntimeError("fail")
        trait._lg = MagicMock()

        result = recall_semantic(trait, "query")

        assert result == []

    def test_empty_results(self):
        trait = MagicMock()
        trait.kelt.atomic.solutions.search.return_value = []

        result = recall_semantic(trait, "query")
        assert result == []

    def test_no_logger(self):
        trait = MagicMock(spec=[])
        trait.kelt = MagicMock()
        trait.kelt.atomic.solutions.search.side_effect = RuntimeError("fail")

        result = recall_semantic(trait, "query")
        assert result == []


# ---------------------------------------------------------------------------
# format_solutions_context
# ---------------------------------------------------------------------------


class TestFormatSolutionsContext:
    def test_empty(self):
        assert format_solutions_context([]) == ""

    def test_format_with_answer_text(self):
        sol = MagicMock()
        sol.solution_details.answer_text = "The answer is 42"

        result = format_solutions_context([sol])

        assert "Previously Completed Tasks" in result
        assert "The answer is 42" in result
        assert "1." in result

    def test_format_fallback_to_answer_dict(self):
        sol = MagicMock()
        sol.solution_details.answer_text = None
        sol.solution_details.answer = {"output": "dict answer"}

        result = format_solutions_context([sol])
        assert "dict answer" in result

    def test_no_solution_details(self):
        sol = MagicMock()
        sol.solution_details = None

        result = format_solutions_context([sol])
        assert result == ""

    def test_truncation(self):
        sol = MagicMock()
        sol.solution_details.answer_text = "x" * 500

        result = format_solutions_context([sol])

        # Content should be truncated to 200 chars
        lines = result.split("\n")
        content_line = [line for line in lines if line.startswith("1.")][0]
        assert len(content_line) <= 210  # "1. " + 200 chars

    def test_empty_output_skipped(self):
        sol1 = MagicMock()
        sol1.solution_details.answer_text = ""
        sol1.solution_details.answer = {"output": ""}

        sol2 = MagicMock()
        sol2.solution_details.answer_text = "real answer"

        result = format_solutions_context([sol1, sol2])

        # sol1 should be skipped, sol2 should still be numbered
        assert "real answer" in result

    def test_multiple_solutions(self):
        sols = []
        for i in range(3):
            s = MagicMock()
            s.solution_details.answer_text = f"Answer {i}"
            sols.append(s)

        result = format_solutions_context(sols)
        assert "1." in result
        assert "2." in result
        assert "3." in result
