# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for archetype decorators."""

from __future__ import annotations

from typing import Any

from llm_gent.flow import Role, extractor, grader, planner, synthesizer


ROLE = Role(name="x", backend="openai", model="gpt-4o-mini")


class TestArchetypes:
    """Each archetype attaches a role AND tags the function with an archetype name."""

    def test_planner_tags_and_binds_role(self) -> None:
        """@planner attaches the role and tags the function as a planner."""

        @planner(role=ROLE)
        async def my_plan(ctx: Any, task: str) -> None:
            """Test planner."""

        assert my_plan.role is ROLE  # type: ignore[attr-defined]
        assert my_plan.archetype == "planner"  # type: ignore[attr-defined]

    def test_extractor_tags(self) -> None:
        """@extractor tags the function as an extractor."""

        @extractor(role=ROLE)
        async def my_extract(ctx: Any, evidence: str) -> None:
            """Test extractor."""

        assert my_extract.archetype == "extractor"  # type: ignore[attr-defined]

    def test_grader_tags(self) -> None:
        """@grader tags the function as a grader."""

        @grader(role=ROLE)
        async def my_grade(ctx: Any, result: str) -> None:
            """Test grader."""

        assert my_grade.archetype == "grader"  # type: ignore[attr-defined]

    def test_synthesizer_tags(self) -> None:
        """@synthesizer tags the function as a synthesizer."""

        @synthesizer(role=ROLE)
        async def my_synth(ctx: Any) -> None:
            """Test synthesizer."""

        assert my_synth.archetype == "synthesizer"  # type: ignore[attr-defined]

    def test_all_four_archetypes_are_distinct(self) -> None:
        """The four archetypes produce distinguishable tags."""

        @planner(role=ROLE)
        async def a(ctx: Any) -> None:
            """Planner."""

        @extractor(role=ROLE)
        async def b(ctx: Any) -> None:
            """Extractor."""

        @grader(role=ROLE)
        async def c(ctx: Any) -> None:
            """Grader."""

        @synthesizer(role=ROLE)
        async def d(ctx: Any) -> None:
            """Synthesizer."""

        tags = {a.archetype, b.archetype, c.archetype, d.archetype}  # type: ignore[attr-defined]
        assert tags == {"planner", "extractor", "grader", "synthesizer"}
