# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the @verb decorator."""

from __future__ import annotations

from typing import Any

import pytest

from llm_gent.flow import Role, verb


ROLE = Role(name="judge", backend="openai", model="gpt-4o-mini")


class TestVerbDecorator:
    """@verb attaches role and preserves the underlying function."""

    def test_attaches_role(self) -> None:
        """@verb sets a .role attribute on the decorated function."""

        @verb(role=ROLE)
        async def judge_bio(ctx: Any, candidate: Any) -> str:
            """Return a stub verdict."""
            return "ok"

        assert judge_bio.role is ROLE  # type: ignore[attr-defined]

    def test_preserves_callable(self) -> None:
        """The decorated function is still callable and awaitable."""

        @verb(role=ROLE)
        async def echo(ctx: Any, value: str) -> str:
            """Return the input value."""
            return value

        import asyncio

        result = asyncio.run(echo(None, "hello"))
        assert result == "hello"

    def test_preserves_name(self) -> None:
        """The decorated function keeps its original __name__."""

        @verb(role=ROLE)
        async def my_verb(ctx: Any) -> None:
            """Do nothing."""

        assert my_verb.__name__ == "my_verb"

    @pytest.mark.parametrize(
        "role",
        [
            Role(name="a", backend="openai", model="gpt-4o-mini"),
            Role(name="b", backend="anthropic", model="claude-3-5"),
        ],
    )
    def test_different_roles_produce_independent_verbs(self, role: Role) -> None:
        """Each decoration binds independently to the role passed in."""

        @verb(role=role)
        async def scoped(ctx: Any) -> None:
            """Do nothing."""

        assert scoped.role is role  # type: ignore[attr-defined]
