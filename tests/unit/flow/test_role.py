# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for llm_gent.flow.role."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from llm_gent.flow import Role


class TestRole:
    """Role dataclass shape and semantics."""

    def test_required_fields(self) -> None:
        """Role requires name, backend, model."""
        r = Role(name="judge", backend="openai", model="gpt-4o-mini")
        assert r.name == "judge"
        assert r.backend == "openai"
        assert r.model == "gpt-4o-mini"

    def test_defaults(self) -> None:
        """Optional fields have conservative defaults."""
        r = Role(name="x", backend="y", model="z")
        assert r.temperature == 0.7
        assert r.max_tokens == 4096
        assert r.style is None

    def test_frozen(self) -> None:
        """Role is immutable — attempting to mutate raises."""
        r = Role(name="x", backend="y", model="z")
        with pytest.raises(FrozenInstanceError):
            r.name = "changed"  # type: ignore[misc]

    def test_replace_returns_new_role(self) -> None:
        """dataclasses.replace produces a modified copy without mutating original."""
        r = Role(name="x", backend="y", model="z", temperature=0.1)
        r2 = replace(r, temperature=0.9)
        assert r.temperature == 0.1
        assert r2.temperature == 0.9
        assert r2.name == r.name

    def test_no_cost_fields(self) -> None:
        """Role deliberately carries no cost/billing fields — that lives outside gent."""
        r = Role(name="x", backend="y", model="z")
        for field in ("cost", "cost_bucket", "bucket", "billing"):
            assert not hasattr(r, field)
