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


class TestRoleParams:
    """``params`` mapping + ``.with_params`` builder for per-run factory inputs."""

    def test_params_defaults_to_empty_dict(self) -> None:
        """Fresh Role has empty ``params`` — no per-run inputs by default."""
        r = Role(name="x", backend="y", model="z")
        assert r.params == {}

    def test_with_params_returns_new_role_with_merged_kv(self) -> None:
        """``with_params(**kv)`` yields a new Role carrying the merged mapping."""
        r = Role(name="x", backend="y", model="z")
        r2 = r.with_params(max_iterations=20, campaign_id="c1")
        assert r.params == {}  # original untouched
        assert r2.params == {"max_iterations": 20, "campaign_id": "c1"}

    def test_with_params_later_overrides_earlier(self) -> None:
        """Chained ``with_params`` merges — later keys override earlier ones."""
        r = Role(name="x", backend="y", model="z")
        r2 = r.with_params(max_iterations=20, tag="a")
        r3 = r2.with_params(max_iterations=30)
        assert r3.params == {"max_iterations": 30, "tag": "a"}

    def test_params_participates_in_equality(self) -> None:
        """Roles differing only in ``params`` are NOT equal — required for cache correctness."""
        r_a = Role(name="x", backend="y", model="z").with_params(a=1)
        r_b = Role(name="x", backend="y", model="z").with_params(a=2)
        assert r_a != r_b

    def test_params_excluded_from_hash(self) -> None:
        """Hash is independent of ``params`` so unhashable values (trackers, state objects) are OK."""

        class _Unhashable:
            __hash__ = None  # type: ignore[assignment]

        r_empty = Role(name="x", backend="y", model="z")
        r_with_dict = r_empty.with_params(tracker=_Unhashable())
        # Both must hash without raising, and hashes must match — same non-params fields.
        assert hash(r_empty) == hash(r_with_dict)

    def test_role_with_unhashable_params_can_be_dict_key(self) -> None:
        """Concrete symptom of the hash-excluded design: Role usable as dict key."""

        class _Unhashable:
            __hash__ = None  # type: ignore[assignment]

        r = Role(name="x", backend="y", model="z").with_params(state=_Unhashable())
        # Would raise if params contributed to hash.
        cache = {r: "value"}
        assert cache[r] == "value"

    def test_hash_bucket_collision_still_distinguishes_via_eq(self) -> None:
        """Same-hash different-params roles miss cache lookups via __eq__ (dict semantics)."""
        r_a = Role(name="x", backend="y", model="z").with_params(a=1)
        r_b = Role(name="x", backend="y", model="z").with_params(a=2)
        # Same hash (params excluded), different eq — dict distinguishes correctly.
        assert hash(r_a) == hash(r_b)
        cache: dict[Role, str] = {r_a: "sa"}
        assert cache.get(r_b) is None
        assert cache.get(r_a) == "sa"

    def test_frozen_semantics_preserved(self) -> None:
        """Role stays frozen — reassigning ``params`` on an instance still raises."""
        from dataclasses import FrozenInstanceError

        r = Role(name="x", backend="y", model="z").with_params(a=1)
        with pytest.raises(FrozenInstanceError):
            r.params = {"b": 2}  # type: ignore[misc]
