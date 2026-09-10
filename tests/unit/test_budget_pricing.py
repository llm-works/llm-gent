# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for pricing: LLMOp, FixedOp, and PricingConfig lookup."""

from __future__ import annotations

import pytest

from llm_gent.core.budget import FixedOp, LLMOp, PricingConfig


pytestmark = pytest.mark.unit


class TestLLMOp:
    """Token-based cost math, with and without a cache-tier discount."""

    def test_uncached_only(self) -> None:
        op = LLMOp(name="m", input_per_mtok=1.0, output_per_mtok=5.0)
        assert op.cost(input_tokens=1_000_000, output_tokens=0) == 1.0
        assert op.cost(input_tokens=0, output_tokens=1_000_000) == 5.0

    def test_cached_defaults_to_input_rate_when_unset(self) -> None:
        op = LLMOp(name="m", input_per_mtok=1.0, output_per_mtok=5.0)
        assert op.cost(input_tokens=1000, output_tokens=0, cached_tokens=500) == pytest.approx(
            0.001
        )

    def test_cached_discount_applied_when_set(self) -> None:
        op = LLMOp(
            name="m",
            input_per_mtok=1.0,
            output_per_mtok=5.0,
            cached_input_per_mtok=0.1,
        )
        cost = op.cost(input_tokens=1_000_000, output_tokens=0, cached_tokens=500_000)
        assert cost == pytest.approx(0.55)

    def test_negative_tokens_clamped_to_zero(self) -> None:
        op = LLMOp(name="m", input_per_mtok=1.0, output_per_mtok=5.0)
        assert op.cost(input_tokens=-100, output_tokens=-100) == 0.0

    def test_cached_over_input_clamped_at_input(self) -> None:
        op = LLMOp(
            name="m",
            input_per_mtok=1.0,
            output_per_mtok=5.0,
            cached_input_per_mtok=0.1,
        )
        cost = op.cost(input_tokens=1000, output_tokens=0, cached_tokens=5000)
        assert cost == pytest.approx(0.0001)


class TestFixedOp:
    """Flat unit cost, scaled by count."""

    def test_single(self) -> None:
        assert FixedOp(name="search", unit_cost=0.001).cost() == pytest.approx(0.001)

    def test_scaled_by_count(self) -> None:
        assert FixedOp(name="search", unit_cost=0.001).cost(count=5) == pytest.approx(0.005)

    def test_negative_count_clamped(self) -> None:
        assert FixedOp(name="search", unit_cost=0.001).cost(count=-3) == 0.0


class TestPricingConfigLookup:
    """Exact / longest-prefix / "default" resolution on .get()."""

    def test_exact_match(self) -> None:
        cfg = PricingConfig(ops={"m": FixedOp(name="m", unit_cost=1.0)})
        assert cfg.get("m").name == "m"

    def test_longest_prefix_match(self) -> None:
        cfg = PricingConfig(
            ops={
                "some-model": FixedOp(name="some-model", unit_cost=1.0),
                "some-model-fast": FixedOp(name="some-model-fast", unit_cost=2.0),
            }
        )
        assert cfg.get("some-model-fast-preview-05").name == "some-model-fast"
        assert cfg.get("some-model-slow").name == "some-model"

    def test_default_key_used_when_no_match(self) -> None:
        cfg = PricingConfig(
            ops={
                "known": FixedOp(name="known", unit_cost=1.0),
                "default": FixedOp(name="default", unit_cost=99.0),
            }
        )
        assert cfg.get("unknown").name == "default"

    def test_default_not_used_when_prefix_matches(self) -> None:
        cfg = PricingConfig(
            ops={
                "known": FixedOp(name="known", unit_cost=1.0),
                "default": FixedOp(name="default", unit_cost=99.0),
            }
        )
        assert cfg.get("known-variant").name == "known"

    def test_none_when_no_match_and_no_default(self) -> None:
        cfg = PricingConfig(ops={"x": FixedOp(name="x", unit_cost=1.0)})
        assert cfg.get("nope") is None

    def test_empty_config(self) -> None:
        assert PricingConfig().get("anything") is None


class TestCustomOp:
    """Consumer-defined Ops satisfy the protocol without inheriting anything."""

    def test_custom_op_with_arbitrary_cost_signature(self) -> None:
        class MagnitudeOp:
            name = "magnitude"

            def cost(self, magnitude: float = 0.0) -> float:
                return 0.5 * magnitude

        cfg = PricingConfig(ops={"magnitude": MagnitudeOp()})
        assert cfg.get("magnitude").cost(magnitude=4.0) == 2.0
