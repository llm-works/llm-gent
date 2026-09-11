# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the batteries-included LLM pricing layer under core/budget/llm/.

Covers:

- ``LLMPricingProvider`` — constructor validation, lookup (publisher-strip,
  boundary-safe prefix, longest-prefix, default fallback), compute()
  precedence, per-provider dispatch (Anthropic / OpenAI-compat / Gemini),
  provider detection, on_missing policy, override validation.
- ``MissingModelCostError`` — carries op_name + configured (sorted, no
  "default").
- ``ProviderCostExtractor`` — xAI cost_in_usd_ticks; cached_tokens from
  both OpenAI-compat and Anthropic shapes.
- ``load_pricing_config`` — dict → PricingConfig, required fields,
  negative / non-finite rejection, optional cached rate.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from llm_infer.client import Provider

from llm_gent.core.budget import (
    FixedOp,
    LLMOp,
    LLMPricingProvider,
    MissingModelCostError,
    OnMissing,
    PricingConfig,
    ProviderCostExtractor,
    load_pricing_config,
)


pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# fixtures / helpers
# -----------------------------------------------------------------------------


def _op(name: str = "some-model", cached: float | None = None) -> LLMOp:
    return LLMOp(
        name=name,
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        cached_input_per_mtok=cached,
    )


def _config(**ops: Any) -> PricingConfig:
    return PricingConfig(ops=dict(ops))


def _resp(provider: str | None, raw: dict[str, Any] | None) -> SimpleNamespace:
    return SimpleNamespace(provider=provider, raw=raw)


# -----------------------------------------------------------------------------
# OnMissing enum
# -----------------------------------------------------------------------------


class TestOnMissingEnum:
    def test_str_values(self) -> None:
        assert OnMissing.RAISE == "raise"
        assert OnMissing.ZERO == "zero"

    def test_is_str_enum(self) -> None:
        assert isinstance(OnMissing.RAISE, str)


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self) -> None:
        p = LLMPricingProvider(_config())
        assert isinstance(p, LLMPricingProvider)

    def test_rejects_str_on_missing(self) -> None:
        with pytest.raises(TypeError, match="OnMissing member"):
            LLMPricingProvider(_config(), on_missing="raise")  # type: ignore[arg-type]

    def test_rejects_negative_cache_write_multiplier(self) -> None:
        with pytest.raises(ValueError, match="anthropic_cache_write_multiplier"):
            LLMPricingProvider(_config(), anthropic_cache_write_multiplier=-1.0)

    def test_rejects_nonfinite_cache_read_multiplier(self) -> None:
        with pytest.raises(ValueError, match="anthropic_cache_read_multiplier"):
            LLMPricingProvider(_config(), anthropic_cache_read_multiplier=math.inf)

    def test_rejects_bool_multiplier(self) -> None:
        with pytest.raises(TypeError):
            LLMPricingProvider(_config(), anthropic_cache_write_multiplier=True)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Lookup: publisher strip, boundary-safe prefix, longest-prefix, default
# -----------------------------------------------------------------------------


class TestLookup:
    def test_exact_match(self) -> None:
        p = LLMPricingProvider(_config(**{"gpt-4o": _op("gpt-4o")}))
        assert p.compute("gpt-4o", input_tokens=1_000_000) == pytest.approx(1.0)

    def test_publisher_prefix_stripped(self) -> None:
        p = LLMPricingProvider(_config(**{"gemini-2.5-flash": _op("gemini-2.5-flash")}))
        assert p.compute("google/gemini-2.5-flash", input_tokens=1_000_000) == pytest.approx(1.0)

    def test_boundary_safe_prefix_matches_boundary(self) -> None:
        p = LLMPricingProvider(_config(**{"grok-3": _op("grok-3")}))
        assert p.compute("grok-3-fast", input_tokens=1_000_000) == pytest.approx(1.0)

    def test_boundary_safe_prefix_rejects_no_boundary(self) -> None:
        p = LLMPricingProvider(_config(**{"grok-3": _op("grok-3")}))
        with pytest.raises(MissingModelCostError):
            p.compute("grok-30", input_tokens=1_000_000)

    def test_longest_prefix_wins(self) -> None:
        p = LLMPricingProvider(
            _config(
                **{
                    "claude": LLMOp(name="claude", input_per_mtok=1.0, output_per_mtok=5.0),
                    "claude-haiku": LLMOp(
                        name="claude-haiku", input_per_mtok=10.0, output_per_mtok=50.0
                    ),
                }
            )
        )
        assert p.compute("claude-haiku-4-5", input_tokens=1_000_000) == pytest.approx(10.0)

    def test_default_fallback(self) -> None:
        p = LLMPricingProvider(
            _config(
                default=LLMOp(name="default", input_per_mtok=100.0, output_per_mtok=500.0),
            )
        )
        assert p.compute("nonexistent-model", input_tokens=1_000_000) == pytest.approx(100.0)

    def test_default_excluded_from_prefix_match(self) -> None:
        p = LLMPricingProvider(
            _config(
                default=LLMOp(name="default", input_per_mtok=100.0, output_per_mtok=500.0),
                **{"default-model": _op("default-model")},
            )
        )
        assert p.compute("default-model", input_tokens=1_000_000) == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# compute() precedence
# -----------------------------------------------------------------------------


class TestPrecedence:
    def test_provider_cost_override_wins(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        assert p.compute("m", provider_cost=42.5, input_tokens=999_999_999) == 42.5

    def test_provider_cost_rejects_nan(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        with pytest.raises(ValueError, match="provider_cost"):
            p.compute("m", provider_cost=math.nan)

    def test_provider_cost_rejects_negative(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        with pytest.raises(ValueError, match="provider_cost"):
            p.compute("m", provider_cost=-0.01)

    def test_provider_cost_rejects_bool(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        with pytest.raises(TypeError, match="provider_cost"):
            p.compute("m", provider_cost=True)

    def test_response_triggers_dispatch(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        resp = _resp("openai", {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
        cost = p.compute("m", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)

    def test_rate_card_lookup_when_no_response(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        assert p.compute("m", input_tokens=1_000_000, output_tokens=0) == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Anthropic dispatch
# -----------------------------------------------------------------------------


class TestAnthropicDispatch:
    def test_cache_read_from_wire_at_default_multiplier(self) -> None:
        # cached_input_per_mtok unset → cache_read bills at 0.1 * input rate.
        p = LLMPricingProvider(_config(**{"claude": _op("claude")}))
        raw = {"usage": {"cache_read_input_tokens": 1_000_000}}
        resp = _resp("anthropic", raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(0.1)

    def test_cache_read_at_configured_cached_rate(self) -> None:
        p = LLMPricingProvider(_config(**{"claude": _op("claude", cached=0.25)}))
        raw = {"usage": {"cache_read_input_tokens": 1_000_000}}
        resp = _resp("anthropic", raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(0.25)

    def test_cache_write_at_default_multiplier(self) -> None:
        # 1.25 * input rate * cache_write_tokens
        p = LLMPricingProvider(_config(**{"claude": _op("claude")}))
        raw = {"usage": {"cache_creation_input_tokens": 1_000_000}}
        resp = _resp("anthropic", raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(1.25)

    def test_cache_write_configurable_multiplier(self) -> None:
        p = LLMPricingProvider(
            _config(**{"claude": _op("claude")}),
            anthropic_cache_write_multiplier=2.0,
        )
        raw = {"usage": {"cache_creation_input_tokens": 1_000_000}}
        resp = _resp("anthropic", raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(2.0)

    def test_cache_read_configurable_multiplier(self) -> None:
        p = LLMPricingProvider(
            _config(**{"claude": _op("claude")}),
            anthropic_cache_read_multiplier=0.05,
        )
        raw = {"usage": {"cache_read_input_tokens": 1_000_000}}
        resp = _resp("anthropic", raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(0.05)

    def test_uncached_input_separate_from_cache(self) -> None:
        # Anthropic prompt_tokens is uncached-only on the wire.
        p = LLMPricingProvider(_config(**{"claude": _op("claude", cached=0.1)}))
        raw = {
            "usage": {
                "cache_read_input_tokens": 500_000,
                "cache_creation_input_tokens": 100_000,
            }
        }
        resp = _resp("anthropic", raw)
        cost = p.compute("claude", response=resp, input_tokens=1_000_000, output_tokens=200_000)
        # 1.0 (input) + 0.05 (cache_read 500k @ 0.1) + 0.125 (cache_write 100k @ 1.25) + 1.0 (output)
        assert cost == pytest.approx(2.175)

    def test_explicit_cached_overrides_wire(self) -> None:
        p = LLMPricingProvider(_config(**{"claude": _op("claude", cached=0.1)}))
        raw = {"usage": {"cache_read_input_tokens": 999_999}}
        resp = _resp("anthropic", raw)
        cost = p.compute(
            "claude", response=resp, input_tokens=0, output_tokens=0, cached_tokens=1_000_000
        )
        assert cost == pytest.approx(0.1)

    def test_explicit_cache_creation_overrides_wire(self) -> None:
        # Symmetric with cached_tokens override — invoice reconciliation
        # or test fixtures can inject cache_creation_tokens without wire.
        p = LLMPricingProvider(_config(**{"claude": _op("claude")}))
        raw = {"usage": {"cache_creation_input_tokens": 999_999}}
        resp = _resp("anthropic", raw)
        cost = p.compute(
            "claude",
            response=resp,
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1_000_000,
        )
        assert cost == pytest.approx(1.25)  # 1M @ input rate * 1.25 mult


# -----------------------------------------------------------------------------
# OpenAI-compat dispatch
# -----------------------------------------------------------------------------


class TestOpenAICompatDispatch:
    def test_input_output_math(self) -> None:
        p = LLMPricingProvider(_config(**{"gpt": _op("gpt")}))
        resp = _resp("openai", {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
        cost = p.compute("gpt", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)

    def test_cached_from_prompt_tokens_details(self) -> None:
        p = LLMPricingProvider(_config(**{"gpt": _op("gpt", cached=0.1)}))
        raw = {
            "usage": {
                "prompt_tokens": 1_000_000,
                "prompt_tokens_details": {"cached_tokens": 500_000},
            }
        }
        resp = _resp("openai", raw)
        cost = p.compute("gpt", response=resp, input_tokens=1_000_000, output_tokens=0)
        # 500k uncached @ 1.0 + 500k cached @ 0.1 = 0.5 + 0.05 = 0.55
        assert cost == pytest.approx(0.55)

    def test_explicit_cached_overrides_wire(self) -> None:
        p = LLMPricingProvider(_config(**{"gpt": _op("gpt", cached=0.1)}))
        raw = {
            "usage": {
                "prompt_tokens": 1_000_000,
                "prompt_tokens_details": {"cached_tokens": 999_999},
            }
        }
        resp = _resp("openai", raw)
        cost = p.compute(
            "gpt", response=resp, input_tokens=1_000_000, output_tokens=0, cached_tokens=500_000
        )
        assert cost == pytest.approx(0.55)


# -----------------------------------------------------------------------------
# Gemini dispatch (base math only; no cache, no tier)
# -----------------------------------------------------------------------------


class TestGeminiDispatch:
    def test_base_math(self) -> None:
        p = LLMPricingProvider(_config(**{"gemini-2.5-flash": _op("gemini-2.5-flash")}))
        resp = _resp("gemini", {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
        cost = p.compute("gemini-2.5-flash", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)

    def test_no_cache_math_applied(self) -> None:
        # Even if raw carries Anthropic-style cache fields, Gemini branch
        # ignores them — priority-tier / caching is consumer concern.
        p = LLMPricingProvider(_config(**{"gemini": _op("gemini", cached=0.1)}))
        raw = {"usage": {"cache_read_input_tokens": 999_999}}
        resp = _resp("google", raw)
        cost = p.compute("gemini", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)  # cache field ignored

    def test_vertex_provider_tag_routes_to_gemini(self) -> None:
        p = LLMPricingProvider(_config(**{"gemini": _op("gemini")}))
        resp = _resp("vertex", {"usage": {}})
        cost = p.compute("gemini", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Float token counts (some JSON decoders): coerced, not silent-zeroed
# -----------------------------------------------------------------------------


class TestFloatTokenCounts:
    def test_float_input_tokens_coerced_not_zeroed(self) -> None:
        # 1500.0 from a JSON decoder that upcast the int must NOT bill $0.
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        resp = _resp("openai", {"usage": {}})
        cost = p.compute("m", response=resp, input_tokens=1_500_000.0, output_tokens=0)
        assert cost == pytest.approx(1.5)

    def test_float_from_wire_cache_read_coerced(self) -> None:
        # Wire-side value arriving as float (uncommon but possible from
        # loose JSON decoders) still bills correctly.
        p = LLMPricingProvider(_config(**{"claude": _op("claude", cached=0.1)}))
        raw = {"usage": {"cache_read_input_tokens": 1_000_000.0}}
        resp = _resp("anthropic", raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(0.1)

    def test_nan_float_still_zeroed(self) -> None:
        # NaN must not silently propagate as a huge int() cast — collapse to 0.
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        resp = _resp("openai", {"usage": {}})
        cost = p.compute("m", response=resp, input_tokens=math.nan, output_tokens=0)
        assert cost == 0.0

    def test_non_numeric_string_still_zeroed(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        resp = _resp("openai", {"usage": {}})
        cost = p.compute("m", response=resp, input_tokens="oops", output_tokens=0)
        assert cost == 0.0


# -----------------------------------------------------------------------------
# Provider detection
# -----------------------------------------------------------------------------


class TestProviderDetection:
    def test_response_provider_preferred(self) -> None:
        # Explicit anthropic tag with no Anthropic cache fields — still routes anthropic.
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        resp = _resp("anthropic", {"usage": {}})
        cost = p.compute("m", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)  # Anthropic uncached input rate

    def test_wire_field_sniff_detects_anthropic(self) -> None:
        # No provider tag — sniff detects Anthropic via cache_creation_input_tokens.
        p = LLMPricingProvider(_config(**{"claude": _op("claude")}))
        raw = {"usage": {"cache_creation_input_tokens": 1_000_000}}
        resp = _resp(None, raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(1.25)  # Anthropic cache_write path

    def test_unknown_provider_falls_to_openai_compat(self) -> None:
        p = LLMPricingProvider(_config(**{"m": _op("m")}))
        resp = _resp(None, {"usage": {"prompt_tokens": 1_000_000}})
        cost = p.compute("m", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)

    def test_provider_enum_member_routes_to_anthropic(self) -> None:
        # llm-infer's ChatResponse.provider is typed as str but stores
        # Provider enum values by convention. Passing the enum member
        # directly must dispatch identically to the string value.
        p = LLMPricingProvider(_config(**{"claude": _op("claude")}))
        raw = {"usage": {"cache_creation_input_tokens": 1_000_000}}
        resp = _resp(Provider.ANTHROPIC, raw)
        cost = p.compute("claude", response=resp, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(1.25)  # Anthropic cache_write path

    def test_provider_enum_google_routes_to_gemini(self) -> None:
        p = LLMPricingProvider(_config(**{"gemini": _op("gemini")}))
        resp = _resp(Provider.GOOGLE, {"usage": {}})
        cost = p.compute("gemini", response=resp, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# on_missing policy
# -----------------------------------------------------------------------------


class TestOnMissing:
    def test_raise_default(self) -> None:
        p = LLMPricingProvider(_config(**{"other": _op("other")}))
        with pytest.raises(MissingModelCostError) as excinfo:
            p.compute("unknown-model", input_tokens=100)
        err = excinfo.value
        assert err.op_name == "unknown-model"
        assert err.configured == ["other"]

    def test_zero_returns_zero(self) -> None:
        p = LLMPricingProvider(_config(**{"other": _op("other")}), on_missing=OnMissing.ZERO)
        assert p.compute("unknown-model", input_tokens=100) == 0.0

    def test_configured_excludes_default_key(self) -> None:
        # Configured list on the error is sorted and excludes any "default"
        # key. A rate card that includes "default" catches misses via the
        # default fallback (no miss fires), so this asserts the shape via
        # a rate card without the default key.
        p = LLMPricingProvider(_config(**{"b": _op("b"), "a": _op("a")}))
        with pytest.raises(MissingModelCostError) as excinfo:
            p.compute("z", input_tokens=1)
        assert excinfo.value.configured == ["a", "b"]


# -----------------------------------------------------------------------------
# non-LLMOp entries pass through op.cost() on response= path
# -----------------------------------------------------------------------------


class TestNonLLMOp:
    def test_fixed_op_uses_own_cost_on_response_path(self) -> None:
        p = LLMPricingProvider(
            _config(**{"web_search": FixedOp(name="web_search", unit_cost=0.001)})
        )
        resp = _resp("openai", {"usage": {}})
        cost = p.compute("web_search", response=resp, count=3)
        assert cost == pytest.approx(0.003)


# -----------------------------------------------------------------------------
# ProviderCostExtractor
# -----------------------------------------------------------------------------


class TestProviderCostExtractor:
    def test_xai_cost_in_usd_ticks(self) -> None:
        raw = {"usage": {"cost_in_usd_ticks": 12_345_678_900}}
        assert ProviderCostExtractor.extract(raw) == pytest.approx(1.23456789)

    def test_extract_none_for_none(self) -> None:
        assert ProviderCostExtractor.extract(None) is None

    def test_extract_none_when_no_cost_field(self) -> None:
        assert ProviderCostExtractor.extract({"usage": {"prompt_tokens": 100}}) is None

    def test_extract_none_when_usage_not_dict(self) -> None:
        assert ProviderCostExtractor.extract({"usage": "not-a-dict"}) is None

    def test_extract_none_when_ticks_not_number(self) -> None:
        raw = {"usage": {"cost_in_usd_ticks": "not-a-number"}}
        assert ProviderCostExtractor.extract(raw) is None

    def test_extract_none_for_negative(self) -> None:
        raw = {"usage": {"cost_in_usd_ticks": -1}}
        assert ProviderCostExtractor.extract(raw) is None

    def test_extract_none_when_ticks_bool(self) -> None:
        raw = {"usage": {"cost_in_usd_ticks": True}}
        assert ProviderCostExtractor.extract(raw) is None

    def test_cached_from_prompt_tokens_details(self) -> None:
        raw = {"usage": {"prompt_tokens_details": {"cached_tokens": 500}}}
        assert ProviderCostExtractor.extract_cached_tokens(raw) == 500

    def test_cached_from_anthropic_cache_read(self) -> None:
        raw = {"usage": {"cache_read_input_tokens": 750}}
        assert ProviderCostExtractor.extract_cached_tokens(raw) == 750

    def test_cached_zero_for_none(self) -> None:
        assert ProviderCostExtractor.extract_cached_tokens(None) == 0

    def test_cached_zero_when_missing(self) -> None:
        assert ProviderCostExtractor.extract_cached_tokens({"usage": {}}) == 0

    def test_cached_zero_when_bool(self) -> None:
        # bool is a subtype of int; must not be counted as a token count.
        raw = {"usage": {"prompt_tokens_details": {"cached_tokens": True}}}
        assert ProviderCostExtractor.extract_cached_tokens(raw) == 0

    def test_cached_zero_when_negative(self) -> None:
        raw = {"usage": {"cache_read_input_tokens": -100}}
        assert ProviderCostExtractor.extract_cached_tokens(raw) == 0


# -----------------------------------------------------------------------------
# load_pricing_config
# -----------------------------------------------------------------------------


class TestLoadPricingConfig:
    def test_valid_entries(self) -> None:
        cfg = load_pricing_config(
            {
                "gpt-4o": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
                "claude-haiku-4-5": {
                    "input_per_mtok": 0.8,
                    "output_per_mtok": 4.0,
                    "cached_input_per_mtok": 0.08,
                },
            }
        )
        gpt = cfg.ops["gpt-4o"]
        haiku = cfg.ops["claude-haiku-4-5"]
        assert isinstance(gpt, LLMOp)
        assert gpt.input_per_mtok == 2.5
        assert gpt.output_per_mtok == 10.0
        assert gpt.cached_input_per_mtok is None
        assert isinstance(haiku, LLMOp)
        assert haiku.cached_input_per_mtok == 0.08

    def test_rejects_non_dict_input(self) -> None:
        with pytest.raises(TypeError, match="dict"):
            load_pricing_config([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_entry_not_dict(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            load_pricing_config({"m": 5.0})

    def test_rejects_missing_required(self) -> None:
        with pytest.raises(ValueError, match="input_per_mtok"):
            load_pricing_config({"m": {"output_per_mtok": 1.0}})

    def test_rejects_negative_input(self) -> None:
        with pytest.raises(ValueError, match="input_per_mtok"):
            load_pricing_config({"m": {"input_per_mtok": -1.0, "output_per_mtok": 1.0}})

    def test_rejects_nonfinite_output(self) -> None:
        with pytest.raises(ValueError, match="output_per_mtok"):
            load_pricing_config({"m": {"input_per_mtok": 1.0, "output_per_mtok": float("inf")}})

    def test_rejects_negative_cached(self) -> None:
        with pytest.raises(ValueError, match="cached_input_per_mtok"):
            load_pricing_config(
                {
                    "m": {
                        "input_per_mtok": 1.0,
                        "output_per_mtok": 1.0,
                        "cached_input_per_mtok": -0.5,
                    }
                }
            )

    def test_rejects_bool_rates(self) -> None:
        with pytest.raises(TypeError, match="must be numbers"):
            load_pricing_config({"m": {"input_per_mtok": True, "output_per_mtok": 1.0}})

    def test_rejects_bool_cached_rate(self) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            load_pricing_config(
                {
                    "m": {
                        "input_per_mtok": 1.0,
                        "output_per_mtok": 1.0,
                        "cached_input_per_mtok": False,
                    }
                }
            )

    def test_rejects_non_string_model_name(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            load_pricing_config({123: {"input_per_mtok": 1.0, "output_per_mtok": 1.0}})

    def test_cached_omitted_stays_none(self) -> None:
        cfg = load_pricing_config({"m": {"input_per_mtok": 1.0, "output_per_mtok": 1.0}})
        op = cfg.ops["m"]
        assert isinstance(op, LLMOp)
        assert op.cached_input_per_mtok is None

    def test_result_composes_with_llm_pricing_provider(self) -> None:
        # Round-trip: load a rate card, wire the provider, price a call.
        cfg = load_pricing_config({"gpt-4o": {"input_per_mtok": 2.5, "output_per_mtok": 10.0}})
        provider = LLMPricingProvider(cfg)
        assert provider.compute("gpt-4o", input_tokens=1_000_000, output_tokens=0) == pytest.approx(
            2.5
        )
