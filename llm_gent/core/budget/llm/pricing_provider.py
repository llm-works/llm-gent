# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Batteries-included LLM :class:`PricingProvider` for OSS consumers.

Wraps a :class:`PricingConfig` rate card and adds the parts every
LLM-cost consumer needs but nobody wants to re-implement:

- **Publisher-prefix strip** — ``google/gemini-2.5-flash`` resolves to
  the bare ``gemini-2.5-flash`` key. llm-infer surfaces qualified
  names; rate cards typically don't duplicate every entry with each
  publisher.
- **Boundary-safe prefix match** — a ``grok-3`` key no longer silently
  matches a call for ``grok-30``. The next character after the match
  must be one of ``-_./`` or end-of-string.
- **Per-provider dispatch** on ``response.provider`` (duck-typed —
  no llm-infer hard dep):

  - Anthropic — ``prompt_tokens`` on the wire is uncached-only;
    ``cache_read_input_tokens`` bills at ``cached_input_per_mtok``
    (or ``anthropic_cache_read_multiplier`` × input rate when the
    per-mtok cached rate is unset), ``cache_creation_input_tokens``
    bills at ``anthropic_cache_write_multiplier`` × input rate.
  - OpenAI-compat (openai, xai, groq, deepseek, together, fireworks,
    …) — ``prompt_tokens`` includes cached; ``cached_tokens`` is a
    subset billed at the cached rate. xAI reports cost in
    ``usage.cost_in_usd_ticks`` (via :class:`ProviderCostExtractor`)
    and that wins verbatim.
  - Gemini — standard input/output token math only. Priority-tier
    surcharges and Gemini's context-caching API are consumer concerns;
    wrap this class if you need them.
  - Unknown / missing ``provider`` — best-effort OpenAI-compat path.

- **``provider_cost=<float>`` scalar override** short-circuits every
  path — trusted invoice values, negotiated flat rates, test injection.
- **``on_missing=RAISE`` by default** — silent-$0 on an unknown model is
  a footgun (unrecorded runs read as free). Callers who genuinely want
  lenient mode opt in with ``OnMissing.ZERO``.

The Anthropic multipliers are constructor-configurable so consumers with
negotiated rates that diverge from list pricing can adjust without
wrapping.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from ..pricing import LLMOp, Op, PricingConfig
from .extractor import ProviderCostExtractor


__all__ = [
    "LLMPricingProvider",
    "MissingModelCostError",
    "OnMissing",
]


class OnMissing(StrEnum):
    """Policy for :meth:`LLMPricingProvider.compute` on an unknown op.

    - ``RAISE`` (default) — raise :class:`MissingModelCostError` so the
      run fails loud instead of silently recording $0 costs.
    - ``ZERO`` — return ``0.0``. Matches the legacy passive-registry
      behaviour; opt-in for test fixtures and prototypes.
    """

    RAISE = "raise"
    ZERO = "zero"


class MissingModelCostError(LookupError):
    """No cost rule matched the requested op name.

    Carries ``op_name`` and ``configured`` so alerting / fallback code
    can react without re-parsing the message string.
    """

    def __init__(self, op_name: str, configured: list[str]) -> None:
        self.op_name = op_name
        self.configured = configured
        super().__init__(f"no cost rule configured for op '{op_name}'. Configured: {configured}")


_DEFAULT_ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25
_DEFAULT_ANTHROPIC_CACHE_READ_MULTIPLIER = 0.1
_ANTHROPIC_PROVIDERS = frozenset({"anthropic"})
_GEMINI_PROVIDERS = frozenset({"gemini", "google", "vertex", "vertexai"})
_BOUNDARY_CHARS = "-_./"


class LLMPricingProvider:
    """Batteries-included :class:`PricingProvider` for LLM cost tracking.

    Constructed with a :class:`PricingConfig` rate card. Typical wiring::

        pricing = PricingConfig(ops={
            "claude-haiku-4-5": LLMOp(
                name="claude-haiku-4-5",
                input_per_mtok=0.80,
                output_per_mtok=4.00,
                cached_input_per_mtok=0.08,
            ),
            "gpt-4o-mini": LLMOp(
                name="gpt-4o-mini",
                input_per_mtok=0.15,
                output_per_mtok=0.60,
                cached_input_per_mtok=0.075,
            ),
        })
        provider = LLMPricingProvider(pricing)
        tracker = Tracker(lg, provider, budget=100.0)

    Verb bodies pass an llm-infer ``ChatResponse`` (or any object with
    ``provider`` / ``raw`` attributes) as ``response=`` for provider
    dispatch, or an explicit ``provider_cost=<float>`` to override.
    """

    def __init__(
        self,
        pricing: PricingConfig,
        *,
        on_missing: OnMissing = OnMissing.RAISE,
        anthropic_cache_write_multiplier: float = _DEFAULT_ANTHROPIC_CACHE_WRITE_MULTIPLIER,
        anthropic_cache_read_multiplier: float = _DEFAULT_ANTHROPIC_CACHE_READ_MULTIPLIER,
    ) -> None:
        if not isinstance(on_missing, OnMissing):
            raise TypeError(
                f"on_missing must be an OnMissing member, got {type(on_missing).__name__}"
            )
        _validate_multiplier("anthropic_cache_write_multiplier", anthropic_cache_write_multiplier)
        _validate_multiplier("anthropic_cache_read_multiplier", anthropic_cache_read_multiplier)
        self._pricing = pricing
        self._on_missing = on_missing
        self._cache_write_mult = anthropic_cache_write_multiplier
        self._cache_read_mult = anthropic_cache_read_multiplier

    def compute(self, op_name: str, /, **usage: Any) -> float:
        """Compute cost for one usage event.

        Precedence:

        1. ``provider_cost=<float>`` in ``usage`` — returned verbatim
           after finite/non-negative validation.
        2. ``response=<llm-infer ChatResponse>`` in ``usage`` — dispatch
           on ``response.provider``.
        3. Neither — rate-card lookup on ``op_name`` and
           ``op.cost(**usage)``.

        Missing rule raises :class:`MissingModelCostError` when
        ``on_missing`` is :attr:`OnMissing.RAISE` (the default); returns
        ``0.0`` when :attr:`OnMissing.ZERO`.
        """
        override = usage.pop("provider_cost", None)
        if override is not None:
            cost = float(override)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"provider_cost must be finite and >= 0, got {override!r}")
            return cost

        response = usage.pop("response", None)
        if response is not None:
            return self._compute_from_response(op_name, response, usage)

        op = self._lookup(op_name)
        if op is None:
            return self._on_miss(op_name)
        return op.cost(**usage)

    def _compute_from_response(
        self, model: str, response: Any, caller_usage: dict[str, Any]
    ) -> float:
        raw = _raw_dict(response)
        if raw is not None:
            reported = ProviderCostExtractor.extract(raw)
            if reported is not None:
                return reported
        op = self._lookup(model)
        if op is None:
            return self._on_miss(model)
        if not isinstance(op, LLMOp):
            return op.cost(**caller_usage)
        provider = _detect_provider(response, raw)
        return self._dispatch_llm_op(op, raw, provider, caller_usage)

    def _dispatch_llm_op(
        self,
        op: LLMOp,
        raw: dict[str, Any] | None,
        provider: str,
        caller_usage: dict[str, Any],
    ) -> float:
        input_tokens = _nonneg_int(caller_usage.get("input_tokens"))
        output_tokens = _nonneg_int(caller_usage.get("output_tokens"))
        explicit_cached = caller_usage.get("cached_tokens")
        if provider in _ANTHROPIC_PROVIDERS:
            return _compute_anthropic(
                op,
                raw,
                input_tokens,
                output_tokens,
                explicit_cached,
                cache_read_mult=self._cache_read_mult,
                cache_write_mult=self._cache_write_mult,
            )
        if provider in _GEMINI_PROVIDERS:
            return _compute_gemini(op, input_tokens, output_tokens)
        return _compute_openai_compat(op, raw, input_tokens, output_tokens, explicit_cached)

    def _lookup(self, name: str) -> Op | None:
        """Resolve ``name`` against the rate card.

        - Publisher-prefix strip: ``google/gemini-2.5-flash`` →
          ``gemini-2.5-flash``.
        - Exact match first.
        - Boundary-safe prefix match (next char in ``-_./`` or
          end-of-string); longest-prefix wins, ``"default"`` excluded.
        - Falls back to the ``"default"`` key when set and nothing
          else fits.
        """
        ops = self._pricing.ops
        if "/" in name:
            name = name.split("/", 1)[1]
        if name in ops:
            return ops[name]

        matches: list[str] = []
        for key in ops:
            if key == "default" or not name.startswith(key):
                continue
            rest = name[len(key) :]
            if not rest or rest[0] in _BOUNDARY_CHARS:
                matches.append(key)
        if matches:
            return ops[max(matches, key=len)]

        return ops.get("default")

    def _on_miss(self, op_name: str) -> float:
        if self._on_missing is OnMissing.RAISE:
            raise MissingModelCostError(
                op_name,
                sorted(k for k in self._pricing.ops if k != "default"),
            )
        return 0.0


def _validate_multiplier(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


def _compute_anthropic(
    op: LLMOp,
    raw: dict[str, Any] | None,
    input_tokens: int,
    output_tokens: int,
    explicit_cached: Any,
    *,
    cache_read_mult: float,
    cache_write_mult: float,
) -> float:
    """Anthropic billing.

    - ``prompt_tokens`` on the wire (and llm-infer's normalized
      ``input_tokens``) is uncached-only. Cache read + cache write are
      separate fields.
    - ``cache_read_input_tokens`` × ``cached_input_per_mtok`` when set,
      otherwise × ``cache_read_mult`` × ``input_per_mtok``.
    - ``cache_creation_input_tokens`` × ``cache_write_mult`` × input rate.
    """
    wire_usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(wire_usage, dict):
        wire_usage = {}

    if explicit_cached is None:
        cache_read = _nonneg_int(wire_usage.get("cache_read_input_tokens"))
    else:
        cache_read = _nonneg_int(explicit_cached)
    cache_write = _nonneg_int(wire_usage.get("cache_creation_input_tokens"))

    cache_read_rate = (
        op.cached_input_per_mtok
        if op.cached_input_per_mtok is not None
        else op.input_per_mtok * cache_read_mult
    )

    return (
        input_tokens * op.input_per_mtok / 1_000_000
        + cache_read * cache_read_rate / 1_000_000
        + cache_write * op.input_per_mtok * cache_write_mult / 1_000_000
        + output_tokens * op.output_per_mtok / 1_000_000
    )


def _compute_openai_compat(
    op: LLMOp,
    raw: dict[str, Any] | None,
    input_tokens: int,
    output_tokens: int,
    explicit_cached: Any,
) -> float:
    """OpenAI-family billing (openai, xai, groq, deepseek, …).

    ``prompt_tokens`` is total input INCLUDING cached; ``cached_tokens``
    is a subset billed at ``cached_input_per_mtok`` (typically ~0.5×
    input for OpenAI, provider-specific for others). Delegates the
    include-cached-in-input math to :meth:`LLMOp.cost`.
    """
    if explicit_cached is None and raw is not None:
        cached: int | None = ProviderCostExtractor.extract_cached_tokens(raw)
    else:
        cached = _nonneg_int(explicit_cached)
    return op.cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached or 0,
    )


def _compute_gemini(op: LLMOp, input_tokens: int, output_tokens: int) -> float:
    """Gemini billing — base input/output only.

    Priority-tier surcharges (``x-gemini-service-tier`` header) and
    context-caching (per-hour storage + per-token access) are consumer
    concerns; wrap this class when you need them.
    """
    return (
        input_tokens * op.input_per_mtok / 1_000_000
        + output_tokens * op.output_per_mtok / 1_000_000
    )


def _detect_provider(response: Any, raw: dict[str, Any] | None) -> str:
    """Return a lower-case provider tag.

    Preference: ``response.provider`` when a non-empty string. Falls
    back to sniffing the wire dict for provider-exclusive field names
    (Anthropic's ``cache_*_input_tokens``). Unknown → ``"openai"``.
    """
    tag = getattr(response, "provider", None)
    if isinstance(tag, str) and tag.strip():
        return tag.strip().lower()
    if isinstance(raw, dict):
        usage = raw.get("usage")
        if isinstance(usage, dict) and any(
            key in usage for key in ("cache_creation_input_tokens", "cache_read_input_tokens")
        ):
            return "anthropic"
    return "openai"


def _raw_dict(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        return response
    raw = getattr(response, "raw", None)
    return raw if isinstance(raw, dict) else None


def _nonneg_int(value: Any) -> int:
    """Coerce to non-negative int; ``None``/non-int/bool/negative collapse to 0."""
    if value is None or isinstance(value, bool):
        return 0
    if not isinstance(value, int):
        return 0
    return max(value, 0)
