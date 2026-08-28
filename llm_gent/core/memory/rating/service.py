# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Generic rating service for LLM-based content evaluation.

This service is backend-agnostic and only handles LLM interaction.
Use with a backend (e.g., AtomicFactsRatingBackend) for persistence.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

from appinfra.log import Logger

from ...llm.json_cleaner import JSONCleaner
from .models import BatchItem, BatchRequest, Criteria, ProviderType, Request, Result


if TYPE_CHECKING:
    from ...llm import LLMCaller


def stars_to_signal(stars: int) -> tuple[Literal["positive", "negative", "dismiss"], float]:
    """Convert star rating to signal and strength.

    Args:
        stars: Star rating (1-5).

    Returns:
        Tuple of (signal, strength).
    """
    mapping: dict[int, tuple[Literal["positive", "negative", "dismiss"], float]] = {
        1: ("negative", 1.0),
        2: ("negative", 0.5),
        3: ("dismiss", 1.0),
        4: ("positive", 0.5),
        5: ("positive", 1.0),
    }
    return mapping.get(stars, ("dismiss", 1.0))


class Service:
    """Generic service for rating content using LLM.

    Backend-agnostic - handles prompt building, LLM calls, and response parsing.
    Does NOT handle persistence - use a backend for saving ratings.

    Example:
        from llm_infer.client import Factory as LLMClientFactory
        from llm_gent.core.llm import LLMCaller
        from llm_gent.core.memory.rating import Service, AtomicFactsBackend
        from llm_gent.core.memory.rating import Request, Criteria

        router = LLMClientFactory(lg).from_config(config)
        caller = LLMCaller(lg, router)
        service = Service(lg, caller)
        backend = AtomicFactsBackend(lg, database)

        # Rate content with LLM
        request = Request(
            fact=123,  # For atomic facts, this is the fact_id
            content="Why did the chicken cross the road?",
            prompt_template="You are rating a joke...",
            criteria=[Criteria(name="humor", description="Is it funny?", weight=1.0)],
            model="auto",
            provider="llm_primary",
        )
        result = service.rate_content(request)

        # Save using backend
        backend.save_rating(result, source="llm_rater")
    """

    def __init__(self, lg: Logger, llm_caller: LLMCaller) -> None:
        """Initialize rating service.

        Args:
            lg: Logger instance.
            llm_caller: LLM caller for rating (provides logging and dry-run).
        """
        self._lg = lg
        self._llm_caller = llm_caller

    def rate_content(self, request: Request) -> Result:
        """Rate content using LLM.

        Args:
            request: Rating request with content and parameters.

        Returns:
            Rating result with scores and reasoning.
        """
        prompt = self._build_prompt(request)
        response = self._call_llm(prompt, request)
        actual_model = response.model or request.model
        return self._parse_response(
            request.fact, response.content, request.criteria, actual_model, request.provider
        )

    def _build_prompt(self, request: Request) -> str:
        """Build rating prompt from template and criteria."""
        criteria_desc = "\n".join(
            f"- {c.name}: {c.description} (weight: {c.weight})" for c in request.criteria
        )

        return f"""{request.prompt_template}

Criteria:
{criteria_desc}

Content to rate:
{request.content}

Provide your rating in JSON format:
{{
  "stars": <1-5>,
  "criteria_scores": {{
    "criterion_name": <0.0-1.0>,
    ...
  }},
  "reasoning": "<brief explanation>"
}}

Rating scale:
- 5 stars (★★★★★): Excellent - exceeds expectations on all criteria
- 4 stars (★★★★☆): Good - meets expectations well
- 3 stars (★★★☆☆): Neutral - acceptable but unremarkable
- 2 stars (★★☆☆☆): Below expectations - has issues
- 1 star (★☆☆☆☆): Poor - fails to meet criteria

Respond only with the JSON, no additional text."""

    def _call_llm(self, prompt: str, request: Request | BatchRequest) -> Any:
        """Call LLM with rating prompt."""
        result = self._llm_caller.chat(
            messages=[{"role": "user", "content": prompt}],
            model=request.model,
            temperature=request.temperature,
            backend=request.backend,
        )
        return result

    def _parse_response(
        self,
        fact: Any,
        response: str,
        criteria: list[Criteria],
        model: str,
        provider: str,
    ) -> Result:
        """Parse LLM response into Result."""
        data = self._extract_json(response, criteria)
        # Clamp stars to valid 1-5 range
        raw_stars = data.get("stars", 3)
        stars = max(1, min(5, int(raw_stars)))
        signal, strength = self._stars_to_signal(stars)

        return Result(
            fact=fact,
            signal=signal,
            strength=strength,
            stars=stars,
            criteria_scores=data.get("criteria_scores", {}),
            reasoning=data.get("reasoning", ""),
            provider_type=ProviderType.LLM,
            model=model,
            provider=provider,
        )

    def _extract_json(self, response: str, criteria: list[Criteria]) -> dict[str, Any]:
        """Extract and parse JSON from LLM response.

        Uses JSONCleaner to handle code fences, auto-close braces,
        and extract only the first JSON object if multiple are returned.
        """
        try:
            cleaner = JSONCleaner()
            json_str = cleaner.extract_first_object(response)
            return cast(dict[str, Any], json.loads(json_str))
        except json.JSONDecodeError as e:
            self._lg.warning(
                "failed to parse rating JSON",
                extra={"exception": e, "response": response},
            )
            return self._fallback_rating(criteria)

    def _fallback_rating(self, criteria: list[Criteria]) -> dict[str, Any]:
        """Return neutral rating when parsing fails."""
        return {
            "stars": 3,
            "criteria_scores": {c.name: 0.5 for c in criteria},
            "reasoning": "Failed to parse rating response",
        }

    def _stars_to_signal(
        self, stars: int
    ) -> tuple[Literal["positive", "negative", "dismiss"], float]:
        """Convert star rating to signal and strength."""
        return stars_to_signal(stars)

    def _validate_stars(self, stars_raw: Any) -> int | None:
        """Validate and normalize star rating from LLM response.

        Accepts int or float (e.g., 3.0) that represents a valid integer in range 1-5.
        Returns None if invalid.
        """
        # Guard against bool (True/False are subclasses of int in Python)
        if isinstance(stars_raw, bool):
            return None
        if isinstance(stars_raw, int):
            if 1 <= stars_raw <= 5:
                return stars_raw
            return None
        if isinstance(stars_raw, float):
            # Accept floats like 3.0 that have no fractional part
            if stars_raw == int(stars_raw) and 1 <= stars_raw <= 5:
                return int(stars_raw)
            return None
        return None

    def _resolve_item_id(self, item_id: Any, item_map: dict[Any, BatchItem]) -> Any | None:
        """Resolve item_id with type normalization.

        LLM may return "123" (string) when we expect 123 (int) or vice versa.
        """
        # Direct match
        if item_id in item_map:
            return item_id
        # Try string conversion if int-like
        if isinstance(item_id, int):
            str_id = str(item_id)
            if str_id in item_map:
                return str_id
        # Try int conversion if numeric string
        if isinstance(item_id, str):
            try:
                int_id = int(item_id)
                if int_id in item_map:
                    return int_id
            except ValueError:
                pass
        return None

    def _truncate(self, text: str, max_len: int = 200) -> str:
        """Truncate text with '... (N more chars)' suffix."""
        if len(text) <= max_len:
            return text
        remaining = len(text) - max_len
        return f"{text[:max_len]}... ({remaining} more chars)"

    def _log_batch_results(self, items: list[BatchItem], results: list[Result], model: str) -> None:
        """Log batch rating results."""
        item_content = {i.fact: i.content for i in items}
        self._lg.debug(
            "rated batch",
            extra={
                "count": len(items),
                "model": model,
                "ratings": [
                    {
                        "fact_id": r.fact,
                        "stars": r.stars,
                        "rating": "★" * r.stars + "☆" * (5 - r.stars),
                        "fact": self._truncate(item_content.get(r.fact, "")),
                    }
                    for r in results
                ],
            },
        )

    # =========================================================================
    # Batch rating
    # =========================================================================

    def rate_batch(self, request: BatchRequest) -> list[Result]:
        """Rate multiple items in a single LLM call.

        Batches items together to reduce API costs by sharing prompt overhead.

        Args:
            request: Batch rating request with items and parameters.

        Returns:
            List of rating results (may be fewer than items if parsing fails).
            Returns empty list in dry-run mode.
        """
        if not request.items:
            return []

        self._lg.trace(
            "rating batch...",
            extra={"items": [{"id": i.fact, "content": i.content} for i in request.items]},
        )

        prompt = self._build_batch_prompt(request)
        response = self._call_llm(prompt, request)

        # Dry-run mode: no real response to parse
        if response.dry_run:
            self._lg.debug("rated batch", extra={"count": len(request.items), "dry_run": True})
            return []

        actual_model = response.model or request.model
        results = self._parse_batch_response(
            request.items, response.content, actual_model, request.provider
        )

        self._log_batch_results(request.items, results, actual_model)

        return results

    def _build_batch_prompt(self, request: BatchRequest) -> str:
        """Build batch rating prompt."""
        items_text = "\n\n".join(
            f"ITEM {i + 1} (ID: {item.fact}):\n{item.content}"
            for i, item in enumerate(request.items)
        )

        # Include structured criteria if provided (matches single-item rating behavior)
        criteria_section = ""
        if request.criteria:
            criteria_desc = "\n".join(
                f"- {c.name}: {c.description} (weight: {c.weight})" for c in request.criteria
            )
            criteria_section = f"\nCriteria:\n{criteria_desc}\n"

        return f"""{request.prompt_template}
{criteria_section}
{items_text}

Rate each item 1-5 stars. Respond with a JSON array:
```json
[
  {{"id": <item_id>, "stars": <1-5>, "reasoning": "<brief explanation>"}},
  ...
]
```

Be strict. Respond only with the JSON array."""

    def _parse_batch_response(
        self,
        items: list[BatchItem],
        response: str,
        model: str,
        provider: str,
    ) -> list[Result]:
        """Parse batch response into individual results.

        Deduplicates by fact_id - if LLM returns same ID twice, only first is kept.
        """
        results: list[Result] = []
        seen_facts: set[Any] = set()
        item_map = {item.fact: item for item in items}

        try:
            parsed = self._extract_batch_json(response)
            for rating_data in parsed:
                result = self._parse_batch_item(rating_data, item_map, model, provider)
                if result and result.fact not in seen_facts:
                    results.append(result)
                    seen_facts.add(result.fact)
        except Exception as e:
            self._lg.warning(
                "failed to parse batch response",
                extra={"exception": e, "response": response[:500]},
            )

        return results

    def _extract_batch_json(self, response: str) -> list[dict[str, Any]]:
        """Extract JSON array from batch response."""
        # Use JSONCleaner for consistent handling of code fences and edge cases
        cleaner = JSONCleaner()
        json_str = cleaner.clean(response)

        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            # Single item wrapped in object
            return [parsed]
        return cast(list[dict[str, Any]], parsed)

    def _parse_batch_item(
        self,
        data: dict[str, Any],
        item_map: dict[Any, BatchItem],
        model: str,
        provider: str,
    ) -> Result | None:
        """Parse a single item from batch response."""
        validated = self._validate_batch_item_data(data, item_map)
        if validated is None:
            return None

        item_id, stars, reasoning = validated
        signal, strength = self._stars_to_signal(stars)

        return Result(
            fact=item_id,
            signal=signal,
            strength=strength,
            stars=stars,
            criteria_scores={},
            reasoning=reasoning,
            provider_type=ProviderType.LLM,
            model=model,
            provider=provider,
        )

    def _validate_batch_item_data(
        self, data: dict[str, Any], item_map: dict[Any, BatchItem]
    ) -> tuple[Any, int, str] | None:
        """Validate and extract batch item data. Returns (item_id, stars, reasoning) or None."""
        item_id = data.get("id")
        stars_raw = data.get("stars")
        reasoning = str(data.get("reasoning", ""))

        # Resolve item_id with type normalization (LLM may return "123" vs 123)
        resolved_id = self._resolve_item_id(item_id, item_map)
        if resolved_id is None:
            self._lg.warning(
                "unknown item ID in batch response",
                extra={"id": item_id, "id_type": type(item_id).__name__},
            )
            return None

        # Accept int or float (e.g., 3.0) that represents a valid integer in range 1-5
        stars = self._validate_stars(stars_raw)
        if stars is None:
            self._lg.warning("invalid stars in batch response", extra={"stars": stars_raw})
            return None

        return (resolved_id, stars, reasoning)
