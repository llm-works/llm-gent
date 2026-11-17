# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Real llm-infer clients for the examples that run against a model.

``--base-url`` selects an OpenAI-compatible endpoint (e.g. a local server);
without it the examples use Anthropic, keyed by ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from appinfra.log import Logger
from llm_infer.client import Factory as LLMInferFactory
from llm_infer.client import LLMClient


ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
"""Cheap, tool-capable Anthropic model. Override with ``--model``."""


def real_client(lg: Logger, base_url: str | None, model: str | None) -> LLMClient:
    """An OpenAI-compatible client when ``base_url`` is set, otherwise Anthropic.

    ``model`` defaults to the first model the endpoint lists, or
    :data:`ANTHROPIC_MODEL`.
    """
    factory = LLMInferFactory(lg)
    if base_url is None:
        return factory.anthropic(default_model=model or ANTHROPIC_MODEL)
    # llm_infer appends endpoint paths (/models, /chat/completions) to
    # base_url verbatim, so a bare server root would 404.
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    model = model or _first_served_model(factory, base_url)
    print(f"  model: {model} @ {base_url}")
    return factory.openai(base_url=base_url, default_model=model)


def _first_served_model(factory: LLMInferFactory, base_url: str) -> str:
    """Return the first model the endpoint lists on ``/v1/models``."""
    with factory.openai(base_url=base_url) as probe:
        models = probe.backend.list_models()
    if not models:
        raise RuntimeError(f"{base_url} lists no models; pass --model")
    return models[0]
