# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""llm-infer clients for the examples.

``--smoke`` runs on llm-infer's scripted client: no network, no model.
Otherwise ``--base-url`` selects an OpenAI-compatible endpoint (e.g. a local
server); without it the examples use Anthropic, keyed by ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import argparse

from appinfra.log import Logger
from llm_infer.client import Factory as LLMInferFactory
from llm_infer.client import LLMClient


ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
"""Cheap, tool-capable Anthropic model. Override with ``--model``."""


def parse_client_args(description: str, argv: list[str]) -> argparse.Namespace:
    """Parse ``--smoke``, ``--base-url`` and ``--model``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true", help="Scripted responses, no network.")
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible server (/v1 appended if missing). Omitted → Anthropic.",
    )
    parser.add_argument(
        "--model", help=f"Model id. Default: the endpoint's first model, else {ANTHROPIC_MODEL}."
    )
    return parser.parse_args(argv)


def example_client(
    lg: Logger, *, smoke: bool, base_url: str | None, model: str | None
) -> LLMClient:
    """The scripted client for ``--smoke``, else :func:`real_client`.

    Without a script, the scripted client calls the first offered tool on the
    first turn and the last one after that, with arguments built from the tool's
    schema, and answers plain text otherwise.
    """
    if smoke:
        return LLMInferFactory(lg).scripted()
    return real_client(lg, base_url, model)


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
