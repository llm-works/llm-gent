"""Role — first-class config for a persona bound to an LLM backend + model.

A ``Role`` wraps the configuration a :class:`SAIAFactory` needs to build a
role-bound saia client. It exists because neither ``llm_saia`` nor
``llm_infer`` carries the "persona with a job" concept as a first-class
object — saia knows verbs, llm-infer knows routing strings. The flow layer
on top of both needs a small explicit shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    """Configuration for a persona bound to an LLM backend + model.

    A ``Role`` is pure config — it does not build clients. A
    :class:`SAIAFactory` reads a ``Role`` and constructs the actual saia
    instance. The ``Role`` name is also the identifier used for llm-infer's
    routing param.
    """

    name: str
    """Role identifier — used as llm-infer's routing key."""

    backend: str
    """llm-infer backend id (e.g. ``"openai"``, ``"anthropic"``, ``"gemini"``)."""

    model: str
    """Model identifier within the backend."""

    temperature: float = 0.7
    """Sampling temperature."""

    max_tokens: int = 4096
    """Maximum completion tokens per call."""

    style: str | None = None
    """Optional prompt preamble injected by the flow into system prompts."""
