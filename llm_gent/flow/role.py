# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Role — first-class config for a persona bound to an LLM backend + model.

A ``Role`` wraps the configuration a :class:`SAIAFactory` needs to build a
role-bound saia client. It exists because neither ``llm_saia`` nor
``llm_infer`` carries the "persona with a job" concept as a first-class
object — saia knows verbs, llm-infer knows routing strings. The flow layer
on top of both needs a small explicit shape.

``Role`` carries two kinds of information: statically-typed persona config
(name, backend, model, temperature, max_tokens, style) and an open
``params`` mapping for per-run parameters the enclosing consumer's
:class:`SAIAFactory` interprets at build time (max_iterations, cost
trackers, campaign identifiers, plan state — anything gent itself doesn't
model). Consumers vary a role for a run via :meth:`Role.with_params`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


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

    params: dict[str, Any] = field(default_factory=dict, hash=False, compare=True)
    """Per-run parameters consumed by the enclosing :class:`SAIAFactory` at
    build time. Opaque to gent — key naming is a contract between the
    factory and its callers (e.g. an xray factory reads ``max_iterations``,
    ``run_cost``, ``campaign_id``, ``plan_state`` from here).

    Excluded from :meth:`__hash__` so ``Role`` remains hashable even when
    ``params`` values are unhashable objects (trackers, state instances).
    Participates in :meth:`__eq__` so roles differing only in ``params``
    correctly miss the ``Flow._saia_by_role`` cache — each unique
    ``params`` snapshot yields a distinct SAIA build.
    """

    def with_params(self, **kv: Any) -> Role:
        """Return a new ``Role`` with ``kv`` merged into :attr:`params`.

        Later calls override earlier keys::

            r = ROLE_XRAY.with_params(max_iterations=20, run_cost=tracker)
            r2 = r.with_params(max_iterations=30)   # r2.params["max_iterations"] == 30
        """
        return replace(self, params={**self.params, **kv})
