# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Role — first-class config for a persona bound to an LLM backend + model.

A ``Role`` wraps the configuration a
:class:`~llm_gent.core.SAIAFactory` needs to build a
role-bound saia client. It exists because neither ``llm_saia`` nor
``llm_infer`` carries the "persona with a job" concept as a first-class
object — saia knows verbs, llm-infer knows routing strings. The flow layer
on top of both needs a small explicit shape.

``Role`` carries two kinds of information: statically-typed persona config
(name, and optionally backend, model, temperature, max_tokens, style) and an open
``params`` mapping of further configuration the enclosing consumer's
:class:`SAIAFactory` interprets at build time (max_iterations, a tool set's
name — anything gent itself doesn't model). A role is the key of a
platform's SAIA cache, so it holds configuration only: a run's own values
(its cost tracker, its state, a session) reach a step through ``ctx.cost``
and ``ctx.state``, never through a role.
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

    Only ``name`` is required: ``Role(name="planning")`` binds a step to a
    role whose backend, model, temperature and max_tokens the app's
    :class:`SAIAFactory` fills in (e.g. from its config, by name). Gent
    itself reads none of them; a field left ``None`` is the factory's to
    decide.
    """

    name: str
    """Role identifier — used as llm-infer's routing key."""

    backend: str | None = None
    """llm-infer backend id (e.g. ``"openai"``, ``"anthropic"``); ``None``: the factory's."""

    model: str | None = None
    """Model identifier within the backend; ``None``: the factory's."""

    temperature: float | None = None
    """Sampling temperature; ``None``: the factory's."""

    max_tokens: int | None = None
    """Maximum completion tokens per call; ``None``: the factory's."""

    style: str | None = None
    """Optional prompt preamble injected by the flow into system prompts."""

    params: dict[str, Any] = field(default_factory=dict, hash=False, compare=True)
    """Further configuration consumed by the enclosing :class:`SAIAFactory` at
    build time. Opaque to gent — key naming is a contract between the
    factory and its callers (e.g. a factory reading ``max_iterations``
    from here).

    Configuration only. A :class:`~llm_gent.core.Platform` keeps the SAIA
    it builds for a role for the life of the process, and roles differing
    in ``params`` are different keys: a value that differs per run (a cost
    tracker, a state, a session id) would leave one SAIA per run in the
    cache. Such values reach a step through ``ctx.cost`` and ``ctx.state``.

    Excluded from :meth:`__hash__` so ``Role`` stays hashable with
    unhashable values (a dict, a list); participates in :meth:`__eq__` so
    roles differing only in ``params`` build apart.

    Do not mutate ``params`` in place after construction — use
    :meth:`with_params` to create a new Role with changed parameters.
    """

    def with_params(self, **kv: Any) -> Role:
        """Return a new ``Role`` with ``kv`` merged into :attr:`params`.

        Later calls override earlier keys::

            r = ROLE.with_params(max_iterations=20, toolset="search")
            r2 = r.with_params(max_iterations=30)   # r2.params["max_iterations"] == 30
        """
        return replace(self, params={**self.params, **kv})
