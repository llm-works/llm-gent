# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shared scaffolding for the Flow-substrate examples.

Provides a scripted stub :class:`SAIAFactory` so the examples run
deterministically without contacting a real LLM backend. Verbs in the
examples reach the stub via ``ctx.saia.answer(prompt)`` — a minimal
example-only surface that maps 1:1 to ``llm_saia.SAIA.complete_structured``
in production (see :class:`ExampleSAIA` docstring for the swap).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from llm_gent.flow import Role


@runtime_checkable
class ExampleSAIA(Protocol):
    """Minimal saia-shaped surface used by verbs in the Flow examples.

    Real applications route LLM calls through
    ``llm_saia.SAIA.complete_structured(prompt, schema)``; the examples
    define a tiny ``.answer(prompt) -> str`` method so :class:`StubSAIAFactory`
    can script responses without wiring a backend. In a production
    SAIAFactory, ``.answer`` is a two-line wrapper::

        def answer(self, prompt: str) -> str:
            return self._saia.complete_structured(prompt, str).value
    """

    def answer(self, prompt: str) -> str:
        """Return the LLM's answer to ``prompt``."""
        ...


@dataclass
class StubSAIA:
    """Per-role scripted stub of :class:`ExampleSAIA`.

    Each :meth:`StubSAIAFactory.build` call for a role returns a fresh
    :class:`StubSAIA` with its own response queue. :meth:`answer` pops
    the queue's head; an exhausted queue is a hard error so an under-
    scripted example fails loud instead of masking a real bug.
    """

    role_name: str
    responses: list[str] = field(default_factory=list)
    call_log: list[str] = field(default_factory=list)

    def answer(self, prompt: str) -> str:
        """Return the next scripted response, logging the prompt."""
        self.call_log.append(prompt)
        if not self.responses:
            raise RuntimeError(
                f"StubSAIA[{self.role_name}] response queue exhausted after "
                f"{len(self.call_log)} calls — script one more response"
            )
        return self.responses.pop(0)


class StubSAIAFactory:
    """:class:`SAIAFactory` mapping :class:`Role` to scripted :class:`StubSAIA`.

    Construct with a ``{role_name: [response, ...]}`` script. Each role
    is built once, cached, and its responses pop in call order. Roles
    named in ``scripts`` but never dispatched sit unused; a role
    dispatched with no script raises the moment its verb calls
    ``ctx.saia.answer(...)``.
    """

    def __init__(self, scripts: Mapping[str, Iterable[str]] | None = None) -> None:
        """Capture per-role response queues; nothing is built until ``build``."""
        self._scripts: dict[str, list[str]] = {k: list(v) for k, v in (scripts or {}).items()}
        self._instances: dict[str, StubSAIA] = {}

    def build(self, role: Role) -> StubSAIA:
        """Return the cached (or freshly-built) stub for ``role``."""
        if role.name not in self._instances:
            self._instances[role.name] = StubSAIA(
                role_name=role.name,
                responses=list(self._scripts.get(role.name, [])),
            )
        return self._instances[role.name]

    def instance(self, role_name: str) -> StubSAIA:
        """Return the built stub for ``role_name`` (raises if not yet built)."""
        return self._instances[role_name]
