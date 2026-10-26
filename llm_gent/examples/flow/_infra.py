# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shared scaffolding for the Flow-substrate examples.

Two scripted stub :class:`SAIAFactory` implementations so the examples
run deterministically without contacting a real LLM backend:

* :class:`StubSAIAFactory` — the minimal ``.answer(prompt) -> str``
  surface, for examples where the LLM's job is just to produce a string.
* :class:`StructuredStubSAIAFactory` — mirrors the production
  :meth:`llm_saia.SAIA.complete_structured(prompt, schema) -> VerbResult[T]`
  surface, for examples that demonstrate schema-typed output.

Both stub factories map :class:`~llm_gent.flow.Role` to per-role
scripted response queues.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from llm_saia import VerbResult

from llm_gent.flow import Role


T = TypeVar("T")


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


@runtime_checkable
class StructuredSAIA(Protocol):
    """Schema-typed saia-shaped surface used by structured-output examples.

    Mirrors :meth:`llm_saia.SAIA.complete_structured`; each verb pins
    the schema at call time and receives a validated :class:`VerbResult`.
    Real applications route to the production SAIA unchanged; the
    example :class:`StructuredStubSAIA` scripts responses so examples
    run without a backend.
    """

    async def complete_structured(self, prompt: str, schema: type[T]) -> VerbResult[T]:
        """Return a :class:`VerbResult` whose ``.value`` matches ``schema``."""
        ...


@dataclass
class StructuredStubSAIA:
    """Per-role stub of :class:`StructuredSAIA` scripted by ``(schema, value)`` pairs.

    Each response is a ``(schema, value)`` tuple. On dispatch the stub
    checks that the schema the verb asks for matches the schema the
    script was written against — drift surfaces at the call site
    instead of later as a validation failure on unrelated data.

    A scripted ``value`` that is a :class:`BaseException` instance is
    raised instead of returned, so examples that need to demonstrate
    rescue / :class:`~llm_gent.flow.Failure` paths can script a per-call
    failure without patching internals.
    """

    role_name: str
    responses: list[tuple[type, Any]] = field(default_factory=list)
    call_log: list[tuple[str, type]] = field(default_factory=list)

    async def complete_structured(self, prompt: str, schema: type[T]) -> VerbResult[T]:
        """Pop the next scripted response, matching schema against the ask."""
        self.call_log.append((prompt, schema))
        if not self.responses:
            raise RuntimeError(
                f"StructuredStubSAIA[{self.role_name}] response queue exhausted "
                f"after {len(self.call_log)} calls — script one more response"
            )
        scripted_schema, value = self.responses.pop(0)
        if scripted_schema is not schema:
            raise RuntimeError(
                f"StructuredStubSAIA[{self.role_name}] scripted "
                f"{scripted_schema.__name__} but verb asked for {schema.__name__}"
            )
        if isinstance(value, BaseException):
            raise value
        if not isinstance(value, schema):
            raise RuntimeError(
                f"StructuredStubSAIA[{self.role_name}] scripted "
                f"{type(value).__name__} for {schema.__name__}"
            )
        return VerbResult(value=value)


class StructuredStubSAIAFactory:
    """SAIAFactory that hands each :class:`Role` its own scripted :class:`StructuredStubSAIA`.

    Construct with a ``{role_name: [(schema, value), ...]}`` script.
    Each role is built once, cached, and its responses pop in call
    order. A role dispatched with no script raises the moment its
    verb calls :meth:`StructuredStubSAIA.complete_structured`.
    """

    def __init__(
        self,
        scripts: Mapping[str, Iterable[tuple[type, Any]]] | None = None,
    ) -> None:
        """Capture per-role ``(schema, value)`` queues; nothing is built until ``build``."""
        self._scripts: dict[str, list[tuple[type, Any]]] = {
            k: list(v) for k, v in (scripts or {}).items()
        }
        self._instances: dict[str, StructuredStubSAIA] = {}

    def build(self, role: Role) -> StructuredStubSAIA:
        """Return the cached (or freshly-built) stub for ``role``."""
        if role.name not in self._instances:
            self._instances[role.name] = StructuredStubSAIA(
                role_name=role.name,
                responses=list(self._scripts.get(role.name, [])),
            )
        return self._instances[role.name]

    def instance(self, role_name: str) -> StructuredStubSAIA:
        """Return the built stub for ``role_name`` (raises if not yet built)."""
        return self._instances[role_name]
