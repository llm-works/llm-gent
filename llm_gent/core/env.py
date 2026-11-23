# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""What a run may use: how roles become SAIA instances, and the traits.

An :class:`Environment` owns nothing: the platform owns the clients and
their lifecycle; an environment selects what a run uses of them. Everything
that uses one environment shares its SAIA instances, one per role. That is
safe because a SAIA holds configuration only (backend, tools and their
executor, system prompt, guards): every call keeps its messages in the
conversation it is passed. Work that needs SAIAs of its own (its own tools,
its own directive) uses an environment of its own.
"""

from __future__ import annotations

from typing import Any, Protocol

from .role import Role
from .traits import Registry as TraitRegistry


class SAIAFactory(Protocol):
    """Constructs a role-bound saia instance.

    Implementations decide how a :class:`Role` maps to a backend + tools +
    system prompt + call options. The framework's only expectation is that
    :meth:`build` returns an object supporting saia's public surface — the
    verb calling code uses ``.verify(...)`` / ``.complete(...)`` / etc.

    Implementations MAY read :attr:`Role.params` for per-run parameters
    (max_iterations, cost trackers, session identifiers, task state —
    anything the consumer needs at build time that isn't captured in the
    typed Role fields). Key naming inside ``params`` is a contract between
    a factory and its callers; gent itself imposes none.

    Reference sketch (users typically write one specific to their yaml
    config)::

        class MySAIAFactory:
            def __init__(self, lg, llm_yaml, tools=None):
                self._lg = lg
                self._llm_yaml = llm_yaml
                self._tools = tools or []

            def build(self, role):
                backend = build_backend_from_config(self._llm_yaml, role)
                builder = SAIA.builder().backend(backend).logger(self._lg)
                if "max_iterations" in role.params:
                    builder = builder.max_iterations(role.params["max_iterations"])
                if self._tools:
                    builder = builder.tools(self._tools, executor)
                if role.style:
                    builder = builder.system(role.style)
                return builder.build()
    """

    def build(self, role: Role) -> Any:
        """Return a saia instance configured for ``role``."""
        ...


class Environment:
    """The SAIA factory and traits a run uses, with one SAIA per role built on first use.

    ``saia_factory`` builds a role's SAIA the first time :meth:`saia` is
    asked for it; later asks get the same instance. Without one, work that
    never asks for a SAIA runs, and an ask raises. ``traits`` is the
    registry steps see as ``ctx.traits``.

    The cache is one ``dict`` per environment. Two tasks on one event loop
    that ask for an unbuilt role at once may each build it; one of the two
    is kept, and both are configured alike.
    """

    def __init__(
        self, saia_factory: SAIAFactory | None = None, traits: TraitRegistry | None = None
    ) -> None:
        self._saia_factory = saia_factory
        self._traits = traits
        self._saia_by_role: dict[Role, Any] = {}

    @property
    def saia_factory(self) -> SAIAFactory | None:
        """How roles become SAIA instances; ``None`` without one."""
        return self._saia_factory

    @property
    def traits(self) -> TraitRegistry | None:
        """The trait registry steps see as ``ctx.traits``; ``None`` without one."""
        return self._traits

    def saia(self, role: Role) -> Any:
        """The SAIA for ``role``, built on the first request and kept.

        Raises:
            RuntimeError: The environment has no SAIA factory.
        """
        cached = self._saia_by_role.get(role)
        if cached is not None:
            return cached
        if self._saia_factory is None:
            raise RuntimeError(
                f"the environment has no SAIAFactory to build a SAIA for role {role.name!r}"
            )
        built = self._saia_factory.build(role)
        self._saia_by_role[role] = built
        return built
