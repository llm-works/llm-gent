# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Factory surfaces for the flow substrate.

Two factories live here, at different scopes:

- :class:`SAIAFactory` — protocol that turns a :class:`Role` into a saia
  instance. Typically one per application; the wiring (backend, tools,
  system prompt) is deployment-specific so the framework only names the
  contract.
- :class:`FlowFactory` — app-scoped bundle of the ambient ``lg`` and (by
  convention) a single ``SAIAFactory``. Provides :meth:`create` for
  building Flows without repeating those two arguments at every
  construction site, and :meth:`with_saia_f` for deriving a factory that
  swaps the SAIAFactory (e.g. a plugin subsystem).

Naming policy: any ``saia_f=`` kwarg on the framework's public API takes a
:class:`SAIAFactory` (never a saia instance). The kwarg is named after
the concept; the type carries the mechanism.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

from appinfra.log import Logger

from ..core.traits import Registry as TraitRegistry
from .nodes import UNSET
from .role import Role


if TYPE_CHECKING:
    from .flow import Flow


class SAIAFactory(Protocol):
    """Constructs a role-bound saia instance.

    Implementations decide how a :class:`Role` maps to a backend + tools +
    system prompt + call options. The framework's only expectation is that
    :meth:`build` returns an object supporting saia's public surface — the
    verb calling code uses ``.verify(...)`` / ``.complete(...)`` / etc.

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
                if self._tools:
                    builder = builder.tools(self._tools, executor)
                if role.style:
                    builder = builder.system(role.style)
                return builder.build()
    """

    def build(self, role: Role) -> Any:
        """Return a saia instance configured for ``role``."""
        ...


class FlowFactory:
    """App-scoped factory for :class:`Flow` — captures ``lg`` and ``saia`` once.

    An application typically has one logger and one :class:`SAIAFactory`
    covering every :class:`Flow` it constructs. Repeating both at every
    Flow-construction site is noise; :class:`FlowFactory` bundles them
    once so subsystem builders read as ``f.create("grade").call(...)``.

    :meth:`create` builds a Flow with the captured defaults;
    :meth:`with_saia_f` returns a new :class:`FlowFactory` whose SAIAFactory
    is swapped (for subsystems that need a different saia builder).
    """

    def __init__(
        self,
        lg: Logger,
        *,
        saia_f: SAIAFactory | None = None,
        state: Any = UNSET,
        traits: TraitRegistry | None = None,
        halt: asyncio.Event | None = None,
    ) -> None:
        """Capture the ambient environment for subsequent :meth:`create` calls.

        Args:
            lg: Logger threaded into every :class:`Flow` this factory builds.
            saia_f: A :class:`SAIAFactory`. The ``_f`` suffix carries the
                framework-wide policy: any ``saia_f=`` kwarg takes a factory,
                never a saia instance.
            state: Default construction ``state`` for built flows. Per-Flow
                overrides go through :meth:`create`; per-run overrides go
                through :meth:`Flow.run`.
            traits: Optional trait registry propagated to every :class:`Flow`
                built by this factory. Verbs reach mounted capabilities via
                ``ctx.traits``. ``None`` yields flows with ``ctx.traits is
                None``.
            halt: Optional :class:`asyncio.Event` attached via
                :meth:`Flow.with_halt` on every built flow. Wire once at
                the factory to thread the same halt handle through an
                entire agent shape.
        """
        self._lg = lg
        self._saia_f = saia_f
        self._state = state
        self._traits = traits
        self._halt = halt

    def create(self, name: str = "", *, state: Any = UNSET) -> Flow:
        """Return a :class:`Flow` using this factory's captured environment.

        Args:
            name: Optional identifier — used in error messages and traces.
                Also lets the flow serve as a named node inside a parent
                chain.
            state: Per-Flow construction ``state`` override. Passing
                :data:`UNSET` (default) inherits the factory's ``state``;
                passing ``None`` explicitly is honored as "payload is
                ``None``"; any other value replaces the factory default.
        """
        from .flow import Flow

        resolved_state = self._state if state is UNSET else state
        flow = Flow(
            self._lg,
            name,
            saia_f=self._saia_f,
            state=resolved_state,
            traits=self._traits,
        )
        if self._halt is not None:
            flow.with_halt(self._halt)
        return flow

    def with_saia_f(self, saia_f: SAIAFactory) -> FlowFactory:
        """Return a new :class:`FlowFactory` whose :class:`SAIAFactory` is swapped.

        ``lg``, ``state``, ``traits``, and ``halt`` are preserved. Useful
        for subsystems that share the app's logger but need a different
        saia builder (e.g. a plugin with its own model wiring).
        """
        return FlowFactory(
            self._lg,
            saia_f=saia_f,
            state=self._state,
            traits=self._traits,
            halt=self._halt,
        )

    def with_traits(self, traits: TraitRegistry | None) -> FlowFactory:
        """Return a new :class:`FlowFactory` whose trait registry is swapped.

        ``lg``, ``saia_f``, ``state``, and ``halt`` are preserved. Mirrors
        :meth:`with_saia_f` for the trait dimension.
        """
        return FlowFactory(
            self._lg,
            saia_f=self._saia_f,
            state=self._state,
            traits=traits,
            halt=self._halt,
        )

    def with_halt(self, event: asyncio.Event) -> FlowFactory:
        """Return a new :class:`FlowFactory` whose halt event is swapped.

        ``lg``, ``saia_f``, ``state``, and ``traits`` are preserved. Every
        subsequently created :class:`Flow` gets ``event`` attached via
        :meth:`Flow.with_halt` — one wiring reaches every layer that
        observes ``ctx.halt``.
        """
        return FlowFactory(
            self._lg,
            saia_f=self._saia_f,
            state=self._state,
            traits=self._traits,
            halt=event,
        )
