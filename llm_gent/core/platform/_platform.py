# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The platform: what a process's agents and runs share, built once.

A :class:`Platform` holds the shared model router, the factories (how
roles become SAIA instances, how traits and tools are built), the trait
registry, one SAIA per role, and the clients it was told it owns. Anyone may build one: the platform
:class:`~llm_gent.core.platform.Factory` builds it from config, and code
that has its own parts passes them to the constructor. Runs that share a
platform share its SAIA instances. That is safe because a SAIA holds
configuration only (backend, tools and their executor, system prompt,
guards): every call keeps its messages in the conversation it is passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Protocol

from appinfra.log import Logger
from llm_infer.client import ChatClient

from ..role import Role
from ..tools.factory import Factory as ToolFactory
from ..traits.factory import Factory as TraitFactory
from ..traits.registry import Registry as TraitRegistry


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


class Closeable(Protocol):
    """A resource the platform may own: closed synchronously, optionally also on the event loop."""

    def close(self) -> None: ...


class Platform:
    """What a process's agents and runs share: the router, factories, traits, SAIAs.

    ``llm`` is the shared model router. ``saia_factory`` builds a role's SAIA the first time a step asks
    (``ctx.saia``); later asks, from any run on this platform, get the same
    instance. ``trait_factory`` and ``tool_factory`` build traits and tools on
    the shared clients. ``traits`` is the registry steps see as
    ``ctx.traits``; it may be empty. The platform closes the resources in
    ``owned`` on :meth:`close` or at the end of a ``with`` / ``async with``
    block; everything else stays with whoever built it.

    The SAIA cache is one ``dict`` per platform. Two runs on one event loop
    that ask for an unbuilt role at once may each build it; one of the two
    is kept, and both are configured alike.

    Example:
        async with PlatformFactory(lg).from_config({}) as platform:
            agent = AgentFactory(platform).from_config({"identity": {"name": "hello"}})
            result = await FlowRunner(lg, platform=platform).start(flow, "input")
    """

    def __init__(
        self,
        lg: Logger,
        *,
        llm: ChatClient,
        saia_factory: SAIAFactory,
        trait_factory: TraitFactory,
        tool_factory: ToolFactory,
        traits: TraitRegistry,
        owned: Sequence[Closeable],
    ) -> None:
        """Hold the platform's parts.

        Args:
            lg: Platform logger.
            llm: The shared model router.
            saia_factory: Builds the SAIA for a role, once per role.
            trait_factory: Builds traits on the shared clients.
            tool_factory: Builds tools.
            traits: The trait registry steps see as ``ctx.traits``.
            owned: Resources the platform closes, in order, on :meth:`close`.
        """
        self._lg = lg
        self._llm = llm
        self._saia_factory = saia_factory
        self._trait_factory = trait_factory
        self._tool_factory = tool_factory
        self._traits = traits
        self._owned = list(owned)
        self._saia_by_role: dict[Role, Any] = {}
        self._closed = False

    @property
    def logger(self) -> Logger:
        """Platform logger."""
        return self._lg

    @property
    def llm(self) -> ChatClient:
        """The shared model router."""
        return self._llm

    @property
    def saia_factory(self) -> SAIAFactory:
        """How roles become SAIA instances."""
        return self._saia_factory

    @property
    def trait_factory(self) -> TraitFactory:
        """Builds traits on the shared clients."""
        return self._trait_factory

    @property
    def tool_factory(self) -> ToolFactory:
        """Builds tools."""
        return self._tool_factory

    @property
    def traits(self) -> TraitRegistry:
        """The trait registry steps see as ``ctx.traits``."""
        return self._traits

    def saia(self, role: Role) -> Any:
        """The SAIA for ``role``, built by the SAIA factory on the first request and kept."""
        cached = self._saia_by_role.get(role)
        if cached is not None:
            return cached
        built = self._saia_factory.build(role)
        self._saia_by_role[role] = built
        return built

    def close(self) -> None:
        """Close the resources this platform owns. Later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        close_resources(self._lg, self._owned)
        self._lg.debug("platform closed", extra={"closed": len(self._owned)})

    async def aclose(self) -> None:
        """Close the owned resources' async clients on this event loop, then their sync side.

        Marked closed before the first ``await``, so a concurrent ``aclose()``
        or ``close()`` does nothing. The sync side closes even when the task is
        cancelled during an async close; the cancellation then propagates.
        """
        if self._closed:
            return
        self._closed = True
        try:
            for resource in self._owned:
                aclose = getattr(resource, "aclose", None)
                if aclose is None:
                    continue
                try:
                    await aclose()
                except Exception as e:
                    self._lg.warning(
                        "platform resource aclose failed",
                        extra={"resource": type(resource).__name__, "exception": e},
                    )
        finally:
            close_resources(self._lg, self._owned)
            self._lg.debug("platform closed", extra={"closed": len(self._owned)})

    def __enter__(self) -> Platform:
        """Use the platform in a ``with`` block; it closes at the end."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the owned resources."""
        self.close()

    async def __aenter__(self) -> Platform:
        """Use the platform in an ``async with`` block; it closes at the end."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the owned resources."""
        await self.aclose()

    def __repr__(self) -> str:
        """String representation of the platform."""
        return f"Platform(traits={self._traits.count()}, owned={len(self._owned)})"


def close_resources(lg: Logger, resources: Sequence[Closeable]) -> None:
    """Close each of ``resources`` in order; a failure is logged and the rest still close."""
    for resource in resources:
        try:
            resource.close()
        except Exception as e:
            lg.warning(
                "platform resource close failed",
                extra={"resource": type(resource).__name__, "exception": e},
            )
