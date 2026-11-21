# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The platform context: a worker's shared scope, passed to everything that needs it."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol

from appinfra.log import Logger
from llm_infer.client import ChatClient

from ..tools.factory import ToolFactory
from ..traits.factory import MemoryResources, TraitFactory
from .config import Config


class Closeable(Protocol):
    """A resource the context may own: closed synchronously, optionally also on the event loop."""

    def close(self) -> None: ...


class Context:
    """The shared scope of a worker: clients, factories and config used by all its agents.

    Built by the platform :class:`~llm_gent.core.platform.Factory`; it builds
    nothing itself. It holds the shared model router, the memory clients (when
    memory is configured), the trait and tool factories, and the config they
    were built from. It closes the resources it was told it owns, on
    :meth:`close` or at the end of a ``with`` / ``async with`` block; injected
    resources stay with whoever built them.

    Example:
        async with PlatformFactory(lg).from_config({}) as platform:
            agent = AgentFactory(platform).from_config({"identity": {"name": "hello"}})
    """

    def __init__(
        self,
        lg: Logger,
        config: Config,
        *,
        llm: ChatClient,
        memory: MemoryResources | None,
        trait_factory: TraitFactory,
        tool_factory: ToolFactory,
        owned: Sequence[Closeable],
    ) -> None:
        """Hold the platform's resources.

        Args:
            lg: Platform logger.
            config: The config the platform was built from.
            llm: The shared model router.
            memory: The shared memory clients, or None without memory config.
            trait_factory: Builds traits on the shared clients.
            tool_factory: Builds tools.
            owned: Resources this context closes, in order, on :meth:`close`.
        """
        self._lg = lg
        self._config = config
        self._llm = llm
        self._memory = memory
        self._trait_factory = trait_factory
        self._tool_factory = tool_factory
        self._owned = list(owned)
        self._closed = False

    @property
    def logger(self) -> Logger:
        """Platform logger."""
        return self._lg

    @property
    def config(self) -> Config:
        """The config the platform was built from."""
        return self._config

    @property
    def llm(self) -> ChatClient:
        """The shared model router."""
        return self._llm

    @property
    def memory(self) -> MemoryResources | None:
        """The shared memory clients, or None without memory config."""
        return self._memory

    @property
    def trait_factory(self) -> TraitFactory:
        """Builds traits on the shared clients."""
        return self._trait_factory

    @property
    def tool_factory(self) -> ToolFactory:
        """Builds tools."""
        return self._tool_factory

    def close(self) -> None:
        """Close the resources this context owns. Later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        close_resources(self._lg, self._owned)
        self._lg.debug("platform closed", extra={"closed": len(self._owned)})

    async def aclose(self) -> None:
        """Close the owned resources' async clients on this event loop, then :meth:`close`."""
        if self._closed:
            return
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
        self.close()

    def __enter__(self) -> Context:
        """Use the context in a ``with`` block; it closes at the end."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the owned resources."""
        self.close()

    async def __aenter__(self) -> Context:
        """Use the context in an ``async with`` block; it closes at the end."""
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
        """String representation of the platform context."""
        return f"platform.Context(memory={self._memory is not None}, owned={len(self._owned)})"


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
