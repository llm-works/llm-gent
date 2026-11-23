# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Platform factory: builds a process's :class:`Platform` from config."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger
from llm_infer.client import ChatClient, EmbeddingClient
from llm_infer.client import Factory as LLMClientFactory
from llm_kelt.core import Database

from ..role import Role
from ..tools.factory import Factory as ToolFactory
from ..traits.factory import Factory as TraitFactory
from ..traits.factory import MemoryResources
from ..traits.registry import Registry as TraitRegistry
from ._platform import Closeable, Platform, SAIAFactory, close_resources
from .config import Config


class Factory:
    """Builds a :class:`Platform`: the clients and factories a process's agents and runs share.

    The model router and, with memory configured, the memory database, chat
    client and embedder are built once here and injected; the platform owns
    and closes them. A client given through :meth:`with_llm` is used instead
    of building a router, and stays the caller's to close. The platform's
    SAIA factory is the one given through :meth:`with_saia_factory`; without
    one, the platform builds no SAIA and a step reading ``ctx.saia`` raises.
    The platform's trait registry starts empty.

    Example:
        async with PlatformFactory(lg).from_config({}) as platform:
            ...

        llm = LLMClientFactory(lg).openai(base_url="http://localhost:8000/v1")
        platform = PlatformFactory(lg).with_llm(llm).from_config({})
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize the factory.

        Args:
            lg: Logger for the platform and the clients it builds.
        """
        self._lg = lg
        self._llm: ChatClient | None = None
        self._saia_factory: SAIAFactory = _NoSAIAFactory()

    def with_llm(self, client: ChatClient) -> Factory:
        """A copy of this factory whose platforms use ``client`` as their model router.

        The caller keeps ownership: the platform does not close ``client``.
        """
        factory = copy.copy(self)
        factory._llm = client
        return factory

    def with_saia_factory(self, saia_factory: SAIAFactory) -> Factory:
        """A copy of this factory whose platforms build their SAIAs with ``saia_factory``."""
        factory = copy.copy(self)
        factory._saia_factory = saia_factory
        return factory

    def from_config(self, config: Config | Mapping[str, Any]) -> Platform:
        """Build the platform described by ``config``.

        Args:
            config: A platform :class:`Config`, or a mapping validated into one.

        Returns:
            The platform, owning the clients built here.
        """
        settings = config if isinstance(config, Config) else Config.model_validate(config)
        router_config = settings.router_config()
        owned: list[Closeable] = []
        try:
            router = self._router(router_config, owned)
            memory = self._memory(settings.memory, owned)
        except Exception:
            close_resources(self._lg, owned)
            raise
        trait_factory = TraitFactory(self._lg, router, memory_config=settings.memory, memory=memory)
        self._lg.debug(
            "platform built",
            extra={"default": router_config.get("default"), "memory": memory is not None},
        )
        return Platform(
            self._lg,
            llm=router,
            saia_factory=self._saia_factory,
            trait_factory=trait_factory,
            tool_factory=ToolFactory(self._lg),
            traits=TraitRegistry(self._lg),
            owned=owned,
        )

    def _router(self, router_config: dict[str, Any], owned: list[Closeable]) -> ChatClient:
        """The injected client, or a router built from ``router_config`` and owned."""
        if self._llm is not None:
            return self._llm
        router = LLMClientFactory(self._lg).from_config(DotDict(router_config))
        owned.append(router)
        return router

    def _memory(
        self, memory_config: dict[str, Any] | None, owned: list[Closeable]
    ) -> MemoryResources | None:
        """The shared memory clients, built when the memory settings name a database."""
        if not memory_config or "db" not in memory_config:
            return None
        from appinfra.db.pg import PG

        database = Database(self._lg, PG(self._lg, memory_config["db"]))
        owned.append(_DisposeOnClose(database))
        client_factory = LLMClientFactory(self._lg)
        chat_client = client_factory.from_config(DotDict(memory_config.get("llm") or {}))
        owned.append(chat_client)
        embedder = _embedder(client_factory, memory_config)
        if embedder is not None:
            owned.append(embedder)
        return MemoryResources(database=database, chat_client=chat_client, embedder=embedder)


def _embedder(
    client_factory: LLMClientFactory, memory_config: Mapping[str, Any]
) -> EmbeddingClient | None:
    """The memory embedder, when the settings name an ``embedder_url``."""
    url = memory_config.get("embedder_url")
    if not url:
        return None
    embedder: EmbeddingClient = client_factory.embeddings(
        base_url=url,
        model=memory_config.get("embedder_model", "default"),
        timeout=memory_config.get("embedder_timeout", 30.0),
    )
    return embedder


class _NoSAIAFactory:
    """The SAIA factory of a platform built without one: building a SAIA raises."""

    def build(self, role: Role) -> Any:
        """Raise: a platform built without a SAIA factory builds no SAIA."""
        raise RuntimeError(
            f"this platform builds no SAIA (role {role.name!r}): build it with "
            f"PlatformFactory(lg).with_saia_factory(...), or pass saia_factory= to Platform"
        )


class _DisposeOnClose:
    """Closes a kelt :class:`Database` by disposing its engine and pool."""

    def __init__(self, database: Database) -> None:
        """Wrap ``database``; nothing is disposed until :meth:`close`."""
        self._database = database

    def close(self) -> None:
        """Dispose the database's engine and connection pool."""
        self._database.dispose()
