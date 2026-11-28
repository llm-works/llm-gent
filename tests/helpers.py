# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shared test utilities."""

import functools
import os
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

from appinfra.cas import FileStore, MemoryStore, Store
from appinfra.log import Logger, quick_console_logger

from llm_gent.core.platform import Platform, SAIAFactory
from llm_gent.core.platform.factory import _NoSAIAFactory
from llm_gent.core.tools.factory import Factory as ToolFactory
from llm_gent.core.traits import Registry as TraitRegistry
from llm_gent.core.traits.builtin.memory import MemoryTrait
from llm_gent.core.traits.factory import Factory as TraitFactory
from llm_gent.flow import Factory, Flow, FlowRunner


@functools.cache
def make_test_logger() -> Logger:
    """Return the process-wide test logger (suppressed output).

    appinfra lets a process create a logger name once, so every test shares one instance.
    The session fixture ``test_logger`` creates it before the first test: appinfra's
    per-test cleanup destroys loggers a test creates, and keeps those that existed before.
    """
    return quick_console_logger("test", config={"level": "error"})


# A flow does not carry the platform its runs run on or where they keep their histories;
# the runner's platform and store do. Tests build flows in one place and run them in
# another, so the test factory and flow below stamp the platform and store a test gives
# them on the flows they make, and run_flow starts the run with them. The library never
# reads these attributes.
_PLATFORM = "_test_platform"
_STORE = "_test_store"


class TestFactory(Factory[Flow]):
    """:class:`Factory` that also takes ``saia_factory=`` / ``traits=`` / ``checkpoint_store=``.

    The flows it creates carry a :class:`Platform` of the first two (one
    per factory, so its flows share their SAIAs) and the store, which
    :func:`run_flow` runs them with.
    """

    __test__ = False

    def __init__(
        self,
        lg: Logger,
        *,
        saia_factory: SAIAFactory | None = None,
        traits: TraitRegistry | None = None,
        checkpoint_store: Store | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(lg, **kwargs)
        self._saia_factory = saia_factory
        self._traits = traits
        self._platform = make_platform(lg, saia_factory=saia_factory, traits=traits)
        self._store = checkpoint_store

    def create(self, name: str = "") -> Flow:
        flow = super().create(name)
        setattr(flow, _PLATFORM, self._platform)
        setattr(flow, _STORE, self._store)
        return flow

    def with_saia_factory(self, saia_factory: SAIAFactory) -> "TestFactory":
        platform = make_platform(self._lg, saia_factory=saia_factory, traits=self._traits)
        return self._replace(saia_factory=saia_factory, platform=platform)  # type: ignore[return-value]

    def with_traits(self, traits: TraitRegistry | None) -> "TestFactory":
        platform = make_platform(self._lg, saia_factory=self._saia_factory, traits=traits)
        return self._replace(traits=traits, platform=platform)  # type: ignore[return-value]

    def with_checkpoint_store(self, store: Store) -> "TestFactory":
        return self._replace(store=store)  # type: ignore[return-value]


class TestFlow(Flow):
    """:class:`Flow` that also takes ``saia_factory=`` / ``traits=`` / ``checkpoint_store=``."""

    __test__ = False

    def __init__(
        self,
        lg: Logger,
        name: str = "",
        *,
        saia_factory: SAIAFactory | None = None,
        traits: TraitRegistry | None = None,
        checkpoint_store: Store | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(lg, name, **kwargs)
        setattr(self, _PLATFORM, make_platform(lg, saia_factory=saia_factory, traits=traits))
        setattr(self, _STORE, checkpoint_store)


def platform_of(flow: Flow) -> Platform:
    """The platform a test stamped on ``flow``; a stub platform when none was."""
    platform: Platform | None = getattr(flow, _PLATFORM, None)
    return platform if platform is not None else make_platform(make_test_logger())


def store_of(flow: Flow) -> Store | None:
    """The store a test stamped on ``flow``; ``None`` when none was."""
    store: Store | None = getattr(flow, _STORE, None)
    return store


async def run_flow(
    flow: Flow,
    *args: Any,
    platform: Platform | None = None,
    store: Store | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``flow`` once on a runner with the test logger and return its result.

    The runner's platform and store are ``platform`` / ``store``, else the
    ones a test factory or test flow stamped on ``flow``. A coroutine, so it
    fits wherever a test awaits a run, gathers runs or hands one to
    ``asyncio.run``; ``FlowRunner.start`` takes the other arguments.
    """
    run_platform = platform if platform is not None else platform_of(flow)
    runner_store = store if store is not None else store_of(flow)
    runner = FlowRunner(make_test_logger(), platform=run_platform, store=runner_store)
    return await runner.start(flow, *args, **kwargs)


def memory_store() -> MemoryStore:
    """Return an empty in-memory checkpoint store."""
    return MemoryStore(make_test_logger())


def file_store(root: str | os.PathLike[str]) -> FileStore:
    """Return a checkpoint store under ``root``, without fsync: tests don't cut the power."""
    return FileStore(make_test_logger(), root, fsync=False)


def make_platform(
    lg: Logger,
    *,
    router: Any = None,
    saia_factory: SAIAFactory | None = None,
    traits: TraitRegistry | None = None,
    memory_config: Mapping[str, Any] | None = None,
    memory: MemoryTrait | None = None,
) -> Platform:
    """A platform on stub clients, built directly: no config file read, no client built.

    ``router`` defaults to a ``MagicMock``. Without ``saia_factory`` the platform
    builds no SAIA, as one from ``PlatformFactory`` without one; without
    ``traits`` its registry is empty. ``memory``, the shared memory trait, joins
    the registry as ``PlatformFactory`` registers it; with ``traits`` given, it
    joins a copy, so the caller's registry is never changed.
    """
    router = router if router is not None else MagicMock()
    trait_factory = TraitFactory(lg, router, memory_config=memory_config)
    return Platform(
        lg,
        llm=router,
        saia_factory=saia_factory if saia_factory is not None else _NoSAIAFactory(),
        trait_factory=trait_factory,
        tool_factory=ToolFactory(lg),
        traits=_registry(lg, traits, memory),
        owned=[],
    )


def _registry(
    lg: Logger, traits: TraitRegistry | None, memory: MemoryTrait | None
) -> TraitRegistry:
    """``traits`` as given without memory; otherwise a new registry of its traits plus ``memory``."""
    if memory is None:
        return traits if traits is not None else TraitRegistry(lg)
    registry = TraitRegistry(lg)
    for trait in traits.all() if traits is not None else []:
        registry.register(trait)
    registry.register(memory)
    return registry


def create_mock_trait(trait_class, **mock_attrs):
    """Create a mock trait that works with the registry.

    Creates an object whose type() returns the trait_class, allowing
    registry lookups to work correctly.

    Args:
        trait_class: The trait class this mock should impersonate.
        **mock_attrs: Attributes to add to the mock. Values can be:
            - MagicMock: Used directly
            - dict: Passed as **kwargs to MagicMock
            - bool/int/float/str: Stored directly (not as mock)
            - Other: Wrapped in MagicMock(return_value=...)

    Returns:
        Mock object that appears as instance of trait_class.
    """

    class MockHolder:
        pass

    obj = MockHolder()

    # Add mock methods/attributes to __dict__ BEFORE setting __class__
    # This avoids triggering property descriptors
    for attr_name, config in mock_attrs.items():
        if isinstance(config, MagicMock):
            # Already a mock, use it directly
            mock_attr = config
        elif isinstance(config, dict):
            # Dict config for MagicMock
            mock_attr = MagicMock(**config)
        else:
            # For simple scalar values, store them directly (not as MagicMock)
            # This is important for properties like has_embedder that return bool
            if isinstance(config, (bool, int, float, str)):
                mock_attr = config
            else:
                # For lists and other objects, create a MagicMock that returns them
                # Note: accessing this attribute returns a callable mock; call it to get the original value
                mock_attr = MagicMock(return_value=config)
        obj.__dict__[attr_name] = mock_attr

    # Set __class__ AFTER populating __dict__
    obj.__class__ = trait_class

    return obj
