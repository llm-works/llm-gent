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

from llm_gent.core.env import Environment, SAIAFactory
from llm_gent.core.platform import Config, Context
from llm_gent.core.tools.factory import Factory as ToolFactory
from llm_gent.core.traits import Registry as TraitRegistry
from llm_gent.core.traits.factory import Factory as TraitFactory
from llm_gent.core.traits.factory import MemoryResources
from llm_gent.flow import Factory, Flow, FlowRunner


@functools.cache
def make_test_logger() -> Logger:
    """Return the process-wide test logger (suppressed output).

    appinfra lets a process create a logger name once, so every test shares one instance.
    The session fixture ``test_logger`` creates it before the first test: appinfra's
    per-test cleanup destroys loggers a test creates, and keeps those that existed before.
    """
    return quick_console_logger("test", config={"level": "error"})


# A flow does not carry what its runs may use or where they keep their histories; the
# runner's environment and store do. Tests build flows in one place and run them in
# another, so the test factory and flow below stamp the environment and store a test gives
# them on the flows they make, and run_flow starts the run with them. The library never
# reads these attributes.
_ENV = "_test_env"
_STORE = "_test_store"


class TestFactory(Factory[Flow]):
    """:class:`Factory` that also takes ``saia_factory=`` / ``traits=`` / ``checkpoint_store=``.

    The flows it creates carry an :class:`Environment` of the first two (one
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
        self._env = Environment(saia_factory, traits)
        self._store = checkpoint_store

    def create(self, name: str = "") -> Flow:
        flow = super().create(name)
        setattr(flow, _ENV, self._env)
        setattr(flow, _STORE, self._store)
        return flow

    def with_saia_factory(self, saia_factory: SAIAFactory) -> "TestFactory":
        traits = self._env.traits
        return self._replace(env=Environment(saia_factory, traits))  # type: ignore[return-value]

    def with_traits(self, traits: TraitRegistry | None) -> "TestFactory":
        saia_factory = self._env.saia_factory
        return self._replace(env=Environment(saia_factory, traits))  # type: ignore[return-value]

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
        setattr(self, _ENV, Environment(saia_factory, traits))
        setattr(self, _STORE, checkpoint_store)


def env_of(flow: Flow) -> Environment:
    """The environment a test stamped on ``flow``; an empty one when none was."""
    env: Environment | None = getattr(flow, _ENV, None)
    return env if env is not None else Environment()


def store_of(flow: Flow) -> Store | None:
    """The store a test stamped on ``flow``; ``None`` when none was."""
    store: Store | None = getattr(flow, _STORE, None)
    return store


async def run_flow(
    flow: Flow,
    *args: Any,
    env: Environment | None = None,
    store: Store | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``flow`` once on a runner with the test logger and return its result.

    The runner's environment and store are ``env`` / ``store``, else the
    ones a test factory or test flow stamped on ``flow``. A coroutine, so it
    fits wherever a test awaits a run, gathers runs or hands one to
    ``asyncio.run``; ``FlowRunner.start`` takes the other arguments.
    """
    run_env = env if env is not None else env_of(flow)
    runner_store = store if store is not None else store_of(flow)
    runner = FlowRunner(make_test_logger(), env=run_env, store=runner_store)
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
    llm_config: Mapping[str, Any] | None = None,
    memory_config: Mapping[str, Any] | None = None,
    memory: MemoryResources | None = None,
) -> Context:
    """A platform context on stub clients, built directly: no config file read, no client built.

    ``router`` defaults to a ``MagicMock``; ``llm_config`` is the router config
    the trait factory hands LLM traits.
    """
    router = router if router is not None else MagicMock()
    config = Config(
        llm=dict(llm_config or {}), memory=dict(memory_config) if memory_config else None
    )
    trait_factory = TraitFactory(
        lg, llm_config or {}, router, memory_config=memory_config, memory=memory
    )
    return Context(
        lg,
        config,
        llm=router,
        memory=memory,
        trait_factory=trait_factory,
        tool_factory=ToolFactory(lg),
        owned=[],
    )


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
