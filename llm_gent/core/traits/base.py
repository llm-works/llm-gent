# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Base trait protocol and convenience base class.

Dependency-inversion pattern for traits that wrap external resources
--------------------------------------------------------------------

A trait that wraps an external resource (an HTTP client, a DB connection
pool, a client factory) MUST take that resource through its constructor.
The trait itself imports no factory; ``TraitFactory`` owns construction
and passes the live resource in.

The ownership contract has three moving parts:

1. ``__init__(agent, ..., *, resource, owns_resource=False)`` — the
   resource is a required keyword argument; ``owns_resource`` defaults
   to False so callers who inject their own resource keep its lifecycle.
2. ``TraitFactory.create_<trait>_trait`` builds the resource and passes
   it with ``owns_resource=True``.
3. ``on_stop`` closes the resource iff ``owns_resource`` is True.

For runtime override, expose an immutable-view fluent
``.with_<resource>(replacement)`` that returns a new trait bound to
``replacement`` with ``owns_resource=False``. The new instance is
detached from the agent's registry; a caller wanting a persistent swap
calls ``agent.replace_trait(new)``.

Worked examples: ``TrainingTrait`` (train factory), ``SAIATrait`` (SAIA).

A trait over the platform's shared clients is built once by the platform
instead and handed to each agent as a view bound to its identity; the
platform closes those clients (``MemoryTrait`` and ``MemoryTrait.view``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from ..agent import Agent


@runtime_checkable
class Trait(Protocol):
    """Protocol for traits: an agent's capabilities, or resources a platform shares.

    The protocol is the lifecycle an agent drives:

    - on_start(): Called when the agent starts
    - on_stop(): Called when the agent stops

    A trait that belongs to one agent usually takes it in its constructor;
    BaseTrait provides that. A trait over the platform's shared clients is
    built once by the platform and handed to agents as views (see
    MemoryTrait.view), with no agent of its own.

    Traits can depend on other traits: use agent.get_trait() in on_start()
    (all traits are added before on_start()).
    """

    def on_start(self) -> None:
        """Called when agent starts."""
        ...

    def on_stop(self) -> None:
        """Called when agent stops."""
        ...


class BaseTrait:
    """Convenience base class for custom traits.

    Provides common boilerplate: agent storage, property access, lifecycle stubs.
    Users can also implement the Trait protocol directly if they need different
    patterns (e.g., dataclass-based traits like SAIATrait).

    Example:
        class MyTrait(BaseTrait):
            def __init__(self, agent: Agent, some_config: str) -> None:
                super().__init__(agent)
                self._config = some_config

            def do_something(self) -> str:
                # Access agent via self.agent property
                result = self.agent.complete("query")
                return result.content

        agent.add_trait(MyTrait(agent, "config"))
        my_trait = agent.get_trait(MyTrait)
        my_trait.do_something()
    """

    def __init__(self, agent: Agent) -> None:
        """Initialize base trait with agent reference.

        Args:
            agent: The agent this trait belongs to.
        """
        self._agent = agent

    @property
    def agent(self) -> Agent:
        """Access the agent this trait belongs to."""
        return self._agent

    def on_start(self) -> None:
        """Called when agent starts. Override to add startup logic."""

    def on_stop(self) -> None:
        """Called when agent stops. Override to add cleanup logic."""
