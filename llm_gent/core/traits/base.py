# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Trait, ScopedTrait and TraitBinding protocols, and a convenience base class.

Traits and bindings
-------------------

A :class:`Trait` lives on the platform (``platform.traits``). It holds what
every agent and run shares (clients, pools, a knowledge graph) and is always
running: the platform starts and stops it, never an agent. It keeps nothing
per agent. A trait with no per-agent data is used as it is:
``agent.require(T)`` and ``ctx.traits.require(T)`` return the platform's
instance.

A trait whose data is scoped to each agent is a :class:`ScopedTrait`.
An agent holds a :class:`TraitBinding` of it, from :meth:`ScopedTrait.bind`.
The binding holds all of that agent's data for the trait, from just the
agent's context key to a large working set in memory; the trait decides what
it holds and provides the interface to manage it. The agent owns its
bindings: it starts and stops them with itself.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable


if TYPE_CHECKING:
    from ..agent import Agent


@runtime_checkable
class TraitBinding(Protocol):
    """What an agent holds: its data for a trait, started and stopped with the agent.

    - on_start(): called when the agent starts (or on binding, if it has)
    - on_stop(): called when the agent stops; releases what the binding holds

    A binding from :meth:`ScopedTrait.bind` reaches its trait's shared
    resources; an agent-built trait (``BaseTrait``) is a binding with no
    platform trait.
    A binding can depend on another: use agent.get() in on_start() (all of
    an agent's bindings are added before its on_start()).
    """

    def on_start(self) -> None:
        """Called when the agent starts."""
        ...

    def on_stop(self) -> None:
        """Called when the agent stops."""
        ...


BindingT_co = TypeVar("BindingT_co", bound=TraitBinding, covariant=True)


@runtime_checkable
class Trait(Protocol):
    """A platform trait: what every agent and run shares, always running.

    ``trait_name`` is the name an agent config requires it by
    (``traits.required``). The platform starts it once it is built and stops
    it on close; an agent never does. It keeps nothing per agent: an agent
    that requires it gets the platform's instance.
    """

    trait_name: str

    def on_start(self) -> None:
        """Called by the platform once the trait is built."""
        ...

    def on_stop(self) -> None:
        """Called by the platform when it closes."""
        ...


@runtime_checkable
class ScopedTrait(Trait, Protocol[BindingT_co]):
    """A platform trait whose data is scoped to each agent.

    :meth:`bind` gives an agent its binding, which holds all of the agent's
    data; the trait keeps nothing per agent. An agent uses it through the
    binding it holds (``agent.use(T)``, then ``agent.require(T)``).
    """

    def bind(self, agent: Agent) -> BindingT_co:
        """A new binding of this trait for ``agent``, holding the agent's data."""
        ...


class BaseTrait:
    """Convenience base class for a trait an agent builds and holds itself.

    Provides common boilerplate: agent storage, property access, lifecycle
    stubs. Such a trait is a :class:`TraitBinding` with no platform trait.
    Users can also implement the protocol directly if they need different
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
        my_trait = agent.require(MyTrait)
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
