"""Base trait protocol and convenience base class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from ..agent import Agent


@runtime_checkable
class Trait(Protocol):
    """Protocol for agent traits.

    Traits extend agent capabilities through composition. Each trait
    is bound to an agent at construction and integrates with the agent
    lifecycle:

    - __init__(agent): Trait is created with agent reference
    - on_start(): Called when agent starts
    - on_stop(): Called when agent stops

    Traits can depend on other traits. Use agent.get_trait() in on_start()
    to get references to required traits (all traits are added before on_start()).

    For convenience, inherit from BaseTrait instead of implementing this
    protocol directly.
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
        pass

    def on_stop(self) -> None:
        """Called when agent stops. Override to add cleanup logic."""
        pass
