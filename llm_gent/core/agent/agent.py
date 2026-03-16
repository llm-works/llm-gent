"""Abstract base class for agents."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from appinfra import DotDict
from appinfra.log import Logger

from ..errors import TraitNotFoundError
from ..runnable import Runnable
from ..traits.base import BaseTrait
from ..traits.registry import Registry as TraitRegistry
from .types import ExecutionResult


if TYPE_CHECKING:
    from .identity import Identity

TraitT = TypeVar("TraitT", bound=BaseTrait)


class Agent(Runnable):
    """Abstract base class for agents.

    Provides trait composition and lifecycle management. Subclasses
    implement the concrete agent behavior.

    For LLM operations, attach SAIATrait and call agent.get_trait(SAIATrait).saia
    directly. For learning, attach LearnTrait and access via get_trait().

    Example:
        class MyAgent(Agent):
            def __init__(self, lg: Logger, name: str) -> None:
                super().__init__(lg)
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            def start(self) -> None:
                self._start_traits()

            def stop(self) -> None:
                self._stop_traits()
    """

    def __init__(self, lg: Logger, config: DotDict | dict[str, Any] | None = None) -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
            config: Agent configuration (converted to DotDict if dict, empty if None).
                   Must contain identity.name if config is provided.

        Raises:
            ConfigError: If identity.name is missing from config.
        """
        self._lg = lg
        if isinstance(config, DotDict):
            self._config = config
        elif config is not None:
            self._config = DotDict(**config)
        else:
            self._config = DotDict()

        # Identity is required - raises ConfigError if missing
        self._identity: Identity = self._resolve_identity()
        self._traits = TraitRegistry(lg)
        self._started = False
        self._cycle_count = 0

    @property
    def name(self) -> str:
        """Agent identifier from identity.

        Returns:
            Agent name from identity.
        """
        return self._identity.name

    @property
    def identity(self) -> Identity:
        """Agent identity.

        Returns:
            Agent's Identity instance.
        """
        return self._identity

    @property
    def cycle_count(self) -> int:
        """Number of execution cycles completed.

        Returns:
            Count of execution cycles. Incremented by subclasses in run_once().
        """
        return self._cycle_count

    @property
    def lg(self) -> Logger:
        """Logger instance for this agent.

        All traits should use self.agent.lg instead of storing their own logger.
        """
        return self._lg

    @property
    def config(self) -> DotDict:
        """Agent configuration.

        Returns empty DotDict if no configuration was provided.
        Traits can access agent configuration via this method.

        Returns:
            Agent's configuration as DotDict.
        """
        return self._config

    @property
    def traits(self) -> TraitRegistry:
        """Trait registry for this agent.

        Provides introspection of attached traits:
        - traits.all() - get all trait instances
        - traits.count() - count attached traits
        - traits.types() - get all trait types

        Note:
            While the registry exposes mutation methods (register, replace, clear),
            prefer using agent.add_trait() to add traits. This ensures proper
            lifecycle management if the agent is already started.

        Returns:
            The agent's trait registry.
        """
        return self._traits

    @abstractmethod
    def start(self) -> None:
        """Start the agent.

        Subclasses should call _start_traits() to start attached traits.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the agent.

        Subclasses should call _stop_traits() to stop attached traits.
        """
        ...

    @abstractmethod
    def record_feedback(self, message: str) -> None:
        """Record feedback about execution.

        Args:
            message: The feedback message.
        """
        ...

    @abstractmethod
    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of recent ExecutionResult objects.
        """
        ...

    # =========================================================================
    # Trait Management
    # =========================================================================

    def add_trait(self, trait: BaseTrait) -> None:
        """Add a trait to this agent.

        Traits must be constructed with this agent as a parameter. If the agent
        is already started, the trait's on_start() is called automatically.

        Args:
            trait: The trait instance to add (must be constructed with this agent).

        Raises:
            DuplicateTraitError: If a trait of this type is already added.
        """
        self._traits.register(trait)

        if self._started:
            try:
                trait.on_start()
            except Exception:
                # Unregister trait if start failed to avoid partial initialization
                self._traits.unregister(type(trait))
                raise

    def get_trait(self, trait_type: type[TraitT]) -> TraitT | None:
        """Get an attached trait by its type.

        Args:
            trait_type: The trait class to look up.

        Returns:
            The trait instance, or None if not attached.
        """
        return self._traits.get(trait_type)

    def has_trait(self, trait_type: type[BaseTrait]) -> bool:
        """Check if a trait is attached.

        Args:
            trait_type: The trait class to check.

        Returns:
            True if the trait is attached.
        """
        return self._traits.has(trait_type)

    def require_trait(self, trait_type: type[TraitT]) -> TraitT:
        """Get a required trait, raising if not attached.

        Args:
            trait_type: The trait class to look up.

        Returns:
            The trait instance.

        Raises:
            TraitNotFoundError: If the trait is not attached.
        """
        try:
            return self._traits.require(trait_type)
        except TraitNotFoundError as e:
            # Re-raise with agent-specific message
            raise TraitNotFoundError(
                f"{trait_type.__name__} required but not attached - "
                f"add it with agent.add_trait({trait_type.__name__}(...))"
            ) from e

    # =========================================================================
    # Trait Lifecycle Helpers
    # =========================================================================

    def _start_traits(self) -> None:
        """Start all attached traits. Call from start()."""
        for trait in self._traits.all():
            trait.on_start()
        # Only mark as started after all traits successfully initialize
        self._started = True

    def _stop_traits(self) -> None:
        """Stop all attached traits. Call from stop()."""
        for trait in self._traits.all():
            try:
                trait.on_stop()
            except Exception as e:
                self._lg.warning(
                    "error stopping trait",
                    extra={"trait": type(trait).__name__, "exception": e},
                )
        self._started = False

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    def _resolve_identity(self) -> Identity:
        """Resolve Identity from self.config.

        Returns:
            Constructed Identity instance.

        Raises:
            ConfigError: If identity.name is missing.
        """
        from ..errors import ConfigError
        from .identity import Identity

        # Check top-level identity first, then fall back to kelt.identity
        identity_config = self.config.get("identity", {})
        if not identity_config.get("name"):
            kelt_config = self.config.get("kelt", {})
            identity_config = kelt_config.get("identity", {})

        name = identity_config.get("name")
        if not name:
            raise ConfigError("identity.name is required in config")

        return Identity.from_config(identity_config, defaults=DotDict(name=name))
