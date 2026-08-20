"""Storage trait for agent-defined relational schemas.

Provides AgentStorage for defining and querying custom tables using SQLAlchemy directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....storage import AgentStorage
from ..base import BaseTrait


if TYPE_CHECKING:
    from ...agent import Agent


class StorageTrait(BaseTrait):
    """Provides AgentStorage for custom relational schemas.

    Wraps KeltClient to provide direct SQLAlchemy access for agent tables
    with automatic isolation.

    **IMPORTANT:** StorageTrait depends on MemoryTrait. MemoryTrait must be attached
    to the agent BEFORE StorageTrait, as StorageTrait.on_start() requires MemoryTrait
    to be available. Ensure correct ordering in factory required_traits or YAML config.

    Philosophy:
        - Use SQLAlchemy directly (no abstraction layer)
        - Automatic isolation via context_key
        - Full SQL power for complex queries

    Example:
        from llm_gent.core.traits import StorageTrait, MemoryTrait
        from llm_gent.storage import AgentTable
        from sqlalchemy import String, Boolean, Float, func, select
        from sqlalchemy.orm import Mapped, mapped_column

        # Define schema
        class JokeTable(AgentTable):
            __tablename__ = "agent_jokester_jokes"

            text: Mapped[str] = mapped_column(String, nullable=False)
            style: Mapped[str] = mapped_column(String(50), nullable=False)
            rated: Mapped[bool] = mapped_column(Boolean, default=False)
            rating: Mapped[float | None] = mapped_column(Float, nullable=True)

        # Attach traits
        agent.add_trait(MemoryTrait(agent, memory_config))
        agent.add_trait(StorageTrait(agent))
        agent.start()

        # Get storage
        storage = agent.require_trait(StorageTrait).storage

        # Register table
        storage.register_table(JokeTable)

        # Simple queries (auto-isolated, returns results immediately)
        unrated = storage.select(JokeTable, rated=False)

        # Complex queries (raw SQLAlchemy)
        stmt = select(
            JokeTable.style,
            func.avg(JokeTable.rating).label('avg_rating')
        ).where(
            JokeTable.context_key == storage.context_key,
            JokeTable.rated == True
        ).group_by(JokeTable.style)

        results = storage.execute(stmt).all()

    Lifecycle:
        - on_start(): Creates AgentStorage from MemoryTrait's KeltClient
        - on_stop(): Cleanup (no-op, AgentStorage is stateless)
    """

    def __init__(self, agent: Agent) -> None:
        """Initialize storage trait.

        Args:
            agent: The agent this trait belongs to.
        """
        super().__init__(agent)
        self._storage: AgentStorage | None = None

    def on_start(self) -> None:
        """Create AgentStorage from MemoryTrait's KeltClient.

        Raises:
            TraitNotFoundError: If MemoryTrait is not attached.
        """
        from .memory import MemoryTrait

        # Require MemoryTrait
        memory_trait = self.agent.require_trait(MemoryTrait)

        # Create storage wrapping KeltClient
        self._storage = AgentStorage(self.agent.lg, memory_trait.kelt)

        self.agent.lg.debug("storage trait started")

    def on_stop(self) -> None:
        """Clean up storage resources."""
        self._storage = None
        self.agent.lg.debug("storage trait stopped")

    @property
    def storage(self) -> AgentStorage:
        """Access the AgentStorage instance.

        Provides direct SQLAlchemy access for agent-defined tables.

        Returns:
            AgentStorage for registering and querying agent tables.

        Raises:
            RuntimeError: If trait not started.
        """
        if self._storage is None:
            raise RuntimeError("StorageTrait not started - ensure agent.start() was called")
        return self._storage
