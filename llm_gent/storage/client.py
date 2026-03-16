"""Agent storage client for custom relational schemas.

Provides AgentStorage for registering and querying agent-defined tables.
Uses SQLAlchemy directly for maximum flexibility while ensuring automatic isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from .schema import AgentTable, validate_agent_table


if TYPE_CHECKING:
    from llm_kelt import Client as KeltClient


class AgentStorage:
    """
    Storage client for agent-defined relational data.

    Wraps Client to provide:
    - Table registration with automatic isolation
    - Query helpers with automatic context_key filtering
    - Direct SQLAlchemy access for complex queries

    Philosophy:
    - Use SQLAlchemy directly (no leaky abstractions)
    - Provide safety helpers for common patterns
    - Automatic isolation by default
    - Escape hatch for power users

    Example:
        from sqlalchemy import String, Boolean, Float, select, func
        from sqlalchemy.orm import Mapped, mapped_column
        from llm_gent.storage import AgentTable

        # Define schema
        class JokeTable(AgentTable):
            __tablename__ = "agent_jokester_jokes"

            text: Mapped[str] = mapped_column(String, nullable=False)
            style: Mapped[str] = mapped_column(String(50), nullable=False)
            rated: Mapped[bool] = mapped_column(Boolean, default=False)
            rating: Mapped[float | None] = mapped_column(Float, nullable=True)

        # Register and use
        agent.storage.register_table(JokeTable)

        # Simple queries (auto-isolated, returns results immediately)
        unrated = agent.storage.select(JokeTable, rated=False)

        # Insert (auto-adds context_key)
        joke_id = agent.storage.insert(JokeTable,
            text="Why did the AI cross the road?",
            style="observational",
            rated=False
        )

        # Complex queries (raw SQLAlchemy with manual isolation)
        stmt = select(
            JokeTable.style,
            func.avg(JokeTable.rating).label('avg_rating')
        ).where(
            JokeTable.context_key == agent.storage.context_key,
            JokeTable.rated == True
        ).group_by(JokeTable.style)

        results = agent.storage.execute(stmt).all()
    """

    def __init__(self, lg: Logger, kelt_client: KeltClient) -> None:
        """Initialize agent storage.

        Args:
            lg: Logger instance.
            kelt_client: KeltClient instance (provides database + isolation context).

        Raises:
            ValueError: If kelt_client's isolation context has no context_key.
        """
        self._lg = lg
        self._kelt = kelt_client
        self._registered_tables: dict[str, type[AgentTable]] = {}

        # Validate that context_key is set (required for isolation)
        if self._kelt.context.context_key is None:
            raise ValueError(
                "AgentStorage requires isolation context with context_key set. "
                "Ensure Client was created with valid ClientContext."
            )

    @property
    def context_key(self) -> str:
        """Get isolation context key.

        Returns:
            The context key for data isolation (guaranteed to be set).
        """
        return self._kelt.context.context_key  # type: ignore[return-value]

    @property
    def schema_name(self) -> str | None:
        """Get schema name from isolation context."""
        return self._kelt.context.schema_name

    def register_table(self, model_class: type[AgentTable]) -> None:
        """Register an agent-defined table schema.

        Validates the model and creates the table if it doesn't exist.
        Tables are created in public schema; isolation is via context_key column.

        Args:
            model_class: SQLAlchemy model inheriting from AgentTable.

        Raises:
            ValueError: If model_class doesn't meet requirements.
        """
        validate_agent_table(model_class)

        table_name = model_class.__tablename__

        # Create table in default schema (isolation via context_key, not schema)
        engine = self._kelt.database.engine
        model_class.__table__.create(engine, checkfirst=True)  # type: ignore[attr-defined]

        self._registered_tables[table_name] = model_class

        self._lg.debug(
            "agent table registered",
            extra={"table": table_name, "context_key": self.context_key},
        )

    def select(self, model_class: type[AgentTable], **filters: Any) -> list[AgentTable]:
        """Query rows from an agent table with automatic isolation.

        Returns rows immediately (not a lazy query object). For complex queries,
        use execute() with SQLAlchemy select() statements.

        Args:
            model_class: The model class to query.
            **filters: Column filters (e.g., rated=False, style="pun").

        Returns:
            List of matching rows.

        Raises:
            ValueError: If table not registered.

        Example:
            # Simple select with filters
            unrated_jokes = agent.storage.select(JokeTable, rated=False)

            # Select all
            all_jokes = agent.storage.select(JokeTable)

            # For complex queries, use execute() instead
            from sqlalchemy import select
            stmt = select(JokeTable).where(
                JokeTable.context_key == agent.storage.context_key,
                JokeTable.rated == False
            ).order_by(JokeTable.created_at.desc()).limit(10)
            results = agent.storage.execute(stmt)  # Returns list directly
        """
        table_name = model_class.__tablename__
        if table_name not in self._registered_tables:
            raise ValueError(f"Table not registered: {table_name}")

        # Build select statement with automatic isolation
        from sqlalchemy import select

        stmt = select(model_class).where(model_class.context_key == self.context_key)

        # Apply additional filters
        for key, value in filters.items():
            if not hasattr(model_class, key):
                raise ValueError(
                    f"Invalid column name '{key}' for table {table_name}. "
                    f"Column does not exist on model."
                )
            stmt = stmt.where(getattr(model_class, key) == value)

        with self._kelt.database.session() as session:
            return list(session.execute(stmt).scalars().all())

    def insert(self, model_class: type[AgentTable], **values: Any) -> int:
        """Insert a row into an agent table.

        Automatically adds context_key from isolation context.

        Args:
            model_class: The model class to insert into.
            **values: Column values.

        Returns:
            Inserted row ID.

        Raises:
            ValueError: If table not registered.

        Example:
            joke_id = agent.storage.insert(JokeTable,
                text="Why did the chicken cross the road?",
                style="classic",
                rated=False
            )
        """
        table_name = model_class.__tablename__
        if table_name not in self._registered_tables:
            raise ValueError(f"Table not registered: {table_name}")

        # Add context_key automatically
        values["context_key"] = self.context_key

        with self._kelt.database.session() as session:
            record = model_class(**values)
            session.add(record)
            session.flush()
            row_id = record.id
            session.commit()
            return row_id

    def execute(self, stmt: Any) -> list[Any]:
        """Execute a raw SQLAlchemy statement.

        For complex queries that need full SQL power.
        Does NOT auto-apply isolation - caller must include context_key filter.

        Args:
            stmt: SQLAlchemy statement (select, update, delete, etc.).

        Returns:
            Materialized list of results (not a live Result object).

        Example:
            from sqlalchemy import select, func

            stmt = select(
                JokeTable.style,
                func.avg(JokeTable.rating).label('avg_rating')
            ).where(
                JokeTable.context_key == agent.storage.context_key,  # Manual filtering
                JokeTable.rated == True
            ).group_by(JokeTable.style)

            results = agent.storage.execute(stmt)  # Returns list directly
        """
        with self._kelt.database.session() as session:
            result = session.execute(stmt)
            return list(result.all())

    def update(self, model_class: type[AgentTable], row_id: int, **values: Any) -> None:
        """Update a row by ID.

        Args:
            model_class: The model class.
            row_id: ID of the row to update.
            **values: Column values to update.

        Raises:
            ValueError: If table not registered, row not found, or attempting to update context_key.

        Example:
            agent.storage.update(JokeTable, joke_id, rated=True, rating=5)
        """
        table_name = model_class.__tablename__
        if table_name not in self._registered_tables:
            raise ValueError(f"Table not registered: {table_name}")

        # Prevent overwriting isolation field
        if "context_key" in values:
            raise ValueError(
                "Cannot update isolation field 'context_key'. "
                "This would bypass data isolation and is not allowed."
            )

        from sqlalchemy import select

        with self._kelt.database.session() as session:
            # Query with isolation (context_key required)
            stmt = select(model_class).where(
                model_class.id == row_id, model_class.context_key == self.context_key
            )
            record = session.execute(stmt).scalar_one_or_none()

            if record is None:
                raise ValueError(f"Row not found: {row_id}")

            # Update fields
            for key, value in values.items():
                setattr(record, key, value)

            session.commit()

    def delete(self, model_class: type[AgentTable], row_id: int) -> None:
        """Delete a row by ID.

        Args:
            model_class: The model class.
            row_id: ID of the row to delete.

        Raises:
            ValueError: If table not registered or row not found.

        Example:
            agent.storage.delete(JokeTable, joke_id)
        """
        table_name = model_class.__tablename__
        if table_name not in self._registered_tables:
            raise ValueError(f"Table not registered: {table_name}")

        from sqlalchemy import select

        with self._kelt.database.session() as session:
            # Query with isolation (context_key required)
            stmt = select(model_class).where(
                model_class.id == row_id, model_class.context_key == self.context_key
            )
            record = session.execute(stmt).scalar_one_or_none()

            if record is None:
                raise ValueError(f"Row not found: {row_id}")

            session.delete(record)
            session.commit()
