"""Schema definitions for agent-defined storage.

Provides AgentTable base class that all agent tables inherit from.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

from llm_kelt.core.base import Base
from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(UTC)


class AgentTable(Base):
    """Base class for all agent-defined tables.

    Provides standard columns that all agent tables must have:
    - Auto-incrementing ID
    - Isolation filter column (context_key)
    - created_at/updated_at timestamps
    - Index on isolation column

    Agents inherit from this class to define custom schemas.

    Example:
        from sqlalchemy import String, Boolean, Float
        from sqlalchemy.orm import Mapped, mapped_column

        class JokeTable(AgentTable):
            __tablename__ = "agent_jokester_jokes"

            text: Mapped[str] = mapped_column(String, nullable=False)
            style: Mapped[str] = mapped_column(String(50), nullable=False)
            rated: Mapped[bool] = mapped_column(Boolean, default=False)
            rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    """

    __abstract__ = True

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    # Isolation column (context_key from llm-kelt ClientContext)
    # Supports hierarchical keys like "domain:workspace:name"
    context_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=utc_now,
        nullable=True,
    )


def validate_agent_table(model_class: type) -> None:
    """Validate that a model class meets AgentTable requirements.

    Args:
        model_class: The model class to validate.

    Raises:
        ValueError: If model class doesn't meet requirements.
    """
    if not issubclass(model_class, AgentTable):
        raise ValueError(f"{model_class.__name__} must inherit from AgentTable")

    if not hasattr(model_class, "__tablename__"):
        raise ValueError(f"{model_class.__name__} must define __tablename__")

    table_name = model_class.__tablename__
    if not table_name.startswith("agent_"):
        warnings.warn(
            f"Table name '{table_name}' doesn't follow agent_* convention. "
            "Consider prefixing with 'agent_' to avoid collisions.",
            stacklevel=2,
        )
