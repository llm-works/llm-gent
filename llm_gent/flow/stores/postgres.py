# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Postgres-backed :class:`CheckpointStore` via the appinfra PG interface.

Flat single-table layout — one row per ``(client_flow_id, node_path,
iteration)``, JSONB payloads for both the framework snapshot and
metadata. Latest-load across every ``node_path`` is
``ORDER BY created_at DESC LIMIT 1`` under the trajectory id. Save is
an upsert (``ON CONFLICT ... DO UPDATE``) that refreshes ``created_at``
so re-saves move to the head of the total order, per the Protocol's
idempotent-overwrite contract.

Schema is not managed by the store. Consumers call
:func:`llm_gent.ensure_schema` (or :class:`llm_gent.schema.SchemaManager`
directly) once at process start to bring the DB to head via alembic;
the store just uses the table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.orm import Mapped, mapped_column

from llm_gent.schema import Base


class FlowCheckpoint(Base):
    """One row = one saved iteration under a ``client_flow_id`` trajectory.

    Kept in sync with :mod:`llm_gent.migrations.versions.001_initial_flow_checkpoint`:
    alembic is the DDL source of truth for prod upgrades, but this model
    is what :meth:`llm_gent.schema.SchemaManager._bootstrap_fresh_database`'s
    ``create_all`` path emits AND what SA references by name in the store's
    ``on_conflict_do_update`` upsert. Any schema change lands as both a
    new alembic revision and a matching model edit.
    """

    __tablename__ = "llm_gent_flow_checkpoint"

    db_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    client_flow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    node_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        UniqueConstraint(
            "client_flow_id",
            "node_path",
            "iteration",
            name="uq_flow_checkpoint_trajectory_iter",
        ),
        Index("ix_flow_checkpoint_trajectory_iter", "client_flow_id", "node_path", "iteration"),
    )


class PgCheckpointStore:
    """Postgres-backed :class:`CheckpointStore` sharing one PG handle."""

    def __init__(
        self,
        lg: Logger,
        pg: PG,
    ) -> None:
        """Bind a PG handle.

        Args:
            lg: Logger for load-time diagnostics.
            pg: :class:`appinfra.db.pg.PG` handle. The store issues all
                statements against this handle's bound schema; passing
                a ``schema=``-isolated PG scopes the checkpoint table
                to that schema for tests / multi-tenancy. The schema's
                DDL must have been brought current via
                :func:`llm_gent.ensure_schema` beforehand.
        """
        self._lg = lg
        self._pg = pg

    def save_checkpoint(
        self,
        client_flow_id: str,
        node_path: str,
        iteration: int,
        state_json: dict[str, Any],
        metadata_json: dict[str, Any],
    ) -> None:
        """Upsert one record per the idempotent-overwrite contract."""
        stmt = insert(FlowCheckpoint).values(
            client_flow_id=client_flow_id,
            node_path=node_path,
            iteration=iteration,
            state_json=state_json,
            metadata_json=metadata_json,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_flow_checkpoint_trajectory_iter",
            set_={
                "state_json": stmt.excluded.state_json,
                "metadata_json": stmt.excluded.metadata_json,
                "created_at": datetime.now(UTC),
            },
        )
        with self._pg.session() as session:
            session.execute(stmt)

    def load_checkpoint(
        self,
        client_flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Read one record; latest across all ``node_path`` when both filters are ``None``.

        Latest = most recent ``created_at`` (refreshed on every save,
        including re-saves that upsert an existing row).
        """
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        stmt = select(FlowCheckpoint.state_json, FlowCheckpoint.metadata_json).where(
            FlowCheckpoint.client_flow_id == client_flow_id
        )
        if node_path is not None:
            stmt = stmt.where(FlowCheckpoint.node_path == node_path)
        if iteration is not None:
            stmt = stmt.where(FlowCheckpoint.iteration == iteration)
        else:
            stmt = stmt.order_by(FlowCheckpoint.created_at.desc()).limit(1)
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        if row is None:
            return None
        return row[0], row[1]

    def delete_checkpoint(self, client_flow_id: str) -> None:
        """Remove every row under this trajectory. Idempotent."""
        stmt = delete(FlowCheckpoint).where(FlowCheckpoint.client_flow_id == client_flow_id)
        with self._pg.session() as session:
            session.execute(stmt)
