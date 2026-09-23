# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Postgres-backed :class:`CheckpointStore` — content-addressed object + ref store.

Two tables, one row per object / ref respectively:

- :class:`FlowObject` — ``(client_flow_id, kind, content_hash)`` PK,
  ``BYTEA payload``. Idempotent puts via ``ON CONFLICT DO NOTHING``:
  content-hashing guarantees same-hash → same bytes, so a re-put is a
  no-op.
- :class:`FlowRef` — ``(client_flow_id, node_path, iteration)`` PK,
  ``VARCHAR(64) commit_hash``, ``TIMESTAMPTZ created_at``. Idempotent
  overwrite via ``ON CONFLICT DO UPDATE`` refreshing ``created_at`` so
  the latest ref is discoverable by ``ORDER BY created_at DESC``.

Both tables scope everything by ``client_flow_id`` — trajectory-scoped
storage, matching the arc's non-goal on cross-trajectory blob sharing.
:meth:`gc_trajectory` is two DELETE statements.

Schema is not managed by the store. Consumers call
:func:`llm_gent.ensure_schema` (or :class:`llm_gent.schema.SchemaManager`
directly) once at process start to bring the DB to head via alembic;
the store just uses the tables.
"""

from __future__ import annotations

from datetime import UTC, datetime

from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import (
    DateTime,
    Integer,
    LargeBinary,
    String,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, mapped_column

from llm_gent.schema import Base

from ..checkpoint import Kind, Retention


class FlowObject(Base):
    """One row = one content-addressed object under a trajectory.

    Kept in sync with :mod:`llm_gent.migrations.versions.001_initial_flow_checkpoint`
    (which owns the DDL). Any schema change lands as both a new alembic
    revision and a matching model edit.
    """

    __tablename__ = "llm_gent_flow_object"

    client_flow_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class FlowRef(Base):
    """One row = one ``(client_flow_id, node_path, iteration)`` → commit_hash.

    ``created_at`` is refreshed on every put so :meth:`resolve_ref` can
    return the newest ref across a trajectory via ``ORDER BY created_at
    DESC``.
    """

    __tablename__ = "llm_gent_flow_ref"

    client_flow_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    node_path: Mapped[str] = mapped_column(String(1024), primary_key=True)
    iteration: Mapped[int] = mapped_column(Integer, primary_key=True)
    commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class PgCheckpointStore:
    """Postgres-backed :class:`CheckpointStore` sharing one PG handle."""

    retention: Retention

    def __init__(
        self,
        lg: Logger,
        pg: PG,
        *,
        retention: Retention = "retain",
    ) -> None:
        """Bind a PG handle + retention policy.

        Args:
            lg: Logger for diagnostics.
            pg: :class:`appinfra.db.pg.PG` handle. The store issues all
                statements against this handle's bound schema.
            retention: ``"retain"`` (default) keeps successful
                trajectories in place; ``"gc_on_success"`` calls
                :meth:`gc_trajectory` on clean run completion.
        """
        self._lg = lg
        self._pg = pg
        self.retention = retention

    # ------------------------------------------------------------------
    # Object store
    # ------------------------------------------------------------------

    def put_object(
        self,
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
        payload: bytes,
    ) -> None:
        """Idempotent insert — ``ON CONFLICT DO NOTHING`` on the PK."""
        stmt = insert(FlowObject).values(
            client_flow_id=client_flow_id,
            kind=kind,
            content_hash=content_hash,
            payload=payload,
        )
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["client_flow_id", "kind", "content_hash"]
        )
        with self._pg.session() as session:
            session.execute(stmt)

    def get_object(
        self,
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bytes | None:
        """Return the payload bytes for the object, or ``None`` on miss."""
        stmt = select(FlowObject.payload).where(
            FlowObject.client_flow_id == client_flow_id,
            FlowObject.kind == kind,
            FlowObject.content_hash == content_hash,
        )
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else bytes(row[0])

    def has_object(
        self,
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bool:
        """Return ``True`` when the object row exists."""
        stmt = select(FlowObject.content_hash).where(
            FlowObject.client_flow_id == client_flow_id,
            FlowObject.kind == kind,
            FlowObject.content_hash == content_hash,
        )
        with self._pg.session() as session:
            return session.execute(stmt).first() is not None

    # ------------------------------------------------------------------
    # Ref store
    # ------------------------------------------------------------------

    def put_ref(
        self,
        client_flow_id: str,
        node_path: str,
        iteration: int,
        commit_hash: str,
    ) -> None:
        """Idempotent overwrite — refreshes ``created_at`` on re-put."""
        stmt = insert(FlowRef).values(
            client_flow_id=client_flow_id,
            node_path=node_path,
            iteration=iteration,
            commit_hash=commit_hash,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["client_flow_id", "node_path", "iteration"],
            set_={"commit_hash": stmt.excluded.commit_hash, "created_at": datetime.now(UTC)},
        )
        with self._pg.session() as session:
            session.execute(stmt)

    def resolve_ref(
        self,
        client_flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None:
        """Return the commit hash for the trajectory key, or ``None``.

        See :class:`~llm_gent.flow.checkpoint.CheckpointStore.resolve_ref`.
        """
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        stmt = select(FlowRef.commit_hash).where(FlowRef.client_flow_id == client_flow_id)
        if node_path is not None:
            stmt = stmt.where(FlowRef.node_path == node_path)
        if iteration is not None:
            stmt = stmt.where(FlowRef.iteration == iteration)
        else:
            stmt = stmt.order_by(FlowRef.created_at.desc()).limit(1)
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else str(row[0])

    # ------------------------------------------------------------------
    # Trajectory cleanup
    # ------------------------------------------------------------------

    def gc_trajectory(self, client_flow_id: str) -> None:
        """Delete every object and ref under ``client_flow_id``. Idempotent."""
        with self._pg.session() as session:
            session.execute(delete(FlowRef).where(FlowRef.client_flow_id == client_flow_id))
            session.execute(delete(FlowObject).where(FlowObject.client_flow_id == client_flow_id))
