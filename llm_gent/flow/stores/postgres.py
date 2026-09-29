# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Postgres-backed :class:`CheckpointStore` — content-addressed object + ref store.

Three tables:

- :class:`FlowName` — ``client_flow_id`` PK → ``flow_id`` (unique): the
  agent's name for a history mapped to gent's internal id.
- :class:`FlowObject` — ``(flow_id, kind, content_hash)`` PK,
  ``BYTEA payload``. Idempotent puts via ``ON CONFLICT DO NOTHING``:
  content-hashing guarantees same-hash → same bytes, so a re-put is a
  no-op.
- :class:`FlowRef` — ``(flow_id, name)`` PK → ``commit_hash``. Created
  with ``INSERT ... ON CONFLICT DO NOTHING`` and moved with ``UPDATE ...
  WHERE commit_hash = <expected>``: each is one atomic statement, so of
  concurrent compare-and-set writers at most one lands.

Object and ref tables scope everything by ``flow_id`` — history-scoped
storage; blobs are deliberately not shared across histories.
:meth:`gc_history` is three DELETE statements.

Schema is not managed by the store. Consumers call
:func:`llm_gent.ensure_schema` (or :class:`llm_gent.schema.SchemaManager`
directly) once at process start to bring the DB to head via alembic;
the store just uses the tables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import DateTime, LargeBinary, String, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from llm_gent.schema import Base

from ..checkpoint import Kind, Retention


_FLOW_ID_LEN = 36
"""Length of a ``flow_id`` column — a canonical UUID string."""


class FlowName(Base):
    """One row = one agent-chosen ``client_flow_id`` bound to its history's ``flow_id``.

    Kept in sync with :mod:`llm_gent.migrations.versions.001_initial_flow_cas`
    (which owns the DDL), like the other models below.
    """

    __tablename__ = "gent_flow_name"

    client_flow_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    flow_id: Mapped[str] = mapped_column(String(_FLOW_ID_LEN), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class FlowObject(Base):
    """One row = one content-addressed object under a history.

    Kept in sync with :mod:`llm_gent.migrations.versions.001_initial_flow_cas`
    (which owns the DDL). Any schema change lands as both a new alembic
    revision and a matching model edit.
    """

    __tablename__ = "gent_flow_object"

    flow_id: Mapped[str] = mapped_column(String(_FLOW_ID_LEN), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class FlowRef(Base):
    """One row = one named ref ``(flow_id, name)`` → commit_hash (``HEAD``, ``tags/...``)."""

    __tablename__ = "gent_flow_ref"

    flow_id: Mapped[str] = mapped_column(String(_FLOW_ID_LEN), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
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
                histories in place; ``"gc_on_success"`` calls
                :meth:`gc_history` on clean run completion.
        """
        self._lg = lg
        self._pg = pg
        self.retention = retention

    # ------------------------------------------------------------------
    # Name map
    # ------------------------------------------------------------------

    def get_flow_id(self, client_flow_id: str) -> str | None:
        """Return the ``flow_id`` bound to ``client_flow_id``, or ``None``."""
        stmt = select(FlowName.flow_id).where(FlowName.client_flow_id == client_flow_id)
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else str(row[0])

    def bind_flow_id(self, client_flow_id: str, flow_id: str) -> str:
        """Bind ``client_flow_id`` → ``flow_id`` unless already bound; return the bound id.

        ``ON CONFLICT (client_flow_id) DO NOTHING`` waits for a concurrent
        binder to commit, so the follow-up read returns the winner. A
        ``flow_id`` already naming another history violates its unique
        constraint and raises :class:`ValueError`.
        """
        stmt = (
            insert(FlowName)
            .values(client_flow_id=client_flow_id, flow_id=flow_id)
            .on_conflict_do_nothing(index_elements=["client_flow_id"])
        )
        query = select(FlowName.flow_id).where(FlowName.client_flow_id == client_flow_id)
        try:
            with self._pg.session() as session:
                session.execute(stmt)
                bound = session.execute(query).scalar_one()
        except IntegrityError as e:
            raise ValueError(f"flow_id {flow_id!r} already names another history") from e
        return str(bound)

    # ------------------------------------------------------------------
    # Object store
    # ------------------------------------------------------------------

    def put_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
        payload: bytes,
    ) -> None:
        """Idempotent insert — ``ON CONFLICT DO NOTHING`` on the PK."""
        stmt = insert(FlowObject).values(
            flow_id=flow_id,
            kind=kind,
            content_hash=content_hash,
            payload=payload,
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["flow_id", "kind", "content_hash"])
        with self._pg.session() as session:
            session.execute(stmt)

    def get_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bytes | None:
        """Return the payload bytes for the object, or ``None`` on miss."""
        stmt = select(FlowObject.payload).where(
            FlowObject.flow_id == flow_id,
            FlowObject.kind == kind,
            FlowObject.content_hash == content_hash,
        )
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else bytes(row[0])

    def has_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bool:
        """Return ``True`` when the object row exists."""
        stmt = select(FlowObject.content_hash).where(
            FlowObject.flow_id == flow_id,
            FlowObject.kind == kind,
            FlowObject.content_hash == content_hash,
        )
        with self._pg.session() as session:
            return session.execute(stmt).first() is not None

    # ------------------------------------------------------------------
    # Refs
    # ------------------------------------------------------------------

    def get_ref(self, flow_id: str, name: str) -> str | None:
        """Return the commit hash ref ``name`` points at, or ``None``."""
        stmt = select(FlowRef.commit_hash).where(FlowRef.flow_id == flow_id, FlowRef.name == name)
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else str(row[0])

    def set_ref(self, flow_id: str, name: str, commit_hash: str, expected: str | None) -> bool:
        """Point ref ``name`` at ``commit_hash`` if it points at ``expected`` (``None``: absent).

        One statement either way — ``INSERT ... ON CONFLICT DO NOTHING`` to
        create, ``UPDATE ... WHERE commit_hash = expected`` to move — so
        the row count says whether this writer won.
        """
        stmt: Any
        if expected is None:
            stmt = (
                insert(FlowRef)
                .values(flow_id=flow_id, name=name, commit_hash=commit_hash)
                .on_conflict_do_nothing(index_elements=["flow_id", "name"])
            )
        else:
            stmt = (
                update(FlowRef)
                .where(
                    FlowRef.flow_id == flow_id,
                    FlowRef.name == name,
                    FlowRef.commit_hash == expected,
                )
                .values(commit_hash=commit_hash, updated_at=datetime.now(UTC))
            )
        with self._pg.session() as session:
            result = cast(CursorResult[Any], session.execute(stmt))
        return bool(result.rowcount == 1)

    # ------------------------------------------------------------------
    # History cleanup
    # ------------------------------------------------------------------

    def gc_history(self, flow_id: str) -> None:
        """Delete every object and ref under ``flow_id`` and its name binding. Idempotent."""
        with self._pg.session() as session:
            session.execute(delete(FlowRef).where(FlowRef.flow_id == flow_id))
            session.execute(delete(FlowObject).where(FlowObject.flow_id == flow_id))
            session.execute(delete(FlowName).where(FlowName.flow_id == flow_id))
