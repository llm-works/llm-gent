# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Postgres-backed :class:`CheckpointStore` — content-addressed object + ref store.

Four tables:

- :class:`FlowName` — ``client_flow_id`` PK → ``flow_id`` (unique): the
  agent's name for a history mapped to gent's internal id.
- :class:`FlowObject` — ``(flow_id, kind, content_hash)`` PK,
  ``BYTEA payload``. Idempotent puts via ``ON CONFLICT DO NOTHING``:
  content-hashing guarantees same-hash → same bytes, so a re-put is a
  no-op.
- :class:`FlowRef` — ``(flow_id, node_path, iteration)`` PK,
  ``VARCHAR(64) commit_hash``, ``BIGINT seq``, ``TIMESTAMPTZ
  created_at``. Idempotent overwrite via ``ON CONFLICT DO UPDATE``
  drawing a new ``seq`` from a database sequence, so the latest ref is
  ``ORDER BY seq DESC`` — write order independent of writer clocks.
- :class:`FlowTag` — ``(flow_id, name)`` PK → ``commit_hash``. Upsert
  moves the tag.

Object, ref and tag tables scope everything by ``flow_id`` —
history-scoped storage; blobs are deliberately not shared across
histories. :meth:`gc_history` is four DELETE statements.

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
    BigInteger,
    DateTime,
    Integer,
    LargeBinary,
    Sequence,
    String,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from llm_gent.schema import Base

from ..checkpoint import Kind, Retention


_FLOW_ID_LEN = 36
"""Length of a ``flow_id`` column — a canonical UUID string."""

_REF_SEQ = Sequence("gent_flow_ref_seq", metadata=Base.metadata)
"""Sequence feeding :attr:`FlowRef.seq` (write order of refs)."""


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
    """One row = one ``(flow_id, node_path, iteration)`` → commit_hash.

    ``seq`` is drawn from :data:`_REF_SEQ` on insert and redrawn on every
    re-put, so :meth:`resolve_ref` returns the newest ref across a history
    via ``ORDER BY seq DESC`` — database write order, not writer clocks.
    """

    __tablename__ = "gent_flow_ref"

    flow_id: Mapped[str] = mapped_column(String(_FLOW_ID_LEN), primary_key=True)
    node_path: Mapped[str] = mapped_column(String(1024), primary_key=True)
    iteration: Mapped[int] = mapped_column(Integer, primary_key=True)
    commit_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    seq: Mapped[int] = mapped_column(
        BigInteger, _REF_SEQ, nullable=False, server_default=_REF_SEQ.next_value()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class FlowTag(Base):
    """One row = one named tag ``(flow_id, name)`` → commit_hash; re-put moves it."""

    __tablename__ = "gent_flow_tag"

    flow_id: Mapped[str] = mapped_column(String(_FLOW_ID_LEN), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), primary_key=True)
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
    # Ref store
    # ------------------------------------------------------------------

    def put_ref(
        self,
        flow_id: str,
        node_path: str,
        iteration: int,
        commit_hash: str,
    ) -> None:
        """Idempotent overwrite — a re-put draws a new ``seq`` and refreshes ``created_at``."""
        stmt = insert(FlowRef).values(
            flow_id=flow_id,
            node_path=node_path,
            iteration=iteration,
            commit_hash=commit_hash,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["flow_id", "node_path", "iteration"],
            set_={
                "commit_hash": stmt.excluded.commit_hash,
                "seq": _REF_SEQ.next_value(),
                "created_at": datetime.now(UTC),
            },
        )
        with self._pg.session() as session:
            session.execute(stmt)

    def resolve_ref(
        self,
        flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None:
        """Return the commit hash for the history key, or ``None``.

        See :class:`~llm_gent.flow.checkpoint.CheckpointStore.resolve_ref`.
        """
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        stmt = select(FlowRef.commit_hash).where(FlowRef.flow_id == flow_id)
        if node_path is not None:
            stmt = stmt.where(FlowRef.node_path == node_path)
        if iteration is not None:
            stmt = stmt.where(FlowRef.iteration == iteration)
        elif node_path is not None:
            # Latest under a specific node_path = highest iteration, not
            # newest write — matches the JsonFile helper's semantics and
            # the Protocol contract.
            stmt = stmt.order_by(FlowRef.iteration.desc()).limit(1)
        else:
            # Latest across the whole history = newest write, by database sequence.
            stmt = stmt.order_by(FlowRef.seq.desc()).limit(1)
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else str(row[0])

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def put_tag(self, flow_id: str, name: str, commit_hash: str) -> None:
        """Upsert — re-put moves the tag and refreshes ``created_at``."""
        stmt = insert(FlowTag).values(flow_id=flow_id, name=name, commit_hash=commit_hash)
        stmt = stmt.on_conflict_do_update(
            index_elements=["flow_id", "name"],
            set_={"commit_hash": stmt.excluded.commit_hash, "created_at": datetime.now(UTC)},
        )
        with self._pg.session() as session:
            session.execute(stmt)

    def resolve_tag(self, flow_id: str, name: str) -> str | None:
        """Return the commit hash tag ``name`` points at, or ``None``."""
        stmt = select(FlowTag.commit_hash).where(FlowTag.flow_id == flow_id, FlowTag.name == name)
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else str(row[0])

    # ------------------------------------------------------------------
    # History cleanup
    # ------------------------------------------------------------------

    def gc_history(self, flow_id: str) -> None:
        """Delete every object, ref and tag under ``flow_id`` and its name binding. Idempotent."""
        with self._pg.session() as session:
            session.execute(delete(FlowTag).where(FlowTag.flow_id == flow_id))
            session.execute(delete(FlowRef).where(FlowRef.flow_id == flow_id))
            session.execute(delete(FlowObject).where(FlowObject.flow_id == flow_id))
            session.execute(delete(FlowName).where(FlowName.flow_id == flow_id))
