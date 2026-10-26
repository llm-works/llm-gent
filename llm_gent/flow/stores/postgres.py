# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Postgres-backed :class:`appinfra.cas.Store` — repositories of objects and refs in two tables.

- :class:`CasObject` — ``(repo, id)`` PK, the object's ``type``, its
  serialized ``data`` and ``written_at``. A re-put refreshes
  ``written_at`` (``ON CONFLICT DO UPDATE``), so a prune treats the object
  as recent again.
- :class:`CasRef` — ``(repo, name)`` PK → ``target``. Created with
  ``INSERT ... ON CONFLICT DO NOTHING``, moved with ``UPDATE ... WHERE
  target = <old>`` and deleted with ``DELETE ... WHERE target = <old>``:
  each is one statement, so of concurrent compare-and-set writers at most
  one lands.

Write times come from the database server's clock (``clock_timestamp()``),
and :meth:`PgStore.now` reads the same clock, so prune's grace window does
not shift with clock skew between hosts.

:meth:`PgStore.lock` takes a session-level advisory lock (shared or
exclusive) keyed by the current schema and the repository name, on a
dedicated autocommit connection that never sits idle in a transaction;
a process that dies releases it with its connection.

Schema is not managed by the store. Consumers call
:func:`llm_gent.ensure_schema` (or :class:`llm_gent.schema.SchemaManager`
directly) once at process start to bring the DB to head via alembic;
the store just uses the tables.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, cast

from appinfra.cas import ObjectType, Store
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import (
    DateTime,
    LargeBinary,
    String,
    delete,
    extract,
    func,
    select,
    text,
    union,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Mapped, mapped_column

from llm_gent.schema import Base


_DELETE_BATCH = 500
"""Object ids per DELETE statement in :meth:`PgStore.delete_objects`."""

_LOCK_KEY = "hashtextextended(current_schema() || '/' || :repo, 0)"
"""Advisory lock key of a repository: the schema in use and the repository name.

A schema name has no ``/``, so the first one separates the two.
"""


class CasObject(Base):
    """One row = one object of a repository.

    Kept in sync with :mod:`llm_gent.migrations.versions.002_appinfra_cas`
    (which owns the DDL), like :class:`CasRef`.
    """

    __tablename__ = "gent_cas_object"

    repo: Mapped[str] = mapped_column(String(255), primary_key=True)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[str] = mapped_column(String(8), nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    written_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.clock_timestamp()
    )


class CasRef(Base):
    """One row = one ref ``(repo, name)`` pointing at an object id (``HEAD``, ``tags/...``)."""

    __tablename__ = "gent_cas_ref"

    repo: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    target: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.clock_timestamp()
    )


class PgStore(Store):
    """Repositories of objects and refs in Postgres, through one :class:`~appinfra.db.pg.PG`."""

    def __init__(self, lg: Logger, pg: PG) -> None:
        """Bind a PG handle; every statement runs against its schema."""
        super().__init__(lg)
        self._pg = pg

    # --- locks and repositories ---

    @contextmanager
    def lock(self, repo: str, exclusive: bool = False) -> Iterator[None]:
        """Hold the repository's advisory lock, shared or exclusive, for the block."""
        mode = "" if exclusive else "_shared"
        with self._pg.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"SELECT pg_advisory_lock{mode}({_LOCK_KEY})"), {"repo": repo})
            try:
                yield
            finally:
                conn.execute(text(f"SELECT pg_advisory_unlock{mode}({_LOCK_KEY})"), {"repo": repo})

    def delete_repo(self, name: str) -> None:
        """Delete the repository ``name`` under its exclusive lock; no-op if absent."""
        with self.lock(name, exclusive=True), self._pg.session() as session:
            refs = session.execute(delete(CasRef).where(CasRef.repo == name))
            objects = session.execute(delete(CasObject).where(CasObject.repo == name))
        if _rowcount(refs) or _rowcount(objects):
            self._lg.debug("deleted repository", extra={"repo": name})

    def list_repos(self) -> list[str]:
        """Return the names of the repositories holding an object or a ref, sorted."""
        stmt = union(select(CasObject.repo), select(CasRef.repo))
        with self._pg.session() as session:
            return sorted(str(row[0]) for row in session.execute(stmt))

    def now(self) -> float:
        """The database server's clock (epoch secs): the one ``written_at`` uses."""
        with self._pg.session() as session:
            return float(
                session.execute(select(extract("epoch", func.clock_timestamp()))).scalar_one()
            )

    # --- objects ---

    def read_object(self, repo: str, object_id: str) -> tuple[ObjectType, bytes] | None:
        """Return the type and serialized bytes of an object, or ``None``."""
        stmt = select(CasObject.type, CasObject.data).where(
            CasObject.repo == repo, CasObject.id == object_id
        )
        with self._pg.session() as session:
            row = session.execute(stmt).first()
        return None if row is None else (cast(ObjectType, row[0]), bytes(row[1]))

    def object_type(self, repo: str, object_id: str) -> ObjectType | None:
        """Return the type of an object, or ``None``."""
        stmt = select(CasObject.type).where(CasObject.repo == repo, CasObject.id == object_id)
        with self._pg.session() as session:
            found = session.execute(stmt).scalar_one_or_none()
        return None if found is None else cast(ObjectType, found)

    def write_object(self, repo: str, object_id: str, object_type: ObjectType, data: bytes) -> None:
        """Store an object; rewriting an existing one refreshes its write time."""
        stmt = (
            insert(CasObject)
            .values(repo=repo, id=object_id, type=object_type, data=data)
            .on_conflict_do_update(
                index_elements=["repo", "id"], set_={"written_at": func.clock_timestamp()}
            )
        )
        with self._pg.session() as session:
            session.execute(stmt)

    def list_objects(self, repo: str) -> dict[str, float]:
        """Return every object id of the repository with its last write time (epoch secs)."""
        stmt = select(CasObject.id, extract("epoch", CasObject.written_at)).where(
            CasObject.repo == repo
        )
        with self._pg.session() as session:
            return {str(oid): float(at) for oid, at in session.execute(stmt)}

    def delete_objects(self, repo: str, object_ids: Iterable[str]) -> None:
        """Delete the given objects in one transaction; missing ids are skipped."""
        ids = list(object_ids)
        with self._pg.session() as session:
            for start in range(0, len(ids), _DELETE_BATCH):
                batch = ids[start : start + _DELETE_BATCH]
                session.execute(
                    delete(CasObject).where(CasObject.repo == repo, CasObject.id.in_(batch))
                )

    # --- refs ---

    def read_ref(self, repo: str, name: str) -> str | None:
        """Return the object id ref ``name`` points at, or ``None``."""
        stmt = select(CasRef.target).where(CasRef.repo == repo, CasRef.name == name)
        with self._pg.session() as session:
            found = session.execute(stmt).scalar_one_or_none()
        return None if found is None else str(found)

    def write_ref(self, repo: str, name: str, new: str | None, old: str | None) -> bool:
        """Point ref ``name`` at ``new`` (``None``: delete) if it points at ``old`` (``None``: absent).

        One statement either way, so the row count says whether this writer won.
        """
        if new is None and old is None:
            return self.read_ref(repo, name) is None
        with self._pg.session() as session:
            result = session.execute(_ref_statement(repo, name, new, old))
        return _rowcount(result) == 1

    def list_refs(self, repo: str) -> dict[str, str]:
        """Return every ref of the repository: name to object id."""
        stmt = select(CasRef.name, CasRef.target).where(CasRef.repo == repo)
        with self._pg.session() as session:
            return {str(name): str(target) for name, target in session.execute(stmt)}


def _ref_statement(repo: str, name: str, new: str | None, old: str | None) -> Any:
    """The compare-and-set statement for one ref write: create, move or delete."""
    if old is None:
        return (
            insert(CasRef)
            .values(repo=repo, name=name, target=new)
            .on_conflict_do_nothing(index_elements=["repo", "name"])
        )
    where = (CasRef.repo == repo, CasRef.name == name, CasRef.target == old)
    if new is None:
        return delete(CasRef).where(*where)
    return update(CasRef).where(*where).values(target=new, updated_at=func.clock_timestamp())


def _rowcount(result: Any) -> int:
    """Rows a write statement touched."""
    return int(cast(CursorResult[Any], result).rowcount)
