# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Package-level SQL schema management for llm-gent.

llm-gent owns one alembic tree at :mod:`llm_gent.migrations` covering
every framework-level persistent table this package ships (currently
just ``llm_gent_flow_checkpoint``; more revisions land here as future
subsystems add persistence). Model classes live near their consumers
(e.g. :class:`llm_gent.flow.stores.postgres.FlowCheckpoint`) but share
the :class:`Base` defined here so ``Base.metadata`` sees every table
alembic needs to autogenerate against.

The pattern mirrors llm-kelt's ``SchemaManager`` — one advisory-lock-guarded
``ensure_schema`` per package, a namespaced version table
(``alembic_version_llm_gent``) so llm-gent's history coexists cleanly
with kelt's own alembic in the same Postgres schema.

Public entry points:

- :func:`ensure_schema` — one-shot: run migrations to head under an
  advisory lock. Idempotent; safe from concurrent processes.
- :class:`SchemaManager` — the object under the hood; use directly when
  you need :meth:`~SchemaManager.get_status` or the ``VERIFY``/``SKIP``
  modes.
- :class:`SchemaMode` — enum for consumers that want to declare intent
  (``ENSURE`` / ``VERIFY`` / ``SKIP``) rather than dispatch themselves.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from appinfra.log import Logger
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase


if TYPE_CHECKING:
    from appinfra.db.pg import PG


# Advisory lock key for llm-gent schema operations. Fixed constant per
# kelt's precedent (Python's hash() is randomized across processes so
# cannot be used). Distinct from kelt's key so the two lock namespaces
# do not collide on a shared engine.
_ADVISORY_LOCK_KEY = 4923108657234587123

# Namespaced alembic bookkeeping table. Coexists with kelt's
# ``alembic_version_kelt`` (and any other component's namespaced version
# row) in the same Postgres schema without collision.
_VERSION_TABLE_NAME = "alembic_version_llm_gent"

_SCHEMA_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class Base(DeclarativeBase):
    """Declarative base shared by every llm-gent framework table.

    Model classes defined anywhere in the package inherit from this so
    ``Base.metadata`` sees them at alembic autogenerate time. The alembic
    ``env.py`` imports every module that registers a subclass — see
    :mod:`llm_gent.migrations.env`.
    """


class SchemaVersionError(RuntimeError):
    """Raised when the database schema is incompatible with this library.

    Currently fires only on :attr:`SchemaState.TOO_NEW` — the DB has a
    revision this installed llm-gent version does not know how to run
    against, and downgrade is not supported.
    """


class SchemaState(Enum):
    """Schema version state relative to library head."""

    MISSING = "missing"
    CURRENT = "current"
    NEEDS_UPGRADE = "upgrade"
    TOO_NEW = "too_new"


class SchemaMode(Enum):
    """Intent declaration for consumers that don't want to dispatch themselves.

    Mirrors :class:`llm_kelt.SchemaMode` so downstream code that already
    reasons in these terms reads the same either side.

    ``ENSURE`` — run migrations to head (needs write access).
    ``VERIFY`` — read-only check that the schema matches head; raise
    :exc:`SchemaVersionError` otherwise.
    ``SKIP``   — do nothing; caller vouches the schema is usable.
    """

    ENSURE = "ensure"
    VERIFY = "verify"
    SKIP = "skip"


@dataclass
class SchemaStatus:
    """Snapshot of the schema's state at :meth:`SchemaManager.get_status` time."""

    state: SchemaState
    current_version: str | None
    head_version: str


class SchemaManager:
    """Advisory-lock-guarded runner for llm-gent's alembic migrations.

    Thread- and process-safe: :meth:`ensure_schema` takes a Postgres
    session-level advisory lock before touching the database, so parallel
    workers (pytest-xdist, multi-tenant servers) migrate exactly once.
    """

    def __init__(
        self,
        lg: Logger,
        pg: PG,
    ) -> None:
        """Bind a logger and appinfra PG handle.

        The bound schema (``pg.schema`` when the PG handle was
        constructed with ``schema=``, else ``"public"``) is used for
        the version-table location, ``search_path`` on migration
        transactions, and ``create_all`` bootstrap on a fresh database.
        """
        self._lg = lg
        self._pg = pg
        self._engine = pg.engine
        self._schema_name = pg.schema or "public"
        if not _SCHEMA_NAME_PATTERN.match(self._schema_name):
            raise ValueError(
                f"Invalid schema name {self._schema_name!r}: must be 1-63 chars, "
                f"start with a lowercase letter, and contain only lowercase letters, "
                f"numbers, and underscores."
            )
        self._migrations_path = Path(__file__).parent / "migrations"

    def _get_alembic_config(self) -> AlembicConfig:
        """Assemble the alembic ``Config`` pointed at this package's migrations."""
        config = AlembicConfig(str(self._migrations_path / "alembic.ini"))
        config.set_main_option("script_location", str(self._migrations_path))
        # Escape % as %% for ConfigParser interpolation safety (passwords may contain %).
        url_escaped = str(self._engine.url).replace("%", "%%")
        config.set_main_option("sqlalchemy.url", url_escaped)
        config.set_main_option("version_table_schema", self._schema_name)
        config.set_main_option("version_table", _VERSION_TABLE_NAME)
        return config

    def _get_head_version(self) -> str:
        """Read the head revision from the on-disk migration chain."""
        config = self._get_alembic_config()
        script = ScriptDirectory.from_config(config)
        head = script.get_current_head()
        if head is None:
            raise SchemaVersionError("no migrations found in llm_gent/migrations/versions")
        return str(head)

    def _get_current_version(self) -> str | None:
        """Read the DB's current revision, or ``None`` if the version table is absent."""
        try:
            with self._engine.connect() as conn:
                mig_context = MigrationContext.configure(
                    conn,
                    opts={
                        "version_table_schema": self._schema_name,
                        "version_table": _VERSION_TABLE_NAME,
                    },
                )
                revision = mig_context.get_current_revision()
                return str(revision) if revision is not None else None
        except Exception as e:
            if "UndefinedTable" in type(e).__name__:
                return None
            raise

    def _is_version_in_chain(self, version: str) -> bool:
        """Check whether ``version`` is a known ancestor of head."""
        config = self._get_alembic_config()
        script = ScriptDirectory.from_config(config)
        try:
            for rev in script.walk_revisions():
                if rev.revision == version:
                    return True
        except Exception:
            return False
        return False

    def get_status(self) -> SchemaStatus:
        """Return the current schema state (missing / current / upgradeable / too new)."""
        head_version = self._get_head_version()
        current_version = self._get_current_version()
        if current_version is None:
            state = SchemaState.MISSING
        elif current_version == head_version:
            state = SchemaState.CURRENT
        elif self._is_version_in_chain(current_version):
            state = SchemaState.NEEDS_UPGRADE
        else:
            state = SchemaState.TOO_NEW
        return SchemaStatus(state=state, current_version=current_version, head_version=head_version)

    def verify_schema(self) -> SchemaStatus:
        """Read-only: raise :exc:`SchemaVersionError` when the schema is not current.

        Never issues DDL. Use from processes that must not migrate on their
        own (read-only replicas, ops-side inspectors).
        """
        status = self.get_status()
        if status.state != SchemaState.CURRENT:
            raise SchemaVersionError(
                f"llm-gent schema is {status.state.value} "
                f"(current={status.current_version}, head={status.head_version}); "
                f"call ensure_schema() or run alembic upgrade manually."
            )
        return status

    def ensure_schema(
        self,
        wait: bool = True,
        timeout_seconds: float = 30.0,
    ) -> SchemaStatus:
        """Run migrations to head under an advisory lock. Idempotent, concurrency-safe."""
        status = self.get_status()
        if status.state == SchemaState.CURRENT:
            self._lg.trace(
                "llm-gent schema already current",
                extra={"schema": self._schema_name, "version": status.current_version},
            )
            return status
        self._check_version_compatible(status)
        return self._migrate_with_lock(wait, timeout_seconds)

    def _migrate_with_lock(self, wait: bool, timeout_seconds: float) -> SchemaStatus:
        """Acquire the advisory lock, re-check under it, apply migration.

        Uses autocommit isolation so the lock connection never enters
        ``idle in transaction`` state while Alembic runs its own DDL.
        """
        with self._engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            if not self._acquire_lock(conn, wait, timeout_seconds):
                raise TimeoutError(
                    f"could not acquire llm-gent schema lock within {timeout_seconds}s"
                )
            try:
                status = self.get_status()
                if status.state == SchemaState.CURRENT:
                    return status
                self._check_version_compatible(status)
                self._apply_migration(status)
                return self.get_status()
            finally:
                self._release_lock(conn)

    def _acquire_lock(self, conn: Any, wait: bool, timeout_seconds: float) -> bool:
        """Take the session-level advisory lock; blocks if ``wait`` is true.

        Uses ``SET statement_timeout`` (not ``SET LOCAL``) because the
        connection runs in autocommit mode — there is no transaction for
        ``LOCAL`` to scope to.
        """
        if wait:
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive and finite")
            timeout_ms = math.ceil(timeout_seconds * 1000)
            conn.execute(text(f"SET statement_timeout = '{timeout_ms}ms'"))
            try:
                conn.execute(text(f"SELECT pg_advisory_lock({_ADVISORY_LOCK_KEY})"))
                return True
            except Exception as e:
                self._lg.warning("failed to acquire llm-gent schema lock", extra={"exception": e})
                return False
            finally:
                conn.execute(text("RESET statement_timeout"))
        result = conn.execute(text(f"SELECT pg_try_advisory_lock({_ADVISORY_LOCK_KEY})")).scalar()
        return bool(result)

    def _release_lock(self, conn: Any) -> None:
        """Release the advisory lock (autocommit — no explicit commit needed)."""
        conn.execute(text(f"SELECT pg_advisory_unlock({_ADVISORY_LOCK_KEY})"))

    def _apply_migration(self, status: SchemaStatus) -> None:
        """Bootstrap or upgrade depending on state."""
        if status.state == SchemaState.MISSING:
            self._bootstrap_fresh_database()
        elif status.state == SchemaState.NEEDS_UPGRADE:
            self._run_upgrade()

    def _bootstrap_fresh_database(self) -> None:
        """Create every table from :attr:`Base.metadata` in the target schema + stamp head."""
        self._lg.debug("creating llm-gent schema", extra={"schema": self._schema_name})
        head_version = self._get_head_version()
        _import_all_models()
        with self._engine.connect() as conn:
            self._set_search_path(conn)
            self._create_tables_in_schema(conn)
            self._stamp_alembic_version(conn, head_version)
            conn.commit()
        self._lg.info(
            "llm-gent schema created",
            extra={"schema": self._schema_name, "version": head_version},
        )

    def _run_upgrade(self) -> None:
        """Run alembic ``upgrade head`` against the bound engine."""
        self._lg.info("upgrading llm-gent schema to head")
        config = self._get_alembic_config()
        command.upgrade(config, "head")
        self._lg.info("llm-gent schema upgrade complete")

    def _check_version_compatible(self, status: SchemaStatus) -> None:
        """Raise :exc:`SchemaVersionError` on :attr:`SchemaState.TOO_NEW`."""
        if status.state == SchemaState.TOO_NEW:
            raise SchemaVersionError(
                f"database schema version {status.current_version!r} is newer than "
                f"library head {status.head_version!r}; downgrade is not supported. "
                f"Upgrade llm-gent."
            )

    def _set_search_path(self, conn: Any) -> None:
        """Point DDL at the target schema for this transaction."""
        conn.execute(text(f'SET LOCAL search_path TO "{self._schema_name}", public'))

    def _create_tables_in_schema(self, conn: Any) -> None:
        """``create_all`` against a copy of ``Base.metadata`` under the target schema."""
        from sqlalchemy import MetaData

        scoped = MetaData()
        for table in Base.metadata.tables.values():
            table.to_metadata(scoped, schema=self._schema_name)
        scoped.create_all(conn)

    def _stamp_alembic_version(self, conn: Any, version: str) -> None:
        """Write ``version`` into the version table (creating it if absent)."""
        conn.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {_VERSION_TABLE_NAME} "
                f"(version_num VARCHAR(32) PRIMARY KEY)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO {_VERSION_TABLE_NAME} (version_num) VALUES (:rev) "
                f"ON CONFLICT (version_num) DO NOTHING"
            ),
            {"rev": version},
        )


def _import_all_models() -> None:
    """Import every module that registers a model on :class:`Base`.

    Called before any alembic autogenerate / ``create_all`` so
    ``Base.metadata`` is populated. New framework tables land here as
    an import line — no dynamic discovery, so a missed import fails
    loudly at migration time rather than shipping an empty schema.
    """
    from llm_gent.flow.stores import postgres  # noqa: F401


def ensure_schema(
    lg: Logger,
    pg: PG,
    wait: bool = True,
    timeout_seconds: float = 30.0,
) -> SchemaStatus:
    """Convenience wrapper: construct a :class:`SchemaManager` and run ``ensure_schema``.

    Consumers call this once at process start before touching any
    llm-gent persistent table (e.g. before wiring a
    :class:`~llm_gent.flow.stores.PgCheckpointStore`).
    """
    return SchemaManager(lg, pg).ensure_schema(wait=wait, timeout_seconds=timeout_seconds)
