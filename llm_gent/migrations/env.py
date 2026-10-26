# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Alembic env.py for llm-gent's framework-level schema.

Two entry points:

1. Called by :class:`llm_gent.schema.SchemaManager` (normal app usage):
   runs on the open connection in ``config.attributes["connection"]``,
   inside SchemaManager's transaction. The version-table location is
   pre-set on the alembic config, and the logger arrives in
   ``config.attributes["lg"]``.
2. Called by the alembic CLI (dev/tooling): loads the URL from the
   config file named in ``LLM_GENT_CONFIG`` (default ``etc/pg.yaml``),
   picking up ``dbs.<LLM_GENT_DB_KEY>.url`` (default key ``main``), and
   creates its own root logger.

Every module that registers a model on :class:`llm_gent.schema.Base` is
imported here so ``target_metadata`` sees the full table set at
autogenerate time.
"""

from __future__ import annotations

import os

from alembic import context
from appinfra.log import LogConfig, LoggerFactory
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from llm_gent.schema import Base, _import_all_models


_import_all_models()


config = context.config

_lg = config.attributes.get("lg") or LoggerFactory.create_root(LogConfig.from_params(level="info"))


def _get_database_url() -> str:
    """Return the URL from alembic config, falling back to ``LLM_GENT_CONFIG``."""
    url: str | None = config.get_main_option("sqlalchemy.url")
    if url and "driver://user:pass@localhost/dbname" not in url:
        return url
    config_path = os.environ.get("LLM_GENT_CONFIG", "etc/pg.yaml")
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"database URL not configured: alembic sqlalchemy.url is a placeholder "
            f"and config file {config_path!r} does not exist. Either drive alembic "
            f"via llm_gent.schema.SchemaManager or set LLM_GENT_CONFIG to a pg.yaml "
            f"path."
        )
    from appinfra.config import Config

    db_key = os.environ.get("LLM_GENT_DB_KEY", "main")
    app_config = Config(config_path)
    return str(app_config.dbs[db_key].url)


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL without connecting (``alembic upgrade --sql``)."""
    version_table = config.get_main_option("version_table") or "alembic_version_gent"
    _lg.info("running offline llm-gent migration")
    context.configure(
        url=_get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=version_table,
    )
    with context.begin_transaction():
        context.run_migrations()


def _current_revision(connection: Connection, schema_name: str, version_table: str) -> str | None:
    """Read the version row; ``None`` while the version table does not exist yet."""
    # A savepoint keeps a missing table from aborting the enclosing transaction.
    connection.execute(text("SAVEPOINT check_version"))
    try:
        result = connection.execute(
            text(f'SELECT version_num FROM "{schema_name}"."{version_table}"')
        )
        current_rev: str | None = result.scalar()
        connection.execute(text("RELEASE SAVEPOINT check_version"))
        return current_rev
    except (OperationalError, ProgrammingError):
        connection.execute(text("ROLLBACK TO SAVEPOINT check_version"))
        return None


def _run_migrations(connection: Connection, schema_name: str, version_table: str) -> None:
    """Run the migrations on ``connection``, inside its owner's transaction."""
    from alembic.script import ScriptDirectory

    connection.execute(text(f'SET LOCAL search_path TO "{schema_name}", public'))
    head_rev = ScriptDirectory.from_config(config).get_current_head()
    current_rev = _current_revision(connection, schema_name, version_table)
    _lg.info(
        "llm-gent migration state",
        extra={"current_revision": current_rev, "head_revision": head_rev},
    )
    if current_rev == head_rev:
        _lg.info("llm-gent schema already at head; nothing to do")
        return

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=schema_name,
        version_table=version_table,
    )
    # The connection is already in a transaction, so alembic starts and commits none.
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection.

    When called by :class:`~llm_gent.schema.SchemaManager`, runs on its
    connection and leaves the commit to it; the schema name is pre-set on
    the alembic config. Via the CLI, connects with the configured URL and
    the schema defaults to ``public``.
    """
    schema_name = config.get_main_option("version_table_schema") or "public"
    version_table = config.get_main_option("version_table") or "alembic_version_gent"
    _lg.info("starting online llm-gent migration", extra={"schema": schema_name})

    connection = config.attributes.get("connection")
    if connection is not None:
        _run_migrations(connection, schema_name, version_table)
        return
    engine = create_engine(_get_database_url())
    try:
        with engine.begin() as conn:
            _run_migrations(conn, schema_name, version_table)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
