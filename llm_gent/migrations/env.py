# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Alembic env.py for llm-gent's framework-level schema.

Two entry points:

1. Called by :class:`llm_gent.schema.SchemaManager` (normal app usage):
   the URL and version-table location are pre-set on the alembic config.
2. Called by the alembic CLI (dev/tooling): loads the URL from the
   config file named in ``LLM_GENT_CONFIG`` (default ``etc/pg.yaml``),
   picking up ``dbs.<LLM_GENT_DB_KEY>.url`` (default key ``main``).

Every module that registers a model on :class:`llm_gent.schema.Base` is
imported here so ``target_metadata`` sees the full table set at
autogenerate time.
"""

from __future__ import annotations

import os

from alembic import context
from appinfra.log import LogConfig, LoggerFactory
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from llm_gent.schema import Base, _import_all_models


_import_all_models()


config = context.config

_lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))


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
    version_table = config.get_main_option("version_table") or "alembic_version_llm_gent"
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


def run_migrations_online() -> None:  # cq: exempt
    """Run migrations against a live connection.

    When called by :class:`~llm_gent.schema.SchemaManager` the schema
    name is pre-set on the alembic config; when called via the CLI it
    defaults to ``public``.
    """
    from alembic.script import ScriptDirectory

    url = _get_database_url()
    schema_name = config.get_main_option("version_table_schema") or "public"
    version_table = config.get_main_option("version_table") or "alembic_version_llm_gent"
    _lg.info("starting online llm-gent migration", extra={"schema": schema_name})

    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            connection.execute(text(f'SET LOCAL search_path TO "{schema_name}", public'))

            script = ScriptDirectory.from_config(config)
            head_rev = script.get_current_head()

            connection.execute(text("SAVEPOINT check_version"))
            try:
                result = connection.execute(
                    text(f'SELECT version_num FROM "{schema_name}"."{version_table}"')
                )
                current_rev = result.scalar()
                connection.execute(text("RELEASE SAVEPOINT check_version"))
            except (OperationalError, ProgrammingError):
                connection.execute(text("ROLLBACK TO SAVEPOINT check_version"))
                current_rev = None

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
            with context.begin_transaction():
                context.run_migrations()
                connection.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
