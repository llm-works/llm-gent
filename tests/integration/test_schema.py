# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for :class:`llm_gent.schema.SchemaManager` coexistence with kelt.

The two packages can share one Postgres database in a deployment that runs both.
This module exercises the invariants the store-side unit tests can't:
that ``ensure_schema`` reads/writes only its own namespaced version
table, leaves any pre-existing kelt state alone, and remains idempotent
in the presence of that state.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from alembic import command
from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import text

from llm_gent.schema import (
    _VERSION_TABLE_NAME,
    SchemaManager,
    SchemaState,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def pg_with_fake_kelt(pg_isolated: PG, pg_test_logger: Logger) -> Generator[PG, None, None]:
    """A schema-isolated PG with a pre-planted ``alembic_version_kelt`` row.

    Simulates a database where llm-kelt was migrated first; llm-gent's
    :meth:`SchemaManager.ensure_schema` must run on top of that without
    touching the kelt bookkeeping row.
    """
    schema = pg_isolated._schema_mgr.schema
    with pg_isolated.session() as session:
        session.execute(
            text(
                f'CREATE TABLE IF NOT EXISTS "{schema}".alembic_version_kelt '
                f"(version_num VARCHAR(32) PRIMARY KEY)"
            )
        )
        session.execute(
            text(
                f'INSERT INTO "{schema}".alembic_version_kelt (version_num) '
                f"VALUES ('fake-kelt-006') ON CONFLICT DO NOTHING"
            )
        )
    yield pg_isolated
    # Cleanup happens via pg_isolated's schema drop; nothing to undo here.


class TestKeltCoexistence:
    """``ensure_schema`` on top of pre-existing kelt state leaves it untouched."""

    def test_ensure_schema_ignores_kelt_version_table(
        self, pg_with_fake_kelt: PG, pg_test_logger: Logger
    ) -> None:
        """Running our migration must not read or write ``alembic_version_kelt``."""
        mgr = SchemaManager(pg_test_logger, pg_with_fake_kelt)
        status = mgr.ensure_schema()
        assert status.state == SchemaState.CURRENT

        schema = pg_with_fake_kelt._schema_mgr.schema
        with pg_with_fake_kelt.session() as session:
            kelt_rev = session.execute(
                text(f'SELECT version_num FROM "{schema}".alembic_version_kelt')
            ).scalar()
            gent_rev = session.execute(
                text(f'SELECT version_num FROM "{schema}".{_VERSION_TABLE_NAME}')
            ).scalar()

        assert kelt_rev == "fake-kelt-006", "kelt version row must not be modified"
        assert gent_rev == status.head_version, "gent version row must be stamped"

    def test_reensure_schema_still_idempotent_with_kelt_present(
        self, pg_with_fake_kelt: PG, pg_test_logger: Logger
    ) -> None:
        """A second ensure_schema against the same DB stays a no-op — including with kelt state."""
        mgr = SchemaManager(pg_test_logger, pg_with_fake_kelt)
        mgr.ensure_schema()
        second = mgr.ensure_schema()
        assert second.state == SchemaState.CURRENT

    def test_gent_version_table_isolated_from_kelt(
        self, pg_with_fake_kelt: PG, pg_test_logger: Logger
    ) -> None:
        """The two version tables coexist as distinct relations in the same schema."""
        SchemaManager(pg_test_logger, pg_with_fake_kelt).ensure_schema()
        schema = pg_with_fake_kelt._schema_mgr.schema
        with pg_with_fake_kelt.session() as session:
            rows = session.execute(
                text(
                    "SELECT tablename FROM pg_catalog.pg_tables "
                    "WHERE schemaname = :s AND tablename LIKE 'alembic_version_%' "
                    "ORDER BY tablename"
                ),
                {"s": schema},
            ).fetchall()
        names = [r[0] for r in rows]
        assert "alembic_version_kelt" in names
        assert _VERSION_TABLE_NAME in names


def _tables(pg: PG) -> set[str]:
    with pg.session() as session:
        rows = session.execute(
            text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = :s"),
            {"s": pg._schema_mgr.schema},
        ).fetchall()
    return {r[0] for r in rows}


class TestUpgrade:
    def test_upgrade_from_001_replaces_the_history_tables(
        self, pg_isolated: PG, pg_test_logger: Logger
    ) -> None:
        """A database at revision 001 upgrades in-process to the appinfra.cas tables."""
        mgr = SchemaManager(pg_test_logger, pg_isolated)
        mgr.ensure_schema()
        schema = pg_isolated._schema_mgr.schema
        with pg_isolated.session() as session:
            for table in ("gent_cas_ref", "gent_cas_object", _VERSION_TABLE_NAME):
                session.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table}"'))
        config = mgr._get_alembic_config()
        config.attributes["lg"] = pg_test_logger
        command.upgrade(config, "001")
        assert {"gent_flow_name", "gent_flow_object", "gent_flow_ref"} <= _tables(pg_isolated)
        assert mgr.get_status().state == SchemaState.NEEDS_UPGRADE

        assert mgr.ensure_schema().state == SchemaState.CURRENT
        tables = _tables(pg_isolated)
        assert {"gent_cas_object", "gent_cas_ref"} <= tables
        assert not tables & {"gent_flow_name", "gent_flow_object", "gent_flow_ref"}
