# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Move checkpoint histories onto :mod:`appinfra.cas` repositories.

Drops the name, object and ref tables of revision 001 — their histories
are not readable as :mod:`appinfra.cas` objects (different ids) — and
creates the tables backing :class:`llm_gent.flow.stores.PgStore`:

- ``gent_cas_object`` — one row per object ``(repo, id)``: its ``type``
  (``blob`` / ``tree`` / ``commit``), serialized ``data`` and
  ``written_at`` (server clock; refreshed by a re-put).
- ``gent_cas_ref`` — one row per ref ``(repo, name)`` pointing at a
  ``target`` object id, moved by compare-and-set, with ``updated_at``.

Revision ID: 002
Revises: 001
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the revision-001 tables and create the object and ref tables of :class:`PgStore`."""
    op.drop_table("gent_flow_ref")
    op.drop_table("gent_flow_object")
    op.drop_table("gent_flow_name")
    op.create_table(
        "gent_cas_object",
        sa.Column("repo", sa.String(length=255), nullable=False),
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=8), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.clock_timestamp(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("repo", "id", name="pk_gent_cas_object"),
    )
    op.create_table(
        "gent_cas_ref",
        sa.Column("repo", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("target", sa.String(length=64), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.clock_timestamp(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("repo", "name", name="pk_gent_cas_ref"),
    )


def downgrade() -> None:
    """Not supported: the revision-001 tables held histories in a format gent no longer reads."""
    raise NotImplementedError("downgrade from 002 is not supported")
