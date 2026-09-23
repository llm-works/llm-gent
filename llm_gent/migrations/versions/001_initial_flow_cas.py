# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Initial schema — Flow CAS object + ref store.

Creates the content-addressed persistence pair backing
:class:`llm_gent.flow.stores.PgCheckpointStore`:

- ``llm_gent_flow_object`` — one row per
  ``(client_flow_id, kind, content_hash)`` with a ``BYTEA payload``.
  ``kind`` is one of ``"blob"`` / ``"tree"`` / ``"commit"`` (see
  :mod:`llm_gent.flow.state.cas`).
- ``llm_gent_flow_ref`` — one row per
  ``(client_flow_id, node_path, iteration)`` pointing at a
  ``commit_hash`` with a ``created_at`` timestamp for latest-ref lookup.

Both are trajectory-scoped by ``client_flow_id``, matching the arc's
non-goal on cross-trajectory blob sharing.

Revision ID: 001
Revises:
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the object + ref tables backing :class:`PgCheckpointStore`."""
    _create_object_table()
    _create_ref_table()


def _create_object_table() -> None:
    """Create ``llm_gent_flow_object`` — content-addressed object rows."""
    op.create_table(
        "llm_gent_flow_object",
        sa.Column("client_flow_id", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint(
            "client_flow_id",
            "kind",
            "content_hash",
            name="pk_flow_object",
        ),
    )


def _create_ref_table() -> None:
    """Create ``llm_gent_flow_ref`` — trajectory-keyed pointers at commit hashes."""
    op.create_table(
        "llm_gent_flow_ref",
        sa.Column("client_flow_id", sa.String(length=255), nullable=False),
        sa.Column("node_path", sa.String(length=1024), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("commit_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "client_flow_id",
            "node_path",
            "iteration",
            name="pk_flow_ref",
        ),
    )
    op.create_index(
        "ix_flow_ref_client_created",
        "llm_gent_flow_ref",
        ["client_flow_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    """Drop the object + ref tables."""
    op.drop_index("ix_flow_ref_client_created", table_name="llm_gent_flow_ref")
    op.drop_table("llm_gent_flow_ref")
    op.drop_table("llm_gent_flow_object")
