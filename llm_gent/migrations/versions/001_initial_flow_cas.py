# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Initial schema — Flow CAS object + ref store.

Creates the content-addressed persistence tables backing
:class:`llm_gent.flow.stores.PgCheckpointStore`:

- ``gent_flow_name`` — one row per agent-chosen ``client_flow_id``,
  bound to its history's gent-generated ``flow_id`` (unique).
- ``gent_flow_object`` — one row per
  ``(flow_id, kind, content_hash)`` with a ``BYTEA payload``.
  ``kind`` is one of ``"blob"`` / ``"tree"`` / ``"commit"`` (see
  :mod:`llm_gent.flow.state.cas`).
- ``gent_flow_ref`` — one row per named ref ``(flow_id, name)``
  (``HEAD``, ``tags/...``) pointing at a ``commit_hash``, moved by
  compare-and-set, with an ``updated_at`` timestamp for inspection.

Objects and refs are history-scoped by ``flow_id``; blobs are
deliberately not shared across histories.

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


_FLOW_ID_LEN = 36
"""``flow_id`` column length — a canonical UUID string."""


def upgrade() -> None:
    """Create the name, object and ref tables backing :class:`PgCheckpointStore`."""
    _create_name_table()
    _create_object_table()
    _create_ref_table()


def _create_name_table() -> None:
    """Create ``gent_flow_name`` — ``client_flow_id`` → ``flow_id`` bindings."""
    op.create_table(
        "gent_flow_name",
        sa.Column("client_flow_id", sa.String(length=255), nullable=False),
        sa.Column("flow_id", sa.String(length=_FLOW_ID_LEN), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("client_flow_id", name="pk_gent_flow_name"),
        sa.UniqueConstraint("flow_id", name="uq_gent_flow_name_flow_id"),
    )


def _create_object_table() -> None:
    """Create ``gent_flow_object`` — content-addressed object rows."""
    op.create_table(
        "gent_flow_object",
        sa.Column("flow_id", sa.String(length=_FLOW_ID_LEN), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint(
            "flow_id",
            "kind",
            "content_hash",
            name="pk_gent_flow_object",
        ),
    )


def _create_ref_table() -> None:
    """Create ``gent_flow_ref`` — named, compare-and-set pointers at commit hashes."""
    op.create_table(
        "gent_flow_ref",
        sa.Column("flow_id", sa.String(length=_FLOW_ID_LEN), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("commit_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("flow_id", "name", name="pk_gent_flow_ref"),
    )


def downgrade() -> None:
    """Drop the ref, object and name tables."""
    op.drop_table("gent_flow_ref")
    op.drop_table("gent_flow_object")
    op.drop_table("gent_flow_name")
