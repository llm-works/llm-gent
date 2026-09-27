# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Initial schema — Flow CAS object + ref store.

Creates the content-addressed persistence tables backing
:class:`llm_gent.flow.stores.PgCheckpointStore`:

- ``llm_gent_flow_name`` — one row per agent-chosen ``client_flow_id``,
  bound to its history's gent-generated ``flow_id`` (unique).
- ``llm_gent_flow_object`` — one row per
  ``(flow_id, kind, content_hash)`` with a ``BYTEA payload``.
  ``kind`` is one of ``"blob"`` / ``"tree"`` / ``"commit"`` (see
  :mod:`llm_gent.flow.state.cas`).
- ``llm_gent_flow_ref`` — one row per
  ``(flow_id, node_path, iteration)`` pointing at a
  ``commit_hash``, with a sequence-assigned ``seq`` for latest-ref lookup
  and a ``created_at`` timestamp for inspection.
- ``llm_gent_flow_tag`` — one row per ``(flow_id, name)`` pointing at a
  ``commit_hash``; re-put moves the tag.

Objects, refs and tags are history-scoped by ``flow_id``; blobs are
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

_REF_SEQ = "llm_gent_flow_ref_seq"
"""Sequence feeding ``llm_gent_flow_ref.seq`` (write order of refs)."""


def upgrade() -> None:
    """Create the name, object, ref and tag tables backing :class:`PgCheckpointStore`."""
    _create_name_table()
    _create_object_table()
    op.execute(sa.schema.CreateSequence(sa.Sequence(_REF_SEQ)))
    _create_ref_table()
    op.create_index("ix_flow_ref_flow_seq", "llm_gent_flow_ref", ["flow_id", sa.text("seq DESC")])
    _create_tag_table()


def _create_name_table() -> None:
    """Create ``llm_gent_flow_name`` — ``client_flow_id`` → ``flow_id`` bindings."""
    op.create_table(
        "llm_gent_flow_name",
        sa.Column("client_flow_id", sa.String(length=255), nullable=False),
        sa.Column("flow_id", sa.String(length=_FLOW_ID_LEN), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("client_flow_id", name="pk_flow_name"),
        sa.UniqueConstraint("flow_id", name="uq_flow_name_flow_id"),
    )


def _create_object_table() -> None:
    """Create ``llm_gent_flow_object`` — content-addressed object rows."""
    op.create_table(
        "llm_gent_flow_object",
        sa.Column("flow_id", sa.String(length=_FLOW_ID_LEN), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint(
            "flow_id",
            "kind",
            "content_hash",
            name="pk_flow_object",
        ),
    )


def _create_ref_table() -> None:
    """Create ``llm_gent_flow_ref`` — history-keyed pointers at commit hashes.

    ``seq`` is drawn from a database sequence on every insert and re-put,
    so "latest ref" is write order as the database saw it — independent
    of the writers' clocks. :func:`upgrade` creates the sequence before
    this table and the ``(flow_id, seq DESC)`` index after it.
    """
    op.create_table(
        "llm_gent_flow_ref",
        sa.Column("flow_id", sa.String(length=_FLOW_ID_LEN), nullable=False),
        sa.Column("node_path", sa.String(length=1024), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("commit_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "seq",
            sa.BigInteger(),
            server_default=sa.text(f"nextval('{_REF_SEQ}')"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "flow_id",
            "node_path",
            "iteration",
            name="pk_flow_ref",
        ),
    )


def _create_tag_table() -> None:
    """Create ``llm_gent_flow_tag`` — named, movable pointers at commit hashes."""
    op.create_table(
        "llm_gent_flow_tag",
        sa.Column("flow_id", sa.String(length=_FLOW_ID_LEN), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("commit_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("flow_id", "name", name="pk_flow_tag"),
    )


def downgrade() -> None:
    """Drop the tag, ref, object and name tables."""
    op.drop_table("llm_gent_flow_tag")
    op.drop_index("ix_flow_ref_flow_seq", table_name="llm_gent_flow_ref")
    op.drop_table("llm_gent_flow_ref")
    op.execute(sa.schema.DropSequence(sa.Sequence(_REF_SEQ)))
    op.drop_table("llm_gent_flow_object")
    op.drop_table("llm_gent_flow_name")
