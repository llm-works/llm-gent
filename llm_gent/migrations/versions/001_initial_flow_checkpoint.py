# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Initial schema — Flow-level checkpoint store table.

Creates ``llm_gent_flow_checkpoint``: one row per
``(client_flow_id, node_path, iteration)``, backing
:class:`llm_gent.flow.stores.PgCheckpointStore`. ``node_path`` scopes
records to one iterate in the composition graph so two iterates in a
chain, nested iterates, or an iterate inside a ``.map`` body never
collide in the store keyspace.

Revision ID: 001
Revises:
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the checkpoint table + unique index on ``(client_flow_id, node_path, iteration)``."""
    op.create_table(
        "llm_gent_flow_checkpoint",
        sa.Column("db_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("client_flow_id", sa.String(length=255), nullable=False),
        sa.Column("node_path", sa.String(length=1024), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("state_json", postgresql.JSONB(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("db_id"),
        sa.UniqueConstraint(
            "client_flow_id",
            "node_path",
            "iteration",
            name="uq_flow_checkpoint_trajectory_iter",
        ),
    )
    op.create_index(
        "ix_flow_checkpoint_trajectory_iter",
        "llm_gent_flow_checkpoint",
        ["client_flow_id", "node_path", "iteration"],
    )


def downgrade() -> None:
    """Drop the checkpoint table."""
    op.drop_index("ix_flow_checkpoint_trajectory_iter", table_name="llm_gent_flow_checkpoint")
    op.drop_table("llm_gent_flow_checkpoint")
