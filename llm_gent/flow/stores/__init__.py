# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Reference :class:`~llm_gent.flow.CheckpointStore` backends.

Two turnkey implementations of the Flow-level Protocol:

- :class:`JsonFileCheckpointStore` — one file per iteration under a
  caller-owned root, atomic writes, no external dependency.
- :class:`PgCheckpointStore` — Postgres via
  :class:`appinfra.db.pg.PG`, upsert-on-save,
  ``ORDER BY iteration DESC LIMIT 1`` load-latest. DDL is owned by
  llm-gent's package-level alembic (:func:`llm_gent.ensure_schema`);
  the store itself never issues DDL.
"""

from .json_file import JsonFileCheckpointStore
from .postgres import PgCheckpointStore


__all__ = [
    "JsonFileCheckpointStore",
    "PgCheckpointStore",
]
