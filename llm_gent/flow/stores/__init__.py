# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Reference :class:`~llm_gent.flow.CheckpointStore` backends.

Two turnkey implementations of the content-addressed object + ref store
Protocol:

- :class:`JsonFileCheckpointStore` — file-per-object under a
  caller-owned root, atomic writes, trivial :meth:`gc_trajectory` via
  ``rmtree``, no external dependency.
- :class:`PgCheckpointStore` — Postgres via :class:`appinfra.db.pg.PG`,
  ``ON CONFLICT DO NOTHING`` for object puts, ``ON CONFLICT DO UPDATE``
  for ref puts, ``ORDER BY created_at DESC LIMIT 1`` for latest-ref
  lookup. DDL is owned by llm-gent's package-level alembic
  (:func:`llm_gent.ensure_schema`); the store never issues DDL.
"""

from .json_file import JsonFileCheckpointStore
from .postgres import PgCheckpointStore


__all__ = [
    "JsonFileCheckpointStore",
    "PgCheckpointStore",
]
