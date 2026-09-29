# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Reference :class:`~llm_gent.flow.CheckpointStore` backends.

Three turnkey implementations of the content-addressed object + ref store
Protocol:

- :class:`InMemoryCheckpointStore` — dicts in the process, for tests and
  short-lived local runs; payloads kept as serialized bytes.
- :class:`JsonFileCheckpointStore` — file-per-object under a
  caller-owned root, atomic writes, trivial :meth:`gc_history` via
  ``rmtree``, no external dependency.
- :class:`PgCheckpointStore` — Postgres via :class:`appinfra.db.pg.PG`,
  ``ON CONFLICT DO NOTHING`` for object puts, single-statement
  compare-and-set for refs. DDL is owned by llm-gent's package-level
  alembic (:func:`llm_gent.ensure_schema`); the store never issues DDL.
"""

from .json_file import JsonFileCheckpointStore
from .memory import InMemoryCheckpointStore
from .postgres import PgCheckpointStore


__all__ = [
    "InMemoryCheckpointStore",
    "JsonFileCheckpointStore",
    "PgCheckpointStore",
]
