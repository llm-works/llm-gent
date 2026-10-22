# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""gent's :class:`appinfra.cas.Store` backend for Postgres.

Checkpoint histories live in any :class:`appinfra.cas.Store`:
:class:`appinfra.cas.MemoryStore` for tests and short-lived local runs,
:class:`appinfra.cas.FileStore` for a directory on disk, and
:class:`PgStore` here — Postgres via :class:`appinfra.db.pg.PG`, its
tables owned by llm-gent's package-level alembic
(:func:`llm_gent.ensure_schema`); the store never issues DDL.
"""

from .postgres import PgStore


__all__ = ["PgStore"]
