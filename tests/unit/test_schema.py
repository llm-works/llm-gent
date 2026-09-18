# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :mod:`llm_gent.schema` — assertions that need no DB.

Covers pure invariants:

- Advisory-lock key distinctness from kelt's (a collision would make
  ``ensure_schema`` calls serialize across packages on a shared engine).
- Namespaced version-table name (kelt-coexistence invariant).
"""

from __future__ import annotations

import pytest

from llm_gent.schema import _ADVISORY_LOCK_KEY, _VERSION_TABLE_NAME


pytestmark = pytest.mark.unit


class TestKeltCoexistenceInvariants:
    """Static checks that llm-gent's alembic infra doesn't collide with kelt's."""

    def test_advisory_lock_key_distinct_from_kelt(self) -> None:
        """Our advisory-lock key must differ from kelt's known value.

        Kelt owns ``7829104563218907456`` (llm_kelt.core.schema:34);
        an identical value here would make :meth:`SchemaManager.ensure_schema`
        calls on a shared engine serialize across the two packages
        instead of running independently.
        """
        from llm_kelt.core.schema import _ADVISORY_LOCK_KEY as KELT_KEY

        assert _ADVISORY_LOCK_KEY != KELT_KEY, (
            f"llm-gent's advisory-lock key must differ from kelt's; both are {_ADVISORY_LOCK_KEY}"
        )

    def test_version_table_name_namespaced(self) -> None:
        """Our alembic version table must not collide with kelt's.

        Kelt uses ``alembic_version_kelt``; a bare ``alembic_version``
        (alembic's default) or an identical name would let one package's
        upgrade overwrite the other's revision row.
        """
        from llm_kelt.core.schema import _VERSION_TABLE_NAME as KELT_TABLE

        assert _VERSION_TABLE_NAME != KELT_TABLE
        assert _VERSION_TABLE_NAME != "alembic_version"
        assert "llm_gent" in _VERSION_TABLE_NAME
