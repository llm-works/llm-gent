# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Pytest configuration for llm-gent tests.

Two responsibilities live here:

1. **Plugin registration** — enables :mod:`appinfra.testing` (markers,
   expected-skip machinery) and :mod:`appinfra.db.pg.testing` (schema-
   isolated PG fixtures).
2. **Postgres availability protocol** — probes the configured PG
   endpoint once per session and, when unreachable, deselects every
   test that transitively pulls in the ``pg_test_config`` fixture.
   Deselecting (vs :func:`pytest.skip`) keeps the failure-step output
   clean when an upstream step (coverage threshold, unrelated failure)
   dumps the log tail — no wall of ``SKIPPED [pg-unavailable]`` lines.
   Mirrors the pattern llm-kelt runs.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
from appinfra.config import Config


pytest_plugins = ["appinfra.testing", "appinfra.db.pg.testing"]


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PG_CONFIG_PATH = PROJECT_ROOT / "etc" / "pg.yaml"


# Fixture every PG-dependent test transitively pulls in (via pg_isolated,
# pg_migrate_factory, pg_migrated, …). Used by
# pytest_collection_modifyitems to identify those tests and deselect them
# en masse when the Postgres probe fails at session start.
_PG_FIXTURE_GATE = "pg_test_config"
_PG_SKIP_REASON = "[expected] pg-unavailable"
_PG_STATUS_KEY: pytest.StashKey[dict] = pytest.StashKey()


def _get_pg_config_path() -> Path:
    """Return the pg config path, honoring ``LLM_GENT_TEST_PG_CONFIG`` overrides."""
    override = os.environ.get("LLM_GENT_TEST_PG_CONFIG")
    if override:
        return Path(override)
    return DEFAULT_PG_CONFIG_PATH


def _load_pg_url_from_config() -> str | None:
    """Return ``dbs.unittest.url`` from ``etc/pg.yaml`` (with substitution), or ``None``.

    Tests read the dedicated ``unittest`` DB, not ``main`` — keeps the
    per-worker isolated schemas out of the production database entirely.
    """
    path = _get_pg_config_path()
    if not path.exists():
        return None
    try:
        cfg = Config(str(path))
        url = cfg.dbs.unittest.url
    except Exception:  # noqa: BLE001 — config parse failure falls through
        return None
    return str(url) if url else None


def _is_server_available(host: str, port: int, timeout: float = 1.0) -> bool:
    """TCP-probe ``(host, port)`` with a short timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _resolve_pg_endpoint() -> tuple[str, int] | None:
    """Pick the host/port to probe.

    Order: ``APPINFRA_TEST_PG_URL`` / ``DATABASE_URL`` env → ``etc/pg.yaml``
    (or ``LLM_GENT_TEST_PG_CONFIG`` override) → ``None``.
    """
    env_url = os.environ.get("APPINFRA_TEST_PG_URL") or os.environ.get("DATABASE_URL")
    url = env_url or _load_pg_url_from_config()
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    return parsed.hostname, parsed.port or 5432


def pytest_configure(config: pytest.Config) -> None:
    """Probe Postgres once per session and stash reachability + endpoint."""
    endpoint = _resolve_pg_endpoint()
    if endpoint is None:
        return
    host, port = endpoint
    available = _is_server_available(host, port)
    config.stash[_PG_STATUS_KEY] = {"host": host, "port": port, "available": available}
    if not available:
        print(
            f"PG probe: {host}:{port} unreachable; PG-dependent tests will be deselected",
            file=sys.stderr,
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect tests that need PG when the session-start probe failed."""
    status = config.stash.get(_PG_STATUS_KEY, None)
    if status is None or status["available"]:
        return
    keep: list[pytest.Item] = []
    dropped: list[pytest.Item] = []
    for item in items:
        if _PG_FIXTURE_GATE in item.fixturenames:
            dropped.append(item)
        else:
            keep.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep
        print(
            f"PG probe: {status['host']}:{status['port']} unreachable; "
            f"deselected {len(dropped)} PG-dependent tests",
            file=sys.stderr,
        )


@pytest.fixture(scope="session")
def pg_test_config() -> dict[str, object]:
    """Provide database config to :mod:`appinfra.db.pg.testing`.

    Order: ``APPINFRA_TEST_PG_URL`` / ``DATABASE_URL`` env → ``etc/pg.yaml``
    (``dbs.main``) → :func:`pytest.skip`. When the session-start probe
    already deselected PG-dependent tests, this fixture is never
    resolved and its skip path is dead code — kept for the
    single-target ``pytest tests/x.py`` case where collection modifiers
    don't fire.
    """
    env_url = os.environ.get("APPINFRA_TEST_PG_URL") or os.environ.get("DATABASE_URL")
    if env_url:
        return {
            "url": env_url,
            "create_db": True,
            "readonly": False,
            "pool_pre_ping": True,
        }
    url = _load_pg_url_from_config()
    if not url:
        pytest.skip(_PG_SKIP_REASON)
    return {
        "url": url,
        "create_db": True,
        "readonly": False,
        "pool_pre_ping": True,
    }


@pytest.fixture(scope="session")
def pg_test_schema(worker_id: str) -> str:
    """Unique isolated schema name per xdist worker + PID.

    Overrides :func:`appinfra.db.pg.testing.pg_test_schema` so parallel
    pytest processes (``make check`` running unit + integration
    together) never share a schema.
    """
    pid = os.getpid()
    if worker_id == "master":
        return f"gent_test_master_{pid}"
    return f"gent_test_{worker_id}_{pid}"
