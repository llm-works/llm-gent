# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A resume in a fresh process matches the same resume run in-process."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from appinfra.cas import FileStore

from llm_gent.flow.testing.checkpoint import build_canonical_flow, resume_in_subprocess
from tests.helpers import file_store, make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.e2e]


async def _halt_at_two(store: FileStore) -> None:
    await build_canonical_flow(
        make_test_logger(),
        max_iters=4,
        halt=asyncio.Event(),
        halt_after_iteration=2,
        store=store,
        client_flow_id="xproc",
    ).run()


async def test_matches_in_process_resume(tmp_path: Path) -> None:
    in_proc = file_store(tmp_path / "in-proc")
    await _halt_at_two(in_proc)
    await _halt_at_two(file_store(tmp_path / "xproc"))

    expected = await build_canonical_flow(
        make_test_logger(), max_iters=4, store=in_proc, client_flow_id="xproc"
    ).run(resume="latest")
    result = resume_in_subprocess(
        store_module="appinfra.cas",
        store_factory="FileStore",
        store_kwargs={"root": str(tmp_path / "xproc"), "fsync": False},
        flow_builder_kwargs={"max_iters": 4},
        client_flow_id="xproc",
        resume="latest",
    )
    assert result == dict(expected)
