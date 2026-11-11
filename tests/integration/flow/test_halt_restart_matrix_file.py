# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The halt/restart matrix against the file store: halt and crash, no policy or a checkpoint per leaf.

The cases and the checks are the unit matrix's
(:mod:`tests.unit.flow.test_halt_restart_matrix`); only the store differs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import file_store

from ...unit.flow.test_halt_restart_matrix import Case, check_resume_after_stop, parametrize_cases


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """The stop/resume test's file-store cases."""
    parametrize_cases(metafunc, "file")


async def test_resume_after_stop_matches_uninterrupted_run(case: Case, tmp_path: Path) -> None:
    await check_resume_after_stop(case, file_store(tmp_path / "cp"))
