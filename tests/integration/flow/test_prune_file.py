# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Pruning a history's repository, against the file store.

The tests are the unit suite's (:mod:`tests.unit.flow.test_prune`); only the store differs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from appinfra.cas import Store

from tests.helpers import file_store

from ...unit.flow import test_prune as unit


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return file_store(tmp_path / "cp")


class TestPrune(unit.TestPrune):
    """The unit suite's pruning tests on a :class:`~appinfra.cas.FileStore`."""
