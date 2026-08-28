# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Version test."""

import pytest

import llm_gent


pytestmark = pytest.mark.unit


def test_version() -> None:
    """Verify version is defined and valid format."""
    assert llm_gent.__version__
    # Version should be semver-ish (e.g., "0.1.0", "0.0.0.dev0", "1.0.0+g1234abc")
    assert isinstance(llm_gent.__version__, str)
    parts = llm_gent.__version__.split(".")
    assert len(parts) >= 3  # At least major.minor.patch
