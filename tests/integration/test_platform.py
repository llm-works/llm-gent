# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The platform builds from the packaged config: the file shipped with llm-gent, read for real."""

import pytest

from llm_gent.core.platform import Factory, Platform
from llm_gent.core.platform.config import packaged_llm_config
from tests.helpers import make_test_logger


pytestmark = pytest.mark.integration


def test_packaged_config_has_a_default_backend():
    config = packaged_llm_config()

    assert config["default"] in config["backends"]


async def test_empty_config_builds_and_closes():
    async with Factory(make_test_logger()).from_config({}) as platform:
        assert isinstance(platform, Platform)
        assert platform.llm is not None
