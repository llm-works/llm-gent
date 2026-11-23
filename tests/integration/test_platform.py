# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The platform for real: built from the packaged config file, and shared by threads."""

import threading
from typing import Any

import pytest

from llm_gent.core import Role
from llm_gent.core.platform import Factory, Platform
from llm_gent.core.platform.config import packaged_llm_config
from tests.helpers import make_platform, make_test_logger


pytestmark = pytest.mark.integration


def test_packaged_config_has_a_default_backend():
    config = packaged_llm_config()

    assert config["default"] in config["backends"]


async def test_empty_config_builds_and_closes():
    async with Factory(make_test_logger()).from_config({}) as platform:
        assert isinstance(platform, Platform)
        assert platform.llm is not None


class _SpyLock:
    """A lock that sets ``contended`` when an acquire has to wait for the holder."""

    def __init__(self, contended: threading.Event) -> None:
        self._lock = threading.Lock()
        self._contended = contended

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            self._contended.set()
            self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def test_concurrent_first_requests_build_a_role_once():
    """A second thread asking during the first build waits for it, then gets the same SAIA."""
    entered, release, contended = threading.Event(), threading.Event(), threading.Event()
    built: list[Role] = []

    class _BlockingFactory:
        def build(self, role: Role) -> Any:
            built.append(role)
            if len(built) > 1:  # a second build: the second request did not wait
                contended.set()
                return object()
            entered.set()
            release.wait()
            return object()

    platform = make_platform(make_test_logger(), saia_factory=_BlockingFactory())
    platform._saia_lock = _SpyLock(contended)
    role = Role(name="planner")
    got: list[Any] = []
    first = threading.Thread(target=lambda: got.append(platform.saia(role)))
    second = threading.Thread(target=lambda: got.append(platform.saia(role)))
    first.start()
    entered.wait()
    second.start()
    contended.wait()  # the second request is at the lock while the first builds
    release.set()
    first.join()
    second.join()
    assert built == [role]
    assert got[0] is got[1]
