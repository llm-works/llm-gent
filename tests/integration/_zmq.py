# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Waiting for a ZMQ message to arrive.

ZMQ signals neither a finished connect nor a subscription in place, and drops what is sent
before: a PUB socket what no subscriber has joined for yet, a ROUTER what it routes to a peer
that has not connected. A test sends until the message arrives; the deadline is the failure
path, never the expected one.
"""

import time
from collections.abc import Callable


POLL_SECS = 0.02
"""How long one send gets to arrive before it is sent again."""


def send_until(
    send: Callable[[], object], arrived: Callable[[float], bool], timeout: float = 5.0
) -> None:
    """Call ``send`` until ``arrived(wait)`` reports the message, waiting up to ``wait`` each time.

    Raises ``AssertionError`` when nothing arrived within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        send()
        if arrived(POLL_SECS):
            return
        assert time.monotonic() < deadline, f"nothing arrived within {timeout}s"


def holds(check: Callable[[], bool]) -> Callable[[float], bool]:
    """An ``arrived`` for :func:`send_until` that polls ``check``: for state with no event."""

    def arrived(wait: float) -> bool:
        time.sleep(wait)
        return check()

    return arrived
