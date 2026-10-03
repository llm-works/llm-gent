# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Free TCP ports for tests that bind a coordinator bus.

A port is found by binding port 0 and closing the socket, and the bus
binds it afterwards. The port comes from the ephemeral range, so in
between any outgoing connection on the machine can take it (under
parallel test runs, often enough to fail a run). Starting retries with
fresh ports when a bind finds its port in use.
"""

import errno
import socket
from collections.abc import Callable
from typing import TypeVar

import zmq


T = TypeVar("T")


def free_ports(n: int) -> list[int]:
    """``n`` distinct ports that were free a moment ago."""
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    try:
        for s in socks:
            s.bind(("", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def start_on_free_ports(start: Callable[[list[int]], T], attempts: int = 5) -> T:
    """``start(ports)`` with three free ports, again with fresh ones while a port is taken.

    ``start`` builds and starts what binds the ports; one whose bind fails
    must leave nothing bound (as ``ZMQCoordinatorBus.start`` does).

    Raises:
        zmq.ZMQError: Every attempt found a port in use (the last error),
            or ``start`` failed otherwise.
    """
    for attempt in range(attempts):
        try:
            return start(free_ports(3))
        except zmq.ZMQError as e:
            if e.errno != errno.EADDRINUSE or attempt == attempts - 1:
                raise
    raise AssertionError("unreachable: attempts must be >= 1")
