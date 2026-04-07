#!/usr/bin/env python3
"""Minimal external agent that connects to a running hub.

Usage:
    python examples/external_agent.py

Requires a hub running (llm-gent serve) on default ports.

Heartbeats are hub-initiated: the hub broadcasts HeartbeatRequest
periodically, and this agent responds with HeartbeatResponse.
"""

import contextlib
import signal
import threading
import time
from typing import Any

from appinfra.service import BufferedChannel

from llm_gent.bus.protocol import (
    AgentJoined,
    AgentLeft,
    HeartbeatRequest,
    Message,
    RegisterRequest,
    ShutdownNotice,
    UnregisterRequest,
)
from llm_gent.bus.transport import WorkerBusConfig, ZMQWorkerBus


class PrintLogger:
    """Minimal logger that prints to stdout."""

    def info(self, msg: str, **kw: Any) -> None:
        print(f"[INFO] {msg}", kw.get("extra", ""))

    def debug(self, msg: str, **kw: Any) -> None:
        pass

    def warning(self, msg: str, **kw: Any) -> None:
        print(f"[WARN] {msg}", kw.get("extra", ""))

    def trace(self, msg: str, **kw: Any) -> None:
        pass


_tick_count = 0
_stop = threading.Event()


def _handle_broadcast(bus: ZMQWorkerBus, message: Message) -> None:
    """Respond to hub heartbeat broadcasts and shutdown notices."""
    if isinstance(message, HeartbeatRequest):
        bus.publish_heartbeat(
            stats={"ticks": _tick_count, "errors": 0},
            round_id=message.round_id,
            request_id=message.id,
        )
    elif isinstance(message, ShutdownNotice):
        print(f"Shutdown notice: {message.reason} (grace={message.grace_period_secs}s)")
        _stop.set()
    elif isinstance(message, AgentJoined):
        print(f"Agent joined: {message.agent_id} caps={message.capabilities}")
    elif isinstance(message, AgentLeft):
        print(f"Agent left: {message.agent_id} reason={message.reason}")
    else:
        print(f"Broadcast: {message}")


def _connect_and_register(bus: ZMQWorkerBus) -> BufferedChannel[Any, Any]:
    """Connect bus, create channel, register with hub."""
    bus.start()
    time.sleep(0.3)

    assert bus.transport is not None
    channel: BufferedChannel[Any, Any] = BufferedChannel(bus.transport)

    req = RegisterRequest(agent_id=bus.agent_id, capabilities=["echo", "demo"])
    channel.submit(req, timeout=5.0)
    print(f"Registered: {bus.agent_id}")
    return channel


def _run_loop(agent_id: str) -> None:
    """Run until interrupted or hub sends ShutdownNotice."""
    global _tick_count

    def stop(sig: int, frame: Any) -> None:
        _stop.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"Agent '{agent_id}' running. Responding to hub heartbeats. Ctrl+C to stop.")
    while not _stop.is_set():
        _tick_count += 1
        _stop.wait(5)


def main() -> None:
    lg = PrintLogger()
    agent_id = "external-dummy"

    bus = ZMQWorkerBus(lg, agent_id, WorkerBusConfig())  # type: ignore[arg-type]
    bus.subscribe("broadcast", lambda msg: _handle_broadcast(bus, msg))

    channel = _connect_and_register(bus)
    try:
        _run_loop(agent_id)
    finally:
        with contextlib.suppress(Exception):
            channel.submit(UnregisterRequest(agent_id=agent_id), timeout=2.0)
            print("Unregistered.")
        channel.close()
        bus.stop()
        print("Done.")


if __name__ == "__main__":
    main()
