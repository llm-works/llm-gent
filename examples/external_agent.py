#!/usr/bin/env python3
"""Minimal external agent that connects to a running hub.

Usage:
    python examples/external_agent.py

Requires a hub running (llm-gent serve) on default ports.
"""

import signal
import time
from typing import Any

from llm_gent.bus.protocol import RegisterRequest, UnregisterRequest
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


def _connect_and_register(bus: ZMQWorkerBus, agent_id: str) -> None:
    """Connect to bus and register with hub."""
    bus.start()
    time.sleep(0.3)
    req = RegisterRequest(
        agent_id=agent_id,
        capabilities=["echo", "demo"],
        metadata={"version": "0.1"},
    )
    resp = bus.send(req, timeout=5.0)
    print(f"Registered: success={resp.success}")


def _run_heartbeat_loop(bus: ZMQWorkerBus, agent_id: str) -> None:
    """Send heartbeats until interrupted."""
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"Agent '{agent_id}' running. Sending heartbeats. Ctrl+C to stop.")
    tick = 0
    while running:
        tick += 1
        bus.publish_heartbeat({"ticks": tick, "errors": 0})
        time.sleep(5)


def main() -> None:
    lg = PrintLogger()
    agent_id = "external-dummy"

    bus = ZMQWorkerBus(lg, agent_id, WorkerBusConfig())  # type: ignore[arg-type]
    bus.subscribe("broadcast", lambda msg: print(f"Broadcast: {msg}"))

    _connect_and_register(bus, agent_id)
    _run_heartbeat_loop(bus, agent_id)

    # Unregister and disconnect
    import contextlib

    with contextlib.suppress(Exception):
        bus.send(UnregisterRequest(agent_id=agent_id), timeout=2.0)
        print("Unregistered.")

    bus.stop()
    print("Done.")


if __name__ == "__main__":
    main()
