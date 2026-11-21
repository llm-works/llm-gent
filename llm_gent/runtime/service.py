# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent service wrapper for appinfra.service integration.

Wraps agent creation and execution as an appinfra Service so it can
be managed by ThreadRunner or ProcessRunner.

For ProcessRunner: the service is pickled and sent to a subprocess.
ProcessRunner injects ``_lg`` and ``_shutdown_event`` before run().
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.service import Service

from ..core.platform import Context
from ..core.platform import Factory as PlatformFactory
from .runner import ManagedAgentRunner


if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as MPEvent

    from appinfra.log import Logger

    from ..bus.transport import WorkerBusConfig


class AgentService(Service):
    """Wraps agent creation and execution as an appinfra Service.

    Lifecycle:
    - setup(): Create platform, factory, agent, start agent
    - run(): Run the AgentRunner (blocks until shutdown)
    - teardown(): Signal runner to stop
    - is_healthy(): True once runner is connected to bus

    For ProcessRunner, ``_lg`` and ``_shutdown_event`` are injected
    before run(). The shutdown watcher thread bridges the mp.Event
    to the runner's ``_running`` flag.
    """

    def __init__(
        self,
        lg: Logger,
        agent_name: str,
        config: DotDict,
        llm_config: Any,
        bus_config: WorkerBusConfig,
        learn_config: Any | None = None,
        variables: dict[str, str] | None = None,
        factory_module: str = "llm_gent.agents.default",
    ) -> None:
        self._lg = lg
        self._agent_name = agent_name
        self._config = config
        self._llm_config = llm_config
        self._bus_config = bus_config
        self._learn_config = learn_config
        self._variables = variables or {}
        self._factory_module = factory_module
        self._runner: ManagedAgentRunner | None = None
        self._platform: Context | None = None
        self._healthy = False
        # Injected by ProcessRunner before run()
        self._shutdown_event: MPEvent | None = None

    @property
    def name(self) -> str:
        return self._agent_name

    def setup(self) -> None:
        """Create platform, load factory, create and start agent."""
        self._platform = PlatformFactory(self._lg).from_config(
            {"llm": self._llm_config or {}, "memory": self._learn_config}
        )
        agent: Any = None
        try:
            factory = self._load_factory(self._platform)
            agent = factory.create(self._config, variables=self._variables)
            agent.start()
            self._runner = ManagedAgentRunner(
                lg=self._lg,
                agent=agent,
                bus_config=self._bus_config,
                schedule_interval=self._extract_schedule(),
            )
        except Exception:
            self._abort_setup(agent)
            raise
        self._lg.info("agent service setup complete", extra={"agent": self._agent_name})

    def _abort_setup(self, agent: Any) -> None:
        """Undo a failed setup: stop the agent (a no-op unless it started), then close the platform."""
        if agent is not None:
            try:
                agent.stop()
            except Exception as e:
                self._lg.warning("agent stop failed", extra={"exception": e})
        self._close_platform()

    def run(self) -> None:
        """Run the agent runner (blocks until shutdown); close the platform once it stops."""
        if self._runner is None:
            raise RuntimeError("setup() must be called before run()")
        self._healthy = True
        self._start_shutdown_watcher()
        try:
            self._runner.run()
        finally:
            self._close_platform()

    def _close_platform(self) -> None:
        """Close the platform's shared clients; the runner has stopped the agent by now."""
        if self._platform is not None:
            self._platform.close()
            self._platform = None

    def teardown(self) -> None:
        """Signal the runner to stop."""
        if self._runner is not None:
            self._runner.request_shutdown()
        self._healthy = False

    def is_healthy(self) -> bool:
        return self._healthy

    def _start_shutdown_watcher(self) -> None:
        """Bridge ProcessRunner's shutdown_event to runner._running flag.

        ProcessRunner injects _shutdown_event (mp.Event) before run().
        This watcher thread monitors it and stops the runner when set.
        For ThreadRunner, _shutdown_event is None so this is a no-op.
        """
        if self._shutdown_event is None:
            return

        def watch() -> None:
            if self._shutdown_event is None:
                return
            self._shutdown_event.wait()
            if self._runner is not None:
                self._runner.request_shutdown()

        threading.Thread(target=watch, daemon=True, name=f"sd-{self._agent_name}").start()

    def _load_factory(self, platform: Any) -> Any:
        """Load agent factory from module."""
        import importlib

        module = self._config.get("module", self._factory_module)
        try:
            mod = importlib.import_module(module)
            return mod.Factory(platform=platform)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Failed to load factory from {module}: {e}") from e

    def _extract_schedule(self) -> float | None:
        """Extract schedule interval from config."""
        schedule = self._config.get("schedule")
        if schedule and isinstance(schedule, dict):
            interval = schedule.get("interval")
            if interval is not None:
                try:
                    return float(interval)
                except (ValueError, TypeError):
                    return None
        return None
