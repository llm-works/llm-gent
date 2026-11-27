# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent service: an agent as a permanent service on appinfra's service Runtime.

The Runtime runs it on a thread, or with ``isolate=True`` in a process of its
own, where setup() builds the agent and the Runtime gives the service a logger
that forwards to the parent's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.service import RestartPolicy, Runner, Runtime, Service

from ..core.platform import Factory as PlatformFactory
from ..core.platform import Platform
from .runner import ManagedAgentRunner


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ..bus.transport import WorkerBusConfig


READY_TIMEOUT = 30.0
"""Seconds an agent service has to set up and run before its start fails."""


def spawn_agent(
    runtime: Runtime, service: AgentService, policy: RestartPolicy | None, *, isolate: bool
) -> Runner:
    """Spawn ``service`` on ``runtime`` and wait until it runs.

    Raises:
        Whatever :meth:`Runtime.ready` raises, e.g. ``SetupError`` when the
        agent could not be built; the service has left the runtime by then.
    """
    runner = runtime.spawn(service, policy, isolate=isolate)
    try:
        runtime.ready(runner, timeout=READY_TIMEOUT)
    except Exception:
        runtime.stop_service(runner)
        raise
    return runner


class AgentService(Service):
    """Wraps agent creation and execution as an appinfra Service.

    Lifecycle:
    - setup(): Create platform, factory, agent, start agent
    - run(): Run the agent's runner until the service's ``stopping`` token is set
    - is_healthy(): True once the runner runs
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
        self._platform: Platform | None = None
        self._healthy = False

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
                stopping=self.stopping,
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
        """Run the agent runner until ``stopping`` is set; close the platform once it stops."""
        if self._runner is None:
            raise RuntimeError("setup() must be called before run()")
        self._healthy = True
        try:
            self._runner.run()
        finally:
            self._healthy = False
            self._close_platform()

    def _close_platform(self) -> None:
        """Close the platform's shared clients; the runner has stopped the agent by now."""
        if self._platform is not None:
            self._platform.close()
            self._platform = None

    def is_healthy(self) -> bool:
        return self._healthy

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
