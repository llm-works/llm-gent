"""Agent service wrapper for appinfra.service integration.

Wraps agent creation and execution as an appinfra Service so it can
be managed by ThreadRunner or ProcessRunner.

For ProcessRunner: the service is pickled and sent to a subprocess.
ProcessRunner injects ``_lg`` and ``_shutdown_event`` before execute().
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.service import Service

from .runner import AgentRunner


if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as MPEvent

    from appinfra.log import Logger

    from llm_gent.bus.transport import WorkerBusConfig


class AgentService(Service):
    """Wraps agent creation and execution as an appinfra Service.

    Lifecycle:
    - setup(): Create platform, factory, agent, start agent
    - execute(): Run the AgentRunner (blocks until shutdown)
    - teardown(): Signal runner to stop
    - is_healthy(): True once runner is connected to bus

    For ProcessRunner, ``_lg`` and ``_shutdown_event`` are injected
    before execute(). The shutdown watcher thread bridges the mp.Event
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
        self._runner: AgentRunner | None = None
        self._healthy = False
        # Injected by ProcessRunner before execute()
        self._shutdown_event: MPEvent | None = None

    @property
    def name(self) -> str:
        return self._agent_name

    def setup(self) -> None:
        """Create platform, load factory, create and start agent."""
        from llm_gent.core.platform import PlatformContext

        platform = PlatformContext.from_config(
            lg=self._lg,
            llm_config=self._llm_config,
            learn_config=self._learn_config,
        )
        factory = self._load_factory(platform)
        agent = factory.create(self._config, variables=self._variables)
        agent.start()

        self._runner = AgentRunner(
            lg=self._lg,
            agent=agent,
            bus_config=self._bus_config,
            schedule_interval=self._extract_schedule(),
        )
        self._lg.info("agent service setup complete", extra={"agent": self._agent_name})

    def execute(self) -> None:
        """Run the agent runner (blocks until shutdown)."""
        if self._runner is None:
            raise RuntimeError("setup() must be called before execute()")
        self._healthy = True
        self._start_shutdown_watcher()
        self._runner.run()

    def teardown(self) -> None:
        """Signal the runner to stop."""
        if self._runner is not None:
            self._runner.request_shutdown()
        self._healthy = False

    def is_healthy(self) -> bool:
        return self._healthy

    def _start_shutdown_watcher(self) -> None:
        """Bridge ProcessRunner's shutdown_event to runner._running flag.

        ProcessRunner injects _shutdown_event (mp.Event) before execute().
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
