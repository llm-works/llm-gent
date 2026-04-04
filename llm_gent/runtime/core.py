"""Runtime core - orchestrates agent lifecycle via appinfra.service.

Core manages agent services using appinfra's ThreadRunner for lifecycle
management. Agents communicate with the hub via the ZMQ bus.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from appinfra.service import ProcessRunner, State, ThreadRunner

from .handle import AgentHandle, AgentInfo
from .service import AgentService


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_gent.bus.transport import WorkerBusConfig, ZMQCoordinatorBus
    from llm_gent.runtime.registry import AgentRegistry


class Core:
    """Runtime core - orchestrates agent lifecycle.

    Uses appinfra's ThreadRunner to manage agent services. Each agent
    runs as an AgentService in a daemon thread. Communication with
    agents goes through the ZMQ bus.
    """

    def __init__(
        self,
        lg: Logger,
        registry: AgentRegistry,
        llm_config: Any,
        bus: ZMQCoordinatorBus,
        bus_config: WorkerBusConfig,
        learn_config: Any | None = None,
        variables: dict[str, str] | None = None,
        factory_module: str = "llm_gent.agents.default",
    ) -> None:
        self._lg = lg
        self._registry = registry
        self._llm_config = llm_config
        self._bus = bus
        self._bus_config = bus_config
        self._learn_config = learn_config
        self._variables = variables or {}
        self._factory_module = factory_module
        self._runners: dict[str, ThreadRunner | ProcessRunner] = {}

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry."""
        return self._registry

    def start(self, name: str) -> AgentInfo:
        """Start an agent as a threaded service.

        Args:
            name: Agent name.

        Returns:
            Updated AgentInfo.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent already active.
        """
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        if handle.state in {State.STARTING, State.RUNNING, State.STOPPING}:
            raise RuntimeError(f"Agent already active: {name} (state={handle.state.value})")

        handle.state = State.STARTING
        handle.error = None

        try:
            self._start_service(handle)
            handle.state = State.RUNNING
            self._lg.debug("agent started", extra={"agent": name})
        except Exception as e:
            handle.state = State.FAILED
            handle.error = str(e)
            self._lg.error("failed to start agent", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def stop(self, name: str) -> AgentInfo:
        """Stop an agent service.

        Args:
            name: Agent name.

        Returns:
            Updated AgentInfo.

        Raises:
            KeyError: If agent not found.
        """
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        if handle.state != State.RUNNING:
            return AgentInfo.from_handle(handle)

        handle.state = State.STOPPING

        try:
            self._stop_service(name)
            handle.state = State.STOPPED
            self._lg.info("agent stopped", extra={"agent": name})
        except Exception as e:
            handle.state = State.FAILED
            handle.error = str(e)
            self._lg.warning("error stopping agent", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def ask(self, name: str, question: str, timeout: float = 60.0) -> str:
        """Ask an agent a question via the bus."""
        self._require_running(name)

        from llm_gent.bus.protocol import AskRequest, AskResponse

        req = AskRequest(question=question)
        try:
            resp = self._bus.send_to_agent(name, req, timeout=timeout)
            if not resp.success:
                raise RuntimeError(resp.error or "ask failed")
            if isinstance(resp, AskResponse):
                return resp.response
            return ""
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def feedback(self, name: str, message: str, timeout: float = 30.0) -> None:
        """Send feedback to an agent via the bus."""
        self._require_running(name)

        from llm_gent.bus.protocol import FeedbackRequest

        req = FeedbackRequest(message=message)
        try:
            resp = self._bus.send_to_agent(name, req, timeout=timeout)
            if not resp.success:
                raise RuntimeError(resp.error or "feedback failed")
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def shutdown(self) -> None:
        """Shut down all running agents."""
        for handle in self._registry.handles():
            if handle.state == State.RUNNING:
                with contextlib.suppress(Exception):
                    self.stop(handle.name)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _require_running(self, name: str) -> AgentHandle:
        """Get handle for a running agent."""
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")
        if handle.state != State.RUNNING:
            raise RuntimeError(f"Agent not running: {name} (state={handle.state.value})")
        return handle

    def _start_service(self, handle: AgentHandle) -> None:
        """Create AgentService and start it via ThreadRunner or ProcessRunner."""
        config = handle.config
        config["name"] = handle.name
        execution = config.get("execution", "process")

        service = AgentService(
            lg=self._lg,
            agent_name=handle.name,
            config=config,
            llm_config=self._llm_config,
            bus_config=self._bus_config,
            learn_config=self._learn_config,
            variables=self._variables,
            factory_module=config.get("module", self._factory_module),
        )

        if execution == "thread":
            runner: ThreadRunner | ProcessRunner = ThreadRunner(service)
        else:
            runner = ProcessRunner(service)

        runner.start()
        self._runners[handle.name] = runner
        self._lg.debug(
            "agent service started", extra={"agent": handle.name, "execution": execution}
        )

    def _stop_service(self, name: str) -> None:
        """Stop the agent's ThreadRunner."""
        runner = self._runners.pop(name, None)
        if runner is not None:
            runner.stop()
