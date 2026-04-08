"""Runtime core - orchestrates agent lifecycle via appinfra.service.

Core manages agent services using appinfra's ThreadRunner/ProcessRunner.
Controller-to-agent communication uses appinfra's BufferedChannel backed
by ZMQ transports. Pub/sub (heartbeats, broadcasts) goes through the bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.service import BufferedChannel, ProcessRunner, State, ThreadRunner

from .handle import AgentHandle, AgentInfo
from .service import AgentService


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ..bus.transport import WorkerBusConfig, ZMQCoordinatorBus
    from .registry import AgentRegistry


class Core:
    """Runtime core - orchestrates agent lifecycle.

    Uses appinfra's runners for process/thread management and
    BufferedChannel over ZMQ transport for controller-to-agent
    request/response communication.
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
        self._channels: dict[str, BufferedChannel[Any, Any]] = {}

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry."""
        return self._registry

    def start(self, name: str) -> AgentInfo:
        """Start an agent as a service with a channel for communication.

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
        """Ask an agent a question via the channel."""
        self._require_running(name)

        from ..bus.protocol import AskRequest, AskResponse

        channel = self._channels.get(name)
        if channel is None:
            raise RuntimeError(f"No channel for agent: {name}")

        req = AskRequest(question=question)
        resp = channel.submit(req, timeout=timeout)
        if isinstance(resp, AskResponse):
            if not resp.success:
                raise RuntimeError(f"Agent {name} ask failed: {resp.error}")
            return resp.response
        return ""

    def feedback(self, name: str, message: str, timeout: float = 30.0) -> None:
        """Send feedback to an agent via the channel."""
        self._require_running(name)

        from ..bus.protocol import FeedbackRequest

        channel = self._channels.get(name)
        if channel is None:
            raise RuntimeError(f"No channel for agent: {name}")

        resp = channel.submit(FeedbackRequest(message=message), timeout=timeout)
        if hasattr(resp, "success") and not resp.success:
            raise RuntimeError(f"Agent {name} feedback failed: {getattr(resp, 'error', 'unknown')}")

    def shutdown(self) -> None:
        """Shut down all running agents."""
        for handle in self._registry.handles():
            if handle.state == State.RUNNING:
                try:
                    self.stop(handle.name)
                except Exception as e:
                    self._lg.warning(
                        "error stopping agent during shutdown",
                        extra={"agent": handle.name, "exception": e},
                    )

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
        """Create AgentService, ZMQ transport + channel, and start runner."""
        config = handle.config
        config["name"] = handle.name

        transport = self._bus.create_agent_transport(handle.name)
        channel: BufferedChannel[Any, Any] = BufferedChannel(transport)
        self._channels[handle.name] = channel

        try:
            runner = self._create_runner(handle)
            runner.start()
        except Exception:
            self._channels.pop(handle.name, None)
            channel.close()
            self._bus.remove_agent_transport(handle.name)
            raise

        self._runners[handle.name] = runner
        self._lg.debug("agent service started", extra={"agent": handle.name})

    def _create_runner(self, handle: AgentHandle) -> ThreadRunner | ProcessRunner:
        """Create an AgentService wrapped in a runner."""
        config = handle.config
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
            return ThreadRunner(service)
        return ProcessRunner(service)

    def _stop_service(self, name: str) -> None:
        """Stop the agent's runner and clean up channel/transport.

        Always cleans up channel and transport even if runner.stop() fails.
        Re-raises runner errors after cleanup.
        """
        runner = self._runners.pop(name, None)
        runner_error: Exception | None = None
        try:
            if runner is not None:
                runner.stop()
        except Exception as e:
            runner_error = e
        finally:
            channel = self._channels.pop(name, None)
            if channel is not None:
                channel.close()
            self._bus.remove_agent_transport(name)
        if runner_error is not None:
            raise runner_error
