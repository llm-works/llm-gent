"""Runtime core - orchestrates agent subprocess/thread lifecycle.

Core manages spawning agents as subprocesses or threads. Agents connect
to the hub via the ZMQ bus for all communication.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.log.mp import LogQueueListener
from appinfra.service import State

from .handle import AgentHandle, AgentInfo


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_gent.bus.transport import ZMQCoordinatorBus
    from llm_gent.core.traits.builtin.learn import LearnConfig
    from llm_gent.core.traits.builtin.llm import LLMConfig
    from llm_gent.runtime.registry import AgentRegistry


class Core:
    """Runtime core - orchestrates agent subprocess lifecycle.

    Spawns and terminates agent processes/threads. Agents communicate
    with the hub via the ZMQ bus (no direct channels).
    """

    def __init__(
        self,
        lg: Logger,
        registry: AgentRegistry,
        llm_config: LLMConfig,
        bus: ZMQCoordinatorBus,
        learn_config: DotDict | None = None,
        variables: dict[str, str] | None = None,
        factory_module: str = "llm_gent.agents.default",
        bus_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the runtime core.

        Args:
            lg: Logger instance.
            registry: Agent registry for handle lookup.
            llm_config: LLM configuration for agents.
            bus: Coordinator bus for sending messages to agents.
            learn_config: Optional learn configuration.
            variables: Variable substitutions for agent configs.
            factory_module: Default module containing the agent Factory class.
            bus_config: Bus config dict passed to agent subprocesses.
        """
        self._lg = lg
        self._registry = registry
        self._llm_config = llm_config
        self._bus = bus
        self._learn_config = learn_config
        self._variables = variables or {}
        self._factory_module = factory_module
        self._bus_config = bus_config

        # Queue-based logging for subprocesses
        self._log_queue: Queue[Any] = Queue()
        self._log_config = lg.queue_config(self._log_queue)
        self._log_listener = LogQueueListener(self._log_queue, lg)
        self._log_listener.start()

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry."""
        return self._registry

    def start(self, name: str) -> AgentInfo:
        """Start an agent process.

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
            self._spawn(handle)
            handle.state = State.RUNNING
            self._lg.debug("agent runtime started", extra={"agent": name})
        except Exception as e:
            handle.state = State.FAILED
            handle.error = str(e)
            self._lg.error("failed to start agent runtime", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def stop(self, name: str) -> AgentInfo:
        """Stop an agent process via bus shutdown request.

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
            self._terminate(handle)
            handle.state = State.STOPPED
            self._lg.info("agent stopped", extra={"agent": name})
        except Exception as e:
            handle.state = State.FAILED
            handle.error = str(e)
            self._lg.warning("error stopping agent", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def ask(self, name: str, question: str, timeout: float = 60.0) -> str:
        """Ask an agent a question via the bus.

        Args:
            name: Agent name.
            question: Question to ask.
            timeout: Response timeout in seconds.

        Returns:
            Agent's response string.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running or request failed.
        """
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
        """Send feedback to an agent via the bus.

        Args:
            name: Agent name.
            message: Feedback message.
            timeout: Response timeout in seconds.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running or feedback failed.
        """
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
        """Shut down all running agents and clean up."""
        for handle in self._registry.handles():
            if handle.state == State.RUNNING:
                try:
                    self.stop(handle.name)
                except Exception as e:
                    self._lg.warning(
                        "error stopping agent during shutdown",
                        extra={"agent": handle.name, "exception": e},
                    )

        self._log_listener.stop()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _require_running(self, name: str) -> AgentHandle:
        """Get handle for a running agent, raising if not found or not running."""
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")
        if handle.state != State.RUNNING:
            raise RuntimeError(f"Agent not running: {name} (state={handle.state.value})")
        return handle

    def _spawn(self, handle: AgentHandle) -> None:
        """Spawn subprocess or thread for agent."""
        execution_mode = handle.config.get("execution", "process")
        self._lg.debug(
            "spawning agent runtime...",
            extra={"agent": handle.name, "execution": execution_mode},
        )

        config = handle.config
        config["name"] = handle.name
        factory_module = config.get("module", self._factory_module)

        if execution_mode == "thread":
            self._spawn_thread(handle, config, factory_module)
        else:
            self._spawn_subprocess(handle, config, factory_module)

    def _spawn_subprocess(self, handle: AgentHandle, config: DotDict, factory_module: str) -> None:
        """Spawn subprocess for agent."""
        learn_config = self._learn_config if self._learn_config else None

        handle.process = mp.Process(
            target=_subprocess_entry,
            args=(
                handle.name,
                config,
                self._llm_config,
                learn_config,
                self._variables,
                self._log_config,
                factory_module,
                self._bus_config,
            ),
            name=f"agent-{handle.name}",
            daemon=True,
        )
        handle.process.start()
        self._lg.debug("spawned subprocess for agent runtime", extra={"agent": handle.name})

    def _spawn_thread(self, handle: AgentHandle, config: DotDict, factory_module: str) -> None:
        """Spawn thread for agent."""
        import threading

        handle.process = threading.Thread(
            target=_thread_entry,
            args=(
                handle.name,
                config,
                self._llm_config,
                self._learn_config,
                self._variables,
                self._lg,
                factory_module,
                self._bus_config,
            ),
            name=f"agent-{handle.name}",
            daemon=True,
        )
        handle.process.start()
        self._lg.debug("spawned thread for agent runtime", extra={"agent": handle.name})

    def _terminate(self, handle: AgentHandle) -> None:
        """Terminate agent process/thread via bus shutdown + process cleanup."""
        import contextlib

        # Send shutdown via bus (best-effort)
        with contextlib.suppress(Exception):
            from llm_gent.bus.protocol import ShutdownRequest

            self._bus.send_to_agent(handle.name, ShutdownRequest(), timeout=5.0)

        # Wait for process to exit
        if handle.process is not None:
            handle.process.join(timeout=5.0)
            if handle.process.is_alive() and isinstance(handle.process, mp.Process):
                handle.process.terminate()
                handle.process.join(timeout=2.0)
                if handle.process.is_alive():
                    handle.process.kill()

        handle.process = None


# =============================================================================
# Subprocess/thread entry points
# =============================================================================


def _subprocess_entry(
    name: str,
    config: DotDict,
    llm_config: LLMConfig,
    learn_config: LearnConfig | None,
    variables: dict[str, str],
    log_config: dict[str, Any],
    factory_module: str,
    bus_config: dict[str, Any] | None = None,
) -> None:
    """Entry point for agent subprocess."""
    from appinfra.log import Logger

    try:
        lg = Logger.from_queue_config(log_config, name=f"agent/{name}")
        _create_and_run_agent(
            lg, config, llm_config, learn_config, variables, factory_module, bus_config
        )
    except Exception as e:
        # Log to stderr as last resort -- no channel to report back
        import sys

        print(f"Agent {name} failed: {e}", file=sys.stderr)


def _thread_entry(
    name: str,
    config: DotDict,
    llm_config: LLMConfig,
    learn_config: LearnConfig | None,
    variables: dict[str, str],
    lg: Logger,
    factory_module: str,
    bus_config: dict[str, Any] | None = None,
) -> None:
    """Entry point for agent thread."""
    try:
        _create_and_run_agent(
            lg, config, llm_config, learn_config, variables, factory_module, bus_config
        )
    except Exception as e:
        lg.warning("agent thread failed", extra={"exception": e})


def _create_and_run_agent(
    lg: Logger,
    config: DotDict,
    llm_config: LLMConfig,
    learn_config: LearnConfig | None,
    variables: dict[str, str],
    factory_module: str,
    bus_config: dict[str, Any] | None,
) -> None:
    """Create agent from config and run it (shared by subprocess and thread entries)."""
    from llm_gent.bus.transport import WorkerBusConfig
    from llm_gent.core.platform import PlatformContext
    from llm_gent.runtime.runner import AgentRunner

    platform = PlatformContext.from_config(lg=lg, llm_config=llm_config, learn_config=learn_config)
    factory = _load_agent_factory(factory_module, platform)
    agent = factory.create(config, variables=variables)
    agent.start()

    worker_bus_config = WorkerBusConfig(**bus_config) if bus_config else WorkerBusConfig()
    runner = AgentRunner(
        lg=lg,
        agent=agent,
        bus_config=worker_bus_config,
        schedule_interval=_extract_schedule_interval(config),
    )
    runner.run()


def _load_agent_factory(factory_module: str, platform: Any) -> Any:
    """Load agent factory from module."""
    import importlib

    try:
        module = importlib.import_module(factory_module)
        return module.Factory(platform=platform)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load agent factory from {factory_module}: {e}") from e


def _extract_schedule_interval(config: dict[str, Any]) -> float | None:
    """Extract schedule interval from agent config."""
    schedule = config.get("schedule")
    if schedule and isinstance(schedule, dict):
        interval = schedule.get("interval")
        if interval is not None:
            try:
                return float(interval)
            except (ValueError, TypeError):
                return None
    return None
