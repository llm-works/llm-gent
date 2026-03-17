"""Runtime core - orchestrates agent subprocess/thread lifecycle.

Core is the runtime orchestrator that manages:
- Subprocess/thread spawning and termination
- Communication with running agents
- Log queue for subprocess logging
"""

from __future__ import annotations

import multiprocessing as mp
from datetime import UTC, datetime
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any, cast

from appinfra import DotDict
from appinfra.log.mp import LogQueueListener
from appinfra.service import (
    ChannelError,
    ChannelFactory,
    ChannelTimeoutError,
    State,
)

from .handle import AgentHandle, AgentInfo
from .messages import AgentMessage, MessageType


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_gent.core.traits.builtin.learn import LearnConfig
    from llm_gent.core.traits.builtin.llm import LLMConfig
    from llm_gent.runtime.registry import AgentRegistry


class Core:
    """Runtime core - orchestrates agent subprocess lifecycle.

    Core manages the runtime aspects of agents:
    - Starting/stopping agent subprocesses
    - Communication with running agents (ask, feedback, insights)
    - Centralized log queue for subprocess logging
    - Graceful shutdown

    It works with AgentRegistry which holds the agent configurations.
    """

    def __init__(
        self,
        lg: Logger,
        registry: AgentRegistry,
        llm_config: LLMConfig,
        learn_config: DotDict | None = None,
        variables: dict[str, str] | None = None,
        factory_module: str = "llm_gent.agents.default",
    ) -> None:
        """Initialize the runtime core.

        Args:
            lg: Logger instance.
            registry: Agent registry for handle lookup.
            llm_config: LLM configuration for agents.
            learn_config: Optional learn configuration (DotDict/LearnConfig).
            variables: Variable substitutions for agent configs.
            factory_module: Module containing the agent Factory class.
        """
        self._lg = lg
        self._registry = registry
        self._llm_config = llm_config
        self._learn_config = learn_config
        self._variables = variables or {}
        self._factory_module = factory_module

        # Queue-based logging for subprocesses
        self._log_queue: Queue[Any] = Queue()
        self._log_config = lg.queue_config(self._log_queue)
        self._log_listener = LogQueueListener(self._log_queue, lg)
        self._log_listener.start()

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry for read operations."""
        return self._registry

    def start(self, name: str) -> AgentInfo:
        """Start an agent process.

        Spawns a new subprocess running the agent.

        Args:
            name: Agent name.

        Returns:
            Updated AgentInfo.

        Raises:
            KeyError: If agent not found.
            InvalidTransitionError: If agent cannot be started (wrong state).
        """
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        # Reject duplicate starts - prevent orphaning running workers
        if handle.state in {State.STARTING, State.RUNNING, State.STOPPING}:
            raise RuntimeError(f"Agent already active: {name} (state={handle.state.value})")

        handle.state = State.STARTING
        handle.error = None

        try:
            self._spawn_process(handle)
            handle.state = State.RUNNING
            self._lg.debug("agent runtime started", extra={"agent": name})
        except Exception as e:
            self._cleanup_failed_start(handle)
            handle.state = State.FAILED
            handle.error = str(e)
            self._lg.error("failed to start agent runtime", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def stop(self, name: str) -> AgentInfo:
        """Stop an agent process.

        Sends shutdown message and terminates the subprocess.

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
            self._terminate_process(handle)
            handle.state = State.STOPPED
            self._lg.info("agent stopped", extra={"agent": name})
        except Exception as e:
            handle.state = State.FAILED
            handle.error = str(e)
            self._lg.warning("error stopping agent", extra={"agent": name, "exception": e})

        return AgentInfo.from_handle(handle)

    def ask(self, name: str, question: str, timeout: float = 60.0) -> str:
        """Ask an agent a question.

        Args:
            name: Agent name.
            question: Question to ask.
            timeout: Response timeout in seconds.

        Returns:
            Agent's response.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running.
            TimeoutError: If no response within timeout.
        """
        handle = self._get_running_handle(name)
        assert handle.channel is not None  # Verified by _get_running_handle

        self._lg.debug("ask request", extra={"agent": name, "question_len": len(question)})

        request = AgentMessage(type=MessageType.ASK, payload={"question": question})
        try:
            response = handle.channel.submit(request, timeout=timeout)
        except ChannelError as e:
            self._lg.warning("ask failed", extra={"agent": name, "error": str(e)})
            raise RuntimeError(str(e)) from e

        self._lg.debug("ask completed", extra={"agent": name})
        return str(response.payload.get("response", ""))

    def feedback(self, name: str, message: str, timeout: float = 30.0) -> None:
        """Send feedback to an agent.

        Args:
            name: Agent name.
            message: Feedback message.
            timeout: Response timeout in seconds.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running or feedback failed.
        """
        handle = self._get_running_handle(name)
        assert handle.channel is not None  # Verified by _get_running_handle

        self._lg.debug("feedback request", extra={"agent": name})

        request = AgentMessage(type=MessageType.FEEDBACK, payload={"message": message})
        try:
            handle.channel.submit(request, timeout=timeout)
        except ChannelError as e:
            self._lg.warning("feedback failed", extra={"agent": name, "error": str(e)})
            raise RuntimeError(str(e)) from e

        self._lg.debug("feedback sent", extra={"agent": name})

    def get_insights(
        self, name: str, limit: int = 10, timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Get recent insights from an agent.

        Args:
            name: Agent name.
            limit: Maximum insights to return.
            timeout: Response timeout in seconds.

        Returns:
            List of insight dictionaries.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running.
        """
        handle = self._get_running_handle(name)
        assert handle.channel is not None  # Verified by _get_running_handle

        self._lg.debug("get_insights request", extra={"agent": name, "limit": limit})

        request = AgentMessage(type=MessageType.GET_INSIGHTS, payload={"limit": limit})
        try:
            response = handle.channel.submit(request, timeout=timeout)
        except ChannelError as e:
            self._lg.warning("get_insights failed", extra={"agent": name, "error": str(e)})
            raise RuntimeError(str(e)) from e

        insights = cast(list[dict[str, Any]], response.payload.get("insights", []))

        # Update handle metrics from response
        cycle_count = response.payload.get("cycle_count")
        if cycle_count is not None:
            handle.cycle_count = cycle_count
            handle.last_run = datetime.now(UTC)

        self._lg.debug("get_insights completed", extra={"agent": name, "count": len(insights)})
        return insights

    def shutdown(self) -> None:
        """Shutdown all agents and clean up.

        Stops all running agents gracefully.
        """
        self._lg.info("shutting down core")
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

    def _get_running_handle(self, name: str) -> AgentHandle:
        """Get handle for a running agent.

        Args:
            name: Agent name.

        Returns:
            AgentHandle for the running agent.

        Raises:
            KeyError: If agent not found.
            RuntimeError: If agent not running.
        """
        handle = self._registry.get(name)
        if handle is None:
            raise KeyError(f"Agent not found: {name}")

        if handle.state != State.RUNNING or handle.channel is None:
            raise RuntimeError(f"Agent not running: {name}")

        return handle

    def _cleanup_failed_start(self, handle: AgentHandle) -> None:
        """Clean up IPC resources after a failed start attempt.

        Called when _spawn_process raises an exception. Ensures channel
        and process are properly closed even on partial initialization.
        """
        import contextlib

        if handle.channel is not None:
            with contextlib.suppress(Exception):
                handle.channel.close()
            handle.channel = None

        if handle.process is not None:
            with contextlib.suppress(Exception):
                # Process has terminate/kill, Thread does not
                if isinstance(handle.process, mp.Process):
                    handle.process.terminate()
                    handle.process.join(timeout=1.0)
                    if handle.process.is_alive():
                        handle.process.kill()
                else:
                    # Thread - just wait briefly (daemon thread will exit with main)
                    handle.process.join(timeout=1.0)
            handle.process = None

    def _spawn_process(self, handle: AgentHandle) -> None:
        """Spawn subprocess or thread for agent based on execution mode."""
        execution_mode = handle.config.get("execution", "process")
        self._lg.debug(
            "spawning agent runtime...",
            extra={"agent": handle.name, "execution": execution_mode},
        )

        if execution_mode == "thread":
            self._spawn_thread(handle)
        else:
            self._spawn_subprocess(handle)

    def _spawn_subprocess(self, handle: AgentHandle) -> None:
        """Spawn subprocess for agent."""
        pair = ChannelFactory().create_process_pair()
        handle.channel = pair.parent

        # LearnConfig can be passed directly - DotDict pickles fine
        learn_config = self._learn_config if self._learn_config else None

        # Determine factory module (per-agent for programmatic, default for prompt)
        factory_module = handle.config.get("module", self._factory_module)

        handle.process = mp.Process(
            target=_subprocess_entry,
            args=(
                handle.name,
                self._build_runner_config(handle),
                pair.child,
                self._llm_config,
                learn_config,
                self._variables,
                self._log_config,
                factory_module,  # Per-agent module for programmatic agents
            ),
            name=f"agent-{handle.name}",
            daemon=True,
        )
        handle.process.start()
        self._wait_for_started(pair.parent)
        self._lg.debug("spawned subprocess for agent runtime", extra={"agent": handle.name})

    def _spawn_thread(self, handle: AgentHandle) -> None:
        """Spawn thread for agent (in-process execution)."""
        import threading

        pair = ChannelFactory().create_thread_pair()
        handle.channel = pair.parent

        # Determine factory module (per-agent for programmatic, default for prompt)
        factory_module = handle.config.get("module", self._factory_module)

        handle.process = threading.Thread(
            target=_thread_entry,
            args=(
                handle.name,
                self._build_runner_config(handle),
                pair.child,
                self._llm_config,
                self._learn_config,
                self._variables,
                self._lg,  # Share logger directly in-process
                factory_module,
            ),
            name=f"agent-{handle.name}",
            daemon=True,
        )
        handle.process.start()
        self._wait_for_started(pair.parent)
        self._lg.debug("spawned thread for agent runtime", extra={"agent": handle.name})

    def _wait_for_started(self, channel: Any) -> None:
        """Wait for subprocess to signal it has started."""
        try:
            msg = channel.recv(timeout=30.0)
        except ChannelTimeoutError:
            raise RuntimeError("Agent process did not start within timeout") from None
        if msg.type == MessageType.ERROR:
            raise RuntimeError(msg.payload.get("error", "Unknown startup error"))

    def _build_runner_config(self, handle: AgentHandle) -> DotDict:
        """Build configuration for the runner (keeps DotDict for dot notation)."""
        config = handle.config  # Keep as DotDict
        config["name"] = handle.name  # DotDict supports both dict and dot notation
        return config

    def _terminate_process(self, handle: AgentHandle) -> None:
        """Terminate agent subprocess/thread."""
        import contextlib

        self._lg.debug("terminating process", extra={"agent": handle.name})

        channel = handle.channel
        if channel is not None:
            with contextlib.suppress(Exception):
                channel.send(AgentMessage(type=MessageType.SHUTDOWN))

        if handle.process is not None:
            handle.process.join(timeout=5.0)
            # Process has terminate/kill, Thread does not (daemon=True handles cleanup)
            if handle.process.is_alive() and isinstance(handle.process, mp.Process):
                handle.process.terminate()
                handle.process.join(timeout=2.0)
                if handle.process.is_alive():
                    handle.process.kill()

        # Close channel after process termination to prevent FD leaks
        if channel is not None:
            with contextlib.suppress(Exception):
                channel.close()

        handle.process = None
        handle.channel = None


def _subprocess_entry(
    name: str,
    config: DotDict,
    channel: Any,
    llm_config: LLMConfig,
    learn_config: LearnConfig | None,
    variables: dict[str, str],
    log_config: dict[str, Any],
    factory_module: str,
) -> None:
    """Entry point for agent subprocess.

    This runs in the subprocess - creates logger, agent, and runner, then runs.
    Initialization errors are caught and sent back to the main process via ERROR message.
    """
    from appinfra.log import Logger

    from llm_gent.runtime.messages import AgentMessage, MessageType
    from llm_gent.runtime.runner import AgentRunner

    try:
        lg = Logger.from_queue_config(log_config, name=f"agent/{name}")

        # Create platform context for this agent subprocess
        from llm_gent.core.platform import PlatformContext

        platform = PlatformContext.from_config(
            lg=lg,
            llm_config=llm_config,
            learn_config=learn_config,
        )

        # Load and create agent using new factory architecture
        factory = _load_agent_factory(factory_module, platform)
        agent = factory.create(config, variables=variables)
        agent.start()

        schedule_interval = _extract_schedule_interval(config)
        runner = AgentRunner(
            lg=lg, agent=agent, channel=channel, schedule_interval=schedule_interval
        )
        runner.run()
    except Exception as e:
        # Send error back to main process for faster feedback
        channel.send(AgentMessage(type=MessageType.ERROR, payload={"error": str(e)}))


def _thread_entry(
    name: str,
    config: DotDict,
    channel: Any,
    llm_config: LLMConfig,
    learn_config: LearnConfig | None,
    variables: dict[str, str],
    lg: Logger,
    factory_module: str,
) -> None:
    """Entry point for agent thread (in-process execution).

    Similar to _subprocess_entry but runs in a thread with shared logger.
    """
    from llm_gent.runtime.messages import AgentMessage, MessageType
    from llm_gent.runtime.runner import AgentRunner

    try:
        # Create platform context for this agent thread
        from llm_gent.core.platform import PlatformContext

        platform = PlatformContext.from_config(
            lg=lg,
            llm_config=llm_config,
            learn_config=learn_config,
        )

        # Load and create agent using factory architecture
        factory = _load_agent_factory(factory_module, platform)
        agent = factory.create(config, variables=variables)
        agent.start()

        schedule_interval = _extract_schedule_interval(config)
        runner = AgentRunner(
            lg=lg, agent=agent, channel=channel, schedule_interval=schedule_interval
        )
        runner.run()
    except Exception as e:
        # Send error back to main thread for faster feedback
        channel.send(AgentMessage(type=MessageType.ERROR, payload={"error": str(e)}))


def _load_agent_factory(factory_module: str, platform: Any) -> Any:
    """Load agent factory from module with descriptive error handling.

    Args:
        factory_module: Module path containing Factory class.
        platform: PlatformContext instance.

    Returns:
        Factory instance.

    Raises:
        RuntimeError: If factory cannot be loaded.
    """
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
