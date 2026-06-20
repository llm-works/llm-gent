"""Serve tool - starts the agent gateway server."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
from collections.abc import Callable
from queue import Empty
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.app.fastapi import ServerBuilder
from appinfra.app.tools import Tool, ToolConfig


if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from ...core.traits.builtin.learn import LearnConfig
    from ...hub import Hub
    from ...runtime.server import AgentServerConfig
    from ...runtime.server.protocol.base import Request, Response


class ServeTool(Tool):
    """Start the agent gateway server."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="serve", help_text="Start the agent gateway server")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--host",
            help="Host to bind to (overrides config)",
        )
        parser.add_argument(
            "--port",
            type=int,
            help="Port to bind to (overrides config)",
        )
        parser.add_argument(
            "-e",
            "--env",
            action="append",
            metavar="KEY=VALUE",
            dest="env_vars",
            help="Set environment variable for agent configs (e.g., -e CODEBASE_PATH=/path/to/code)",
        )

    def _parse_env_vars(self) -> dict[str, str]:
        """Parse -e KEY=VALUE arguments into a dict."""
        env_vars = {}
        if self.args.env_vars:
            for item in self.args.env_vars:
                if "=" in item:
                    key, value = item.split("=", 1)
                    env_vars[key] = value
        return env_vars

    def run(self, **kwargs: Any) -> int:
        from ...runtime.server import AgentServerConfig

        raw_config = self.app.config if self.app.config else DotDict()
        config = AgentServerConfig.from_dict(raw_config)
        self._apply_cli_overrides(config)

        learn_config = self._create_learn_config(config)
        hub = self._create_hub(config, learn_config)
        hub.start()

        try:
            self._start_agents(hub, config)
            self._log_startup(config, hub)
            self._run_server(config, hub)
        finally:
            hub.stop()

        return 0

    def _create_learn_config(self, config: AgentServerConfig) -> LearnConfig | None:
        """Create LearnConfig from server configuration.

        Creates a template config with global settings (llm, db, embedder).
        Each agent will resolve its own identity from agent YAML.
        """
        from ...core.traits.builtin.learn import LearnConfig

        if config.learn is None:
            return None

        return LearnConfig(
            llm=config.llm,
            db=config.learn.db,
            embedder_url=config.learn.embedder_url,
            embedder_model=config.learn.embedder_model,
            embedder_timeout=config.learn.embedder_timeout,
            training=config.learn.training,
            adapters=config.learn.adapters,
            # Note: identity is set per-agent in factory
        )

    def _create_hub(self, config: AgentServerConfig, learn_config: LearnConfig | None) -> Hub:
        """Create and configure the swarm hub."""
        from appinfra import DotDict

        from ...bus.transport import CoordinatorBusConfig, WorkerBusConfig
        from ...hub import Hub, HubConfig

        hub_yaml = config.hub
        hub_config = HubConfig(
            bus=CoordinatorBusConfig(
                router_port=hub_yaml.router_port,
                pub_port=hub_yaml.pub_port,
                sub_port=hub_yaml.sub_port,
            ),
            dead_timeout=hub_yaml.dead_timeout,
            health_check_interval=hub_yaml.health_check_interval,
            max_restarts=hub_yaml.max_restarts,
        )
        bus_config = WorkerBusConfig(
            router_port=hub_yaml.router_port,
            pub_port=hub_yaml.pub_port,
            sub_port=hub_yaml.sub_port,
        )

        return Hub(
            lg=self.lg,
            config=hub_config,
            bus_config=bus_config,
            llm_config=DotDict(config.llm),
            learn_config=learn_config,
            variables=self._parse_env_vars(),
        )

    def _log_startup(self, config: AgentServerConfig, hub: Hub) -> None:
        """Log server startup information."""
        agents = [a.id for a in hub.registry.list_agents()]
        self.lg.info(
            "agent server started",
            extra={
                "host": config.server.host,
                "port": config.server.port,
                "agents": agents,
                "bus_router_port": config.hub.router_port,
            },
        )

    def _run_server(
        self,
        config: AgentServerConfig,
        hub: Hub,
    ) -> None:
        """Run the server with signal handling."""

        request_q: Queue[Any] = mp.Queue()
        response_q: Queue[Any] = mp.Queue()
        shutdown_state = {"requested": False}

        def do_shutdown() -> None:
            if not shutdown_state["requested"]:
                shutdown_state["requested"] = True
                hub.stop()

        self._install_signal_handlers(do_shutdown)
        server = self._build_server(config, request_q, response_q)

        try:
            process = server.start_subprocess()

            def is_shutdown() -> bool:
                return shutdown_state["requested"] or not process.is_alive()

            self._ipc_loop(hub, request_q, response_q, is_shutdown)
            process.join()
        finally:
            do_shutdown()

    def _install_signal_handlers(self, do_shutdown: Callable[[], None]) -> None:
        """Install signal handlers for graceful shutdown.

        Note: do_shutdown() calls core.shutdown() which uses bounded timeouts
        (5s graceful + 2s terminate + kill) per agent, so this won't hang indefinitely.
        """

        def shutdown_handler(signum: int, frame: Any) -> None:
            self.lg.info("shutdown signal received")
            do_shutdown()

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

    def _build_server(
        self,
        config: AgentServerConfig,
        request_q: Queue[Any],
        response_q: Queue[Any],
    ) -> Any:
        """Build the FastAPI server with subprocess mode and IPC."""
        from appinfra.app.fastapi.runtime.server import Server

        from ...runtime.server.management import create_management_routes

        server: Server = (
            ServerBuilder(self.lg, "agent-gateway")
            .with_config(config.server)
            .subprocess.with_ipc(request_q, response_q)
            .done()
            .routes.with_router(create_management_routes())
            .done()
            .uvicorn.with_config(config.server.uvicorn)
            .done()
            .build()
        )
        return server

    def _ipc_loop(
        self,
        hub: Hub,
        request_q: Queue[Any],
        response_q: Queue[Any],
        is_shutdown: Callable[[], bool],
        poll_timeout: float = 0.1,
    ) -> None:
        """Process IPC requests from FastAPI subprocess."""
        while not is_shutdown():
            request = None
            try:
                request = request_q.get(timeout=poll_timeout)
                response = self._handle_request(hub, request)
                response_q.put(response)
            except Empty:
                pass
            except Exception:
                self.lg.exception("IPC processing error")
                if request is not None:
                    from ...runtime.server.protocol.base import Response

                    error_resp = Response(
                        id=getattr(request, "id", "unknown"),
                        success=False,
                        error="Internal server error",
                    )
                    response_q.put(error_resp)

    def _get_ipc_handlers(self) -> dict[str, Callable[..., Response]]:
        """Get mapping of message types to handlers."""
        from ...runtime.server.protocol.management import (
            AskAgentRequest,
            FeedbackAgentRequest,
            GetAgentRequest,
            GetInsightsRequest,
            ListAgentsRequest,
            MgmtHealthRequest,
            StartAgentRequest,
            StopAgentRequest,
        )

        return {
            MgmtHealthRequest.message_type: self._handle_health,
            ListAgentsRequest.message_type: self._handle_list_agents,
            GetAgentRequest.message_type: self._handle_get_agent,
            StartAgentRequest.message_type: self._handle_start_agent,
            StopAgentRequest.message_type: self._handle_stop_agent,
            AskAgentRequest.message_type: self._handle_ask_agent,
            FeedbackAgentRequest.message_type: self._handle_feedback_agent,
            GetInsightsRequest.message_type: self._handle_get_insights,
        }

    def _handle_request(self, hub: Hub, request: Request) -> Response:
        """Dispatch IPC request to appropriate handler."""
        from ...runtime.server.protocol.base import Response

        handler = self._get_ipc_handlers().get(request.message_type)
        if handler is None:
            return Response(
                id=request.id,
                success=False,
                error=f"Unknown message type: {request.message_type}",
            )

        return handler(hub, request)

    def _handle_health(self, hub: Hub, request: Request) -> Response:
        """Handle health check request."""
        from ...runtime.server.protocol.management import MgmtHealthResponse

        return MgmtHealthResponse(
            id=request.id,
            status="ok",
            agent_count=hub.registry.count,
        )

    def _handle_list_agents(self, hub: Hub, request: Request) -> Response:
        """Handle list agents request."""
        from ...runtime.server.protocol.management import ListAgentsResponse

        agents = hub.registry.list_agents()
        return ListAgentsResponse(
            id=request.id,
            agents=[self._agent_entry_to_dict(a) for a in agents],
        )

    def _handle_get_agent(self, hub: Hub, request: Request) -> Response:
        """Handle get agent request."""
        from ...runtime.server.protocol.management import (
            GetAgentRequest,
            GetAgentResponse,
        )

        req = (
            request
            if isinstance(request, GetAgentRequest)
            else GetAgentRequest(**request.model_dump())
        )
        entry = hub.registry.get(req.agent_name)
        if entry is None:
            return GetAgentResponse(
                id=req.id,
                success=False,
                error=f"Agent not found: {req.agent_name}",
            )

        return GetAgentResponse(
            id=req.id,
            name=entry.id,
            status=entry.health.value,
            cycle_count=entry.stats.ticks,
            last_run=entry.last_run.isoformat() if entry.last_run else None,
            error=entry.error,
            schedule_interval=int(entry.schedule_interval) if entry.schedule_interval else None,
        )

    def _handle_start_agent(self, hub: Hub, request: Request) -> Response:
        """Handle start agent request."""
        from ...runtime.server.protocol.management import StartAgentResponse

        # TODO: start_agent needs config -- this HTTP handler needs rework
        return StartAgentResponse(
            id=request.id, success=False, error="Dynamic start not yet supported via HTTP"
        )

    def _handle_stop_agent(self, hub: Hub, request: Request) -> Response:
        """Handle stop agent request."""
        from ...runtime.server.protocol.management import (
            StopAgentRequest,
            StopAgentResponse,
        )

        req = (
            request
            if isinstance(request, StopAgentRequest)
            else StopAgentRequest(**request.model_dump())
        )
        try:
            hub.stop_agent(req.agent_name)
            return StopAgentResponse(id=req.id, name=req.agent_name, status="stopped")
        except KeyError:
            return StopAgentResponse(
                id=req.id, success=False, error=f"Agent not found: {req.agent_name}"
            )

    def _handle_ask_agent(self, hub: Hub, request: Request) -> Response:
        """Handle ask agent request."""
        from ...runtime.server.protocol.management import (
            AskAgentRequest,
            AskAgentResponse,
        )

        req = (
            request
            if isinstance(request, AskAgentRequest)
            else AskAgentRequest(**request.model_dump())
        )
        try:
            response = hub.ask(req.agent_name, req.question)
            return AskAgentResponse(id=req.id, response=response)
        except Exception as e:
            return AskAgentResponse(id=req.id, success=False, error=str(e))

    def _handle_feedback_agent(self, hub: Hub, request: Request) -> Response:
        """Handle feedback request."""
        from ...runtime.server.protocol.management import (
            FeedbackAgentRequest,
            FeedbackAgentResponse,
        )

        req = (
            request
            if isinstance(request, FeedbackAgentRequest)
            else FeedbackAgentRequest(**request.model_dump())
        )
        try:
            hub.feedback(req.agent_name, req.message)
            return FeedbackAgentResponse(id=req.id)
        except Exception as e:
            return FeedbackAgentResponse(id=req.id, success=False, error=str(e))

    def _handle_get_insights(self, hub: Hub, request: Request) -> Response:
        """Handle get insights request."""
        from ...runtime.server.protocol.management import (
            GetInsightsRequest,
            GetInsightsResponse,
        )

        req = (
            request
            if isinstance(request, GetInsightsRequest)
            else GetInsightsRequest(**request.model_dump())
        )
        insights = hub.get_insights(req.agent_name, limit=req.limit)
        return GetInsightsResponse(id=req.id, insights=insights)

    def _agent_entry_to_dict(self, entry: Any) -> dict[str, Any]:
        """Convert AgentEntry to dict for response."""
        return {
            "name": entry.id,
            "status": entry.health.value,
            "cycle_count": entry.stats.ticks,
            "last_run": entry.last_run.isoformat() if entry.last_run else None,
            "error": entry.error,
            "schedule_interval": entry.schedule_interval,
        }

    def _start_agents(self, hub: Hub, config: AgentServerConfig) -> None:
        """Start agents from configuration."""
        for name, agent_config in config.agents.items():
            if not agent_config.enabled:
                self.lg.debug("skipping disabled agent", extra={"agent": name})
                continue

            config_dict = self._build_agent_config_dict(name, agent_config)
            try:
                if agent_config.schedule is not None:
                    hub.start_agent(name, config_dict)
                else:
                    # Message-only agent: start with runner for ask/feedback
                    hub.start_agent(name, config_dict)
            except Exception as e:
                self.lg.error("failed to start agent", extra={"agent": name, "exception": e})

    def _build_agent_config_dict(self, name: str, agent_config: Any) -> DotDict:
        """Build config DotDict for agent registration.

        For programmatic agents, includes module, factory, identity, config.
        For prompt agents, includes task, tools, conversation, events.
        """
        config_dict = DotDict()
        config_dict["name"] = name
        config_dict["type"] = agent_config.type_
        config_dict["execution"] = agent_config.execution
        config_dict["task"] = agent_config.task.model_dump()

        # Add type-specific fields
        self._add_type_specific_fields(config_dict, agent_config)

        # Common optional fields
        self._add_optional_fields(config_dict, agent_config)

        # Add extra fields from YAML (rating, max_retries, similarity_threshold, etc.)
        # These are fields not explicitly defined in AgentConfigYAML schema
        if hasattr(agent_config, "__pydantic_extra__") and agent_config.__pydantic_extra__:
            config_dict.update(agent_config.__pydantic_extra__)

        return config_dict

    def _add_type_specific_fields(self, config_dict: DotDict, agent_config: Any) -> None:
        """Add type-specific fields to config dict."""
        # Identity is common to all agent types
        config_dict["identity"] = agent_config.identity

        if agent_config.type_ == "programmatic":
            config_dict["module"] = agent_config.module
            config_dict["factory"] = agent_config.factory
            config_dict["config"] = agent_config.config
        else:
            # Prompt agents use conversation and events
            config_dict["conversation"] = agent_config.conversation
            if agent_config.events:
                config_dict["events"] = {
                    name: handler.model_dump() for name, handler in agent_config.events.items()
                }

    def _add_optional_fields(self, config_dict: DotDict, agent_config: Any) -> None:
        """Add optional fields to config dict."""
        if agent_config.directive is not None:
            config_dict["directive"] = (
                agent_config.directive
                if isinstance(agent_config.directive, str)
                else agent_config.directive.model_dump()
            )

        if agent_config.method is not None:
            config_dict["method"] = agent_config.method

        if agent_config.tools:
            config_dict["tools"] = agent_config.tools

        if agent_config.schedule is not None:
            config_dict["schedule"] = agent_config.schedule.model_dump()

    def _apply_cli_overrides(self, config: Any) -> None:
        """Apply command-line overrides to config."""
        if self.args.host:
            config.server.host = self.args.host
        if self.args.port:
            config.server.port = self.args.port
