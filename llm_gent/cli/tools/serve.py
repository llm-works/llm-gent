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

    from llm_gent.core.traits.builtin.learn import LearnConfig
    from llm_gent.runtime import AgentInfo, AgentRegistry, Core
    from llm_gent.runtime.server import AgentServerConfig
    from llm_gent.runtime.server.protocol.base import Request, Response


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
        from llm_gent.runtime.server import AgentServerConfig

        raw_config = self.app.config if self.app.config else DotDict()
        config = AgentServerConfig.from_dict(raw_config)
        self._apply_cli_overrides(config)

        learn_config = self._create_learn_config(config)
        # Note: learn_trait removed - was a template not actually used in shutdown
        registry = self._create_registry()
        core = self._create_core(registry, config, learn_config)
        self._register_agents(registry, core, config)

        self._log_startup(config)
        self._run_server(config, core)
        return 0

    def _create_learn_config(self, config: AgentServerConfig) -> LearnConfig | None:
        """Create LearnConfig from server configuration.

        Creates a template config with global settings (llm, db, embedder).
        Each agent will resolve its own identity from agent YAML.
        """
        from llm_gent.core.traits.builtin.learn import LearnConfig

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

    def _create_registry(self) -> AgentRegistry:
        """Create agent registry."""
        from llm_gent.runtime import AgentRegistry

        return AgentRegistry(lg=self.lg)

    def _create_core(
        self,
        registry: AgentRegistry,
        config: AgentServerConfig,
        learn_config: LearnConfig | None,
    ) -> Core:
        """Create runtime core."""
        from appinfra import DotDict

        from llm_gent.runtime import Core

        return Core(
            lg=self.lg,
            registry=registry,
            llm_config=DotDict(config.llm),
            learn_config=learn_config,
            variables=self._parse_env_vars(),
        )

    def _log_startup(self, config: AgentServerConfig) -> None:
        """Log server startup information."""
        self.lg.info(
            "agent server started",
            extra={
                "host": config.server.host,
                "port": config.server.port,
                "agents": list(config.agents.keys()),
            },
        )

    def _run_server(
        self,
        config: AgentServerConfig,
        core: Core,
    ) -> None:
        """Run the server with signal handling."""

        request_q: Queue[Any] = mp.Queue()
        response_q: Queue[Any] = mp.Queue()
        shutdown_state = {"requested": False}

        def do_shutdown() -> None:
            self._do_shutdown(shutdown_state, core)

        self._install_signal_handlers(do_shutdown)
        server = self._build_server(config, request_q, response_q)

        try:
            process = server.start_subprocess()

            def is_shutdown() -> bool:
                return shutdown_state["requested"] or not process.is_alive()

            self._ipc_loop(core, request_q, response_q, is_shutdown)
            process.join()
        finally:
            do_shutdown()

    def _do_shutdown(
        self,
        state: dict[str, bool],
        core: Core,
    ) -> None:
        """Execute shutdown sequence (idempotent)."""
        if state["requested"]:
            return
        state["requested"] = True
        core.shutdown()

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

        from llm_gent.runtime.server.management import create_management_routes

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
        core: Core,
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
                response = self._handle_request(core, request)
                response_q.put(response)
            except Empty:
                pass
            except Exception:
                self.lg.exception("IPC processing error")
                if request is not None:
                    from llm_gent.runtime.server.protocol.base import Response

                    error_resp = Response(
                        id=getattr(request, "id", "unknown"),
                        success=False,
                        error="Internal server error",
                    )
                    response_q.put(error_resp)

    def _get_ipc_handlers(self) -> dict[str, Callable[[Core, Request], Response]]:
        """Get mapping of message types to handlers."""
        from llm_gent.runtime.server.protocol.management import (
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

    def _handle_request(self, core: Core, request: Request) -> Response:
        """Dispatch IPC request to appropriate handler."""
        from llm_gent.runtime.server.protocol.base import Response

        handler = self._get_ipc_handlers().get(request.message_type)
        if handler is None:
            return Response(
                id=request.id,
                success=False,
                error=f"Unknown message type: {request.message_type}",
            )

        return handler(core, request)

    def _handle_health(self, core: Core, request: Request) -> Response:
        """Handle health check request."""
        from llm_gent.runtime.server.protocol.management import MgmtHealthResponse

        return MgmtHealthResponse(
            id=request.id,
            status="ok",
            agent_count=len(core.registry.list_agents()),
        )

    def _handle_list_agents(self, core: Core, request: Request) -> Response:
        """Handle list agents request."""
        from llm_gent.runtime.server.protocol.management import ListAgentsResponse

        agents = core.registry.list_agents()
        return ListAgentsResponse(
            id=request.id,
            agents=[self._agent_info_to_dict(a) for a in agents],
        )

    def _handle_get_agent(self, core: Core, request: Request) -> Response:
        """Handle get agent request."""
        from llm_gent.runtime.handle import AgentInfo
        from llm_gent.runtime.server.protocol.management import (
            GetAgentRequest,
            GetAgentResponse,
        )

        req = (
            request
            if isinstance(request, GetAgentRequest)
            else GetAgentRequest(**request.model_dump())
        )
        handle = core.registry.get(req.agent_name)
        if handle is None:
            return GetAgentResponse(
                id=req.id,
                success=False,
                error=f"Agent not found: {req.agent_name}",
            )

        info = AgentInfo.from_handle(handle)
        return GetAgentResponse(
            id=req.id,
            name=info.name,
            status=info.status,
            cycle_count=info.cycle_count,
            last_run=info.last_run.isoformat() if info.last_run else None,
            error=info.error,
            schedule_interval=info.schedule_interval,
        )

    def _handle_start_agent(self, core: Core, request: Request) -> Response:
        """Handle start agent request."""
        from llm_gent.runtime.server.protocol.management import (
            StartAgentRequest,
            StartAgentResponse,
        )

        req = (
            request
            if isinstance(request, StartAgentRequest)
            else StartAgentRequest(**request.model_dump())
        )
        try:
            info = core.start(req.agent_name)
            return StartAgentResponse(
                id=req.id,
                name=info.name,
                status=info.status,
                cycle_count=info.cycle_count,
                last_run=info.last_run.isoformat() if info.last_run else None,
                error=info.error,
                schedule_interval=info.schedule_interval,
            )
        except KeyError:
            return StartAgentResponse(
                id=req.id,
                success=False,
                error=f"Agent not found: {req.agent_name}",
            )
        except ValueError as e:
            return StartAgentResponse(id=req.id, success=False, error=str(e))

    def _handle_stop_agent(self, core: Core, request: Request) -> Response:
        """Handle stop agent request."""
        from llm_gent.runtime.server.protocol.management import (
            StopAgentRequest,
            StopAgentResponse,
        )

        req = (
            request
            if isinstance(request, StopAgentRequest)
            else StopAgentRequest(**request.model_dump())
        )
        try:
            info = core.stop(req.agent_name)
            return StopAgentResponse(
                id=req.id,
                name=info.name,
                status=info.status,
                cycle_count=info.cycle_count,
                last_run=info.last_run.isoformat() if info.last_run else None,
                error=info.error,
                schedule_interval=info.schedule_interval,
            )
        except KeyError:
            return StopAgentResponse(
                id=req.id,
                success=False,
                error=f"Agent not found: {req.agent_name}",
            )

    def _handle_ask_agent(self, core: Core, request: Request) -> Response:
        """Handle ask agent request."""
        from llm_gent.runtime.server.protocol.management import (
            AskAgentRequest,
            AskAgentResponse,
        )

        req = (
            request
            if isinstance(request, AskAgentRequest)
            else AskAgentRequest(**request.model_dump())
        )
        try:
            response = core.ask(req.agent_name, req.question)
            return AskAgentResponse(id=req.id, response=response)
        except KeyError:
            return AskAgentResponse(
                id=req.id,
                success=False,
                error=f"Agent not found: {req.agent_name}",
            )
        except RuntimeError as e:
            return AskAgentResponse(id=req.id, success=False, error=str(e))

    def _handle_feedback_agent(self, core: Core, request: Request) -> Response:
        """Handle feedback request."""
        from llm_gent.runtime.server.protocol.management import (
            FeedbackAgentRequest,
            FeedbackAgentResponse,
        )

        req = (
            request
            if isinstance(request, FeedbackAgentRequest)
            else FeedbackAgentRequest(**request.model_dump())
        )
        try:
            core.feedback(req.agent_name, req.message)
            return FeedbackAgentResponse(id=req.id)
        except KeyError:
            return FeedbackAgentResponse(
                id=req.id,
                success=False,
                error=f"Agent not found: {req.agent_name}",
            )
        except RuntimeError as e:
            return FeedbackAgentResponse(id=req.id, success=False, error=str(e))

    def _handle_get_insights(self, core: Core, request: Request) -> Response:
        """Handle get insights request."""
        from llm_gent.runtime.server.protocol.management import (
            GetInsightsRequest,
            GetInsightsResponse,
        )

        req = (
            request
            if isinstance(request, GetInsightsRequest)
            else GetInsightsRequest(**request.model_dump())
        )
        try:
            insights = core.get_insights(req.agent_name, req.limit)
            return GetInsightsResponse(id=req.id, insights=insights)
        except KeyError:
            return GetInsightsResponse(
                id=req.id,
                success=False,
                error=f"Agent not found: {req.agent_name}",
            )
        except RuntimeError as e:
            return GetInsightsResponse(id=req.id, success=False, error=str(e))

    def _agent_info_to_dict(self, info: AgentInfo) -> dict[str, Any]:
        """Convert AgentInfo to dict for response."""
        return {
            "name": info.name,
            "status": info.status,
            "cycle_count": info.cycle_count,
            "last_run": info.last_run.isoformat() if info.last_run else None,
            "error": info.error,
            "schedule_interval": info.schedule_interval,
        }

    def _register_agents(
        self, registry: AgentRegistry, core: Core, config: AgentServerConfig
    ) -> None:
        """Register and auto-start agents from configuration."""
        for name, agent_config in config.agents.items():
            config_dict = self._build_agent_config_dict(name, agent_config)
            try:
                registry.register(name, config_dict)
                # Auto-start agents with a schedule
                if agent_config.schedule is not None:
                    core.start(name)
            except Exception as e:
                self.lg.error("failed to register agent", extra={"agent": name, "exception": e})

    def _build_agent_config_dict(self, name: str, agent_config: Any) -> DotDict:
        """Build config DotDict for agent registration.

        For programmatic agents, includes module, factory, identity, config.
        For prompt agents, includes task, tools, conversation, events.
        """
        config_dict = DotDict()
        config_dict["name"] = name
        config_dict["type"] = agent_config.type_
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
