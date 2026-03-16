"""HTTP server for agent communication using FastAPI.

Uses appinfra's subprocess + IPC pattern: FastAPI runs in a subprocess,
handlers forward requests to main process via queues.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger


if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from fastapi import APIRouter


@dataclass
class HTTPServerConfig:
    """Configuration for HTTP server.

    Attributes:
        host: Host to bind to.
        port: Port to bind to.
        title: API title for OpenAPI docs.
        description: API description for OpenAPI docs.
        version: API version for OpenAPI docs.
        auto_restart: Whether to auto-restart on crash.
        max_restarts: Maximum restart attempts (0 = unlimited).
        response_timeout: Request timeout in seconds.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    title: str = "Agent API"
    description: str = "Agent HTTP API"
    version: str = "1.0.0"
    auto_restart: bool = True
    max_restarts: int = 5
    response_timeout: float = 60.0


class HTTPServer:
    """HTTP server for agent communication.

    Uses appinfra's FastAPI framework with subprocess isolation.
    FastAPI runs in a subprocess, handlers forward requests to main process
    via IPC queues for processing.

    Architecture:
        HTTP Request -> FastAPI subprocess -> request_q -> Main process
        HTTP Response <- FastAPI subprocess <- response_q <- Main process

    Usage:
        def create_routes() -> APIRouter:
            router = APIRouter(prefix="/v1")
            # ... define routes ...
            return router

        server = HTTPServer(lg, HTTPServerConfig(port=8080), create_routes)
        server.start()  # Non-blocking, starts subprocess

        # Main process reads requests and sends responses
        while running:
            request = server.request_queue.get(timeout=0.1)
            response = handle(request)
            server.response_queue.put(response)

        server.stop()
    """

    def __init__(
        self,
        lg: Logger,
        config: HTTPServerConfig,
        router_factory: Callable[[], APIRouter],
    ) -> None:
        """Initialize the HTTP server.

        Args:
            lg: Logger instance.
            config: Server configuration.
            router_factory: Function that creates the APIRouter for this server.
        """
        self._lg = lg
        self._config = config
        self._router_factory = router_factory
        self._request_q: Queue[Any] = mp.Queue()
        self._response_q: Queue[Any] = mp.Queue()
        self._server: Any = None  # appinfra.app.fastapi.Server (external type)
        self._process: mp.Process | None = None

    @property
    def config(self) -> HTTPServerConfig:
        """Server configuration."""
        return self._config

    @property
    def request_queue(self) -> Queue[Any]:
        """Queue for receiving requests from FastAPI subprocess."""
        return self._request_q

    @property
    def response_queue(self) -> Queue[Any]:
        """Queue for sending responses to FastAPI subprocess."""
        return self._response_q

    @property
    def is_running(self) -> bool:
        """Check if server subprocess is running."""
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        """Start the server in subprocess (non-blocking)."""
        self._server = self._build_server()
        self._process = self._server.start_subprocess()

    def stop(self) -> None:
        """Stop the server subprocess."""
        if self._server:
            self._server.stop()
            self._server = None
            self._process = None

    def _build_server(self) -> Any:
        """Build the FastAPI server with subprocess mode."""
        from appinfra.app.fastapi import ServerBuilder

        router = self._router_factory()

        return (
            ServerBuilder(self._lg, self._config.title)
            .with_host(self._config.host)
            .with_port(self._config.port)
            .with_title(self._config.title)
            .with_description(self._config.description)
            .with_version(self._config.version)
            .subprocess.with_ipc(self._request_q, self._response_q)
            .with_response_timeout(self._config.response_timeout)
            .with_auto_restart(
                enabled=self._config.auto_restart,
                max_restarts=self._config.max_restarts,
            )
            .done()
            .routes.with_router(router)
            .done()
            .build()
        )
