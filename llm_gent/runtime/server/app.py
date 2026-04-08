"""FastAPI application factory.

Creates the FastAPI app with routes configured for the runtime Core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from .management import create_management_routes


if TYPE_CHECKING:
    from .. import Core


def create_app(
    core: Core,
    title: str = "Agent Gateway",
    bus_config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create FastAPI application with routes.

    Args:
        core: Runtime core for managing agents.
        title: API title for OpenAPI docs.
        bus_config: Bus connection config for external agent discovery.
            When provided, a ``GET /bus/config`` endpoint is registered
            so external agents can discover how to connect to the swarm.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        description="Gateway for managing multiple LLM agents",
        version="1.0.0",
    )

    # Store core in app state for route handlers
    app.state.core = core

    # Include management routes
    app.include_router(create_management_routes())

    # Bus config discovery endpoint for external agents
    if bus_config is not None:
        _frozen = dict(bus_config)

        @app.get("/bus/config")
        async def get_bus_config() -> dict[str, Any]:
            """Return bus connection config for external agents."""
            return _frozen

    return app
