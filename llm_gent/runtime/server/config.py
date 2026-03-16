"""Typed configuration for agent server.

Parses raw YAML config into typed Pydantic models.
"""

from __future__ import annotations

from typing import Any, Literal

from appinfra.app.fastapi.config import ApiConfig
from pydantic import BaseModel, Field

from ...core.traits.builtin.directive import Directive


class LearnBackendConfig(BaseModel):
    """Learn trait backend configuration.

    Note: profile_id is deprecated. Each agent now specifies its own
    identity configuration in agent YAML (name and optional context_key).
    """

    db: dict[str, Any]
    """Database configuration dict (url, extensions, etc.)."""
    profile_id: str | None = None
    """Legacy profile ID (deprecated, use per-agent identity config instead)."""
    embedder_url: str | None = None
    embedder_model: str = "default"
    embedder_timeout: float = 30.0
    """Embedder timeout in seconds."""
    training: dict[str, Any] | None = None
    """Training configuration (default profiles, etc.)."""
    adapters: dict[str, Any] | None = None
    """Adapter configuration (lora base_path, etc.)."""


class TaskConfigYAML(BaseModel):
    """Task configuration from YAML."""

    description: str
    output_schema: dict[str, Any] | None = None
    max_iterations: int = 10
    """Maximum LLM round-trips before stopping. 0 means no limit."""
    timeout_secs: float = 0
    """Maximum seconds for task execution. 0 means no limit."""


class ScheduleConfigYAML(BaseModel):
    """Schedule configuration from YAML."""

    interval: int
    """Seconds between automatic executions."""


class EventHandlerConfig(BaseModel):
    """Event handler configuration for declarative behavior.

    Defines how the agent should behave for specific events (schedule, question).
    Maps to memory strategies and prompt composition.
    """

    recall_strategy: Literal["chronological", "semantic"] = "chronological"
    """Memory recall strategy: 'chronological' for recency, 'semantic' for relevance."""

    recall_limit: int = 5
    """Maximum number of past solutions to recall."""

    compose_layers: list[str] = ["identity", "context", "task"]
    """Prompt composition layers (identity, method, context, task/question)."""


class AgentConfigYAML(BaseModel):
    """Configuration for a single agent from YAML.

    Example:
        codebase-explorer:
          class: prompt
          identity: |
            You are a codebase exploration agent.
            You value curiosity over speed, insight over coverage.
          method: |
            - Start with high-level structure
            - Read important files first
            - Follow interesting patterns deeper
          task:
            description: Explore the codebase
          tools:
            shell: {allowed_commands: [grep, find]}
          schedule:
            interval: 600
    """

    type_: Literal["prompt", "programmatic"] = Field(alias="type", default="prompt")
    """Agent type: 'prompt' for YAML-only, 'programmatic' for custom Python."""

    module: str | None = None
    """Module path for programmatic agents (e.g., 'llm_gent.agents.jokester_p')."""

    factory: str = "Factory"
    """Factory class name for programmatic agents (default: 'Factory')."""

    identity: dict[str, Any] = {}
    """Identity configuration for agent (name and optional context_key)."""

    config: dict[str, Any] = {}
    """Agent-specific configuration passed to __init__."""

    directive: Directive | str | None = None
    """Agent's directive - why it exists (string or Directive object)."""

    method: str | None = None
    """Agent's operational method - how it works."""

    task: TaskConfigYAML
    """Primary task configuration."""

    tools: dict[str, dict[str, Any]] = {}
    """Tool configurations keyed by tool type."""

    schedule: ScheduleConfigYAML | None = None
    """Optional automatic execution schedule."""

    conversation: dict[str, Any] = {}
    """Conversation settings (max_tokens, compact_threshold, etc.)."""

    events: dict[str, EventHandlerConfig] = {}
    """Event handlers keyed by event name (schedule, question)."""

    # Allow extra fields (rating, max_retries, etc.) to pass through from YAML
    model_config = {"populate_by_name": True, "extra": "allow"}


class AgentServerConfig(BaseModel):
    """Complete server configuration.

    Loaded from etc/llm-gent.yaml:

        server:
          host: 0.0.0.0
          port: 8080
          uvicorn: !include './uvicorn.yaml'

        llm: !include './llm.yaml'

        learn:
          db: !include './infra.yaml#dbs.main'

        agents:
          codebase-explorer:
            class: prompt
            ...
    """

    model_config = {"arbitrary_types_allowed": True}

    server: ApiConfig
    """HTTP server settings (appinfra ApiConfig)."""

    llm: dict[str, Any]
    """LLM backend configuration (see llm-infer LLMClient.from_config)."""

    learn: LearnBackendConfig | None = None
    """Optional learn backend for memory/feedback."""

    agents: dict[str, AgentConfigYAML] = {}
    """Agent definitions keyed by name."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentServerConfig:
        """Create config from raw YAML dict.

        Args:
            raw: Raw configuration dictionary from YAML.

        Returns:
            Validated AgentServerConfig.

        Raises:
            ValidationError: If config is invalid.
        """
        from appinfra.app.fastapi.config import UvicornConfig

        # Parse server config into ApiConfig
        server_raw = raw.get("server", {})
        uvicorn_raw = server_raw.get("uvicorn", {})

        server = ApiConfig(
            host=server_raw.get("host", "0.0.0.0"),
            port=server_raw.get("port", 8080),
            response_timeout=server_raw.get("response_timeout", 60.0),
            uvicorn=UvicornConfig(**uvicorn_raw) if uvicorn_raw else UvicornConfig(),
        )

        return cls(
            server=server,
            llm=raw.get("llm", {}),
            learn=raw.get("learn"),
            agents=raw.get("agents", {}),
        )
