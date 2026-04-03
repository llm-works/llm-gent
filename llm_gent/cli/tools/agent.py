"""Agent tool - run agent-specific CLI commands."""

from __future__ import annotations

import argparse
import importlib
from typing import TYPE_CHECKING, Any

from appinfra.app.tools import Tool, ToolConfig


if TYPE_CHECKING:
    from llm_gent.core.agent import Factory


class AgentTool(Tool):
    """Route to agent-specific CLI commands.

    Each agent's Factory can optionally expose a cli_tool class that provides
    agent-specific commands. This tool discovers and delegates to those commands.

    Usage:
        ./llm-gent.py agent <agent-name> <command> [args]

    Example:
        ./llm-gent.py agent my-agent stats
        ./llm-gent.py agent my-agent pairs-sync --dry-run
    """

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="agent", help_text="Run agent-specific CLI commands")
        super().__init__(parent, config)
        self._agent_tool: Tool | None = None

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "agent_name",
            help="Name of the agent",
        )
        # Capture remaining args for the agent's CLI tool
        parser.add_argument(
            "agent_args",
            nargs=argparse.REMAINDER,
            help="Arguments to pass to the agent's CLI tool",
        )

    def configure(self) -> None:
        """Load agent's CLI tool if available."""
        agent_name = self.args.agent_name
        agent_args = self.args.agent_args or []

        # Load cli_tool class from agent's factory
        cli_tool_class = self._get_cli_tool_class(agent_name)
        if cli_tool_class is None:
            return

        # Create and configure the agent's CLI tool
        self._agent_tool = cli_tool_class(parent=self)
        if not self._setup_agent_tool(agent_name, agent_args):
            self._agent_tool = None

    def _get_cli_tool_class(self, agent_name: str) -> type[Tool] | None:
        """Get cli_tool class from agent's factory."""
        factory_module = self._get_factory_module(agent_name)
        if factory_module is None:
            return None

        factory_class = self._load_factory_class(factory_module)
        if factory_class is None:
            return None

        cli_tool_class = getattr(factory_class, "cli_tool", None)
        if cli_tool_class is None:
            print(f"Agent '{agent_name}' does not expose CLI commands.")
            print(f"Factory {factory_module}.Factory has no cli_tool attribute.")
            return None

        return cli_tool_class  # type: ignore[no-any-return]

    def _setup_agent_tool(self, agent_name: str, agent_args: list[str]) -> bool:
        """Parse args and configure the agent tool. Returns True on success."""
        assert self._agent_tool is not None

        help_text = self._agent_tool.config.help_text if self._agent_tool.config else ""
        agent_parser = argparse.ArgumentParser(
            prog=f"agent {agent_name}",
            description=help_text,
        )
        self._agent_tool.add_args(agent_parser)

        try:
            self._agent_tool._parsed_args = agent_parser.parse_args(agent_args)  # type: ignore[attr-defined]
        except SystemExit:
            return False  # argparse exits on --help or errors

        self._agent_tool._app = self.app  # type: ignore[attr-defined]
        self._agent_tool._logger = self.lg
        self._agent_tool.configure()
        return True

    def run(self, **kwargs: Any) -> int:
        if self._agent_tool is None:
            return 1
        return self._agent_tool.run(**kwargs)

    def _get_factory_module(self, agent_name: str) -> str | None:
        """Get factory module path from agent config."""
        if not hasattr(self.app, "config") or not hasattr(self.app.config, "agents"):
            print("Error: No agents configured")
            return None

        agents = self.app.config.agents
        if agent_name not in agents:
            print(f"Error: Agent '{agent_name}' not found")
            print(f"Available agents: {list(agents.keys())}")
            return None

        agent_config = agents[agent_name]
        module = agent_config.get("module")
        if not module:
            print(f"Error: Agent '{agent_name}' has no module configured")
            print("Only programmatic agents (type: programmatic) support CLI tools.")
            return None

        return str(module)

    def _load_factory_class(self, factory_module: str) -> type[Factory] | None:
        """Load Factory class from module."""
        try:
            module = importlib.import_module(factory_module)
            factory_class = getattr(module, "Factory", None)
            if factory_class is None:
                print(f"Error: Module {factory_module} has no Factory class")
                return None
            return factory_class  # type: ignore[no-any-return]
        except ImportError as e:
            self.lg.error("failed to load factory module", extra={"exception": e})
            print(f"Error: Could not load module {factory_module}: {e}")
            return None
