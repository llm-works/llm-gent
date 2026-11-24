# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shipped agents for real: their YAML read from disk, their tools run against a codebase.

What a shipped agent declares in its YAML reaches the agent ``llm-gent serve``
builds, and codebase-qna's shell runs in a real directory.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from appinfra import DotDict

from llm_gent.agents.default import Factory
from llm_gent.core.agent import SessionConfig
from llm_gent.core.traits.builtin.memory import MemoryTrait
from llm_gent.core.traits.builtin.saia import SAIATrait
from llm_gent.core.traits.builtin.tools import ToolsTrait
from llm_gent.runtime.server.config import AgentServerConfig
from tests.helpers import make_platform

from ..unit.test_shipped_agents import _build as _build_on_codebase


pytestmark = pytest.mark.integration

AGENTS_DIR = Path(__file__).parents[2] / "llm_gent" / "etc" / "agents"
SHIPPED = sorted(AGENTS_DIR.glob("*.yaml"))


def _raw(path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return raw


def _build(path: Path) -> Any:
    """Build the agent at ``path`` as serve does, on a platform without memory."""
    name = path.stem
    config = AgentServerConfig.from_dict({"agents": {name: _raw(path)}}).agents[name]
    factory = Factory(platform=make_platform(MagicMock()))
    return factory.create(config.factory_config(name), variables={"CODEBASE_PATH": "/srv/code"})


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_the_directive_and_method_are_the_system_prompt(path: Path) -> None:
    raw = _raw(path)

    prompt = _build(path).require_trait(SAIATrait).config.system_prompt

    assert prompt is not None
    assert raw["directive"].strip() in prompt
    assert raw["method"].strip() in prompt


def test_memory_tools_attach_memory_on_a_platform_with_memory() -> None:
    """Jokester's remember / recall need a MemoryTrait; a platform with memory gives it one."""
    path = AGENTS_DIR / "jokester.yaml"
    config = AgentServerConfig.from_dict({"agents": {"jokester": _raw(path)}}).agents["jokester"]
    platform = make_platform(MagicMock(), memory_config={"db": {}}, memory=_memory(_database()))

    agent = Factory(platform=platform).create(config.factory_config("jokester"))

    assert agent.require_trait(MemoryTrait).identity == agent.identity
    names = {tool.name for tool in agent.require_trait(ToolsTrait).registry.list_tools()}
    assert {"remember", "recall"} <= names


def test_agents_on_one_platform_prepare_the_memory_schema_once() -> None:
    """Each agent's memory is a view of the platform's: the schema is prepared once."""
    database = _database()
    platform = make_platform(MagicMock(), memory_config={"db": {}}, memory=_memory(database))
    factory = Factory(platform=platform)
    raw = _raw(AGENTS_DIR / "jokester.yaml")

    agents = []
    for name in ("jokester", "jokester-2"):
        agent_raw = {**raw, "identity": {"name": name}}
        config = AgentServerConfig.from_dict({"agents": {name: agent_raw}}).agents[name]
        agents.append(factory.create(config.factory_config(name)))

    database.ensure_schema.assert_called_once()
    keys = {agent.require_trait(MemoryTrait).kelt.context_key for agent in agents}
    assert keys == {"jokester", "jokester-2"}


def _database() -> MagicMock:
    database = MagicMock()
    database.schema = "public"  # kelt checks the client's schema against the database's
    return database


def _memory(database: MagicMock) -> MemoryTrait:
    """The platform's memory trait: a real kelt client over a stub database."""
    return MemoryTrait(MagicMock(), DotDict({}), database=database, chat_client=MagicMock())


def test_memory_tools_are_skipped_on_a_platform_without_memory() -> None:
    assert _build(AGENTS_DIR / "jokester.yaml").get_trait(MemoryTrait) is None


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_the_chat_section_configures_the_sessions(path: Path) -> None:
    declared = _raw(path)["chat"]

    config = _build(path)._sessions._config

    assert config == SessionConfig(**declared)


class TestCodebaseQnaShell:
    """codebase-qna's shell runs its allowed programs in the codebase, and nothing else."""

    @pytest.fixture
    def shell(self, tmp_path):
        (tmp_path / "README.md").write_text("hello\n")
        agent = _build_on_codebase(AGENTS_DIR / "codebase-qna.yaml", tmp_path)
        tool = agent.get_trait(ToolsTrait).registry.get("shell")
        assert tool is not None
        return tool

    def test_runs_in_the_codebase(self, shell):
        result = shell.execute(command="ls")

        assert result.success is True
        assert "README.md" in result.output

    @pytest.mark.parametrize("command", ["find . -name x", "git log", "tree", "sh -c id"])
    def test_other_programs_are_not_allowed(self, shell, command):
        result = shell.execute(command=command)

        assert result.success is False
        assert "not in allowed list" in result.error
