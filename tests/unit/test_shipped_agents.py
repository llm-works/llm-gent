# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The agent configs shipped under ``etc/agents`` load and build the way ``llm-gent serve`` does."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from appinfra import DotDict

from llm_gent.agents.default import Factory
from llm_gent.cli.tools.serve import ServeTool
from llm_gent.core.platform import PlatformContext
from llm_gent.core.traits.builtin.tools import ToolsTrait
from llm_gent.runtime.server.config import AgentServerConfig


pytestmark = pytest.mark.unit

AGENTS_DIR = Path(__file__).parents[2] / "llm_gent" / "etc" / "agents"
SHIPPED = sorted(AGENTS_DIR.glob("*.yaml"))


def _build(path: Path, codebase: Path) -> Any:
    """Parse ``path`` as serve's config does and build the agent with the default factory."""
    name = path.stem
    raw = yaml.safe_load(path.read_text())
    agent_config = AgentServerConfig.from_dict({"agents": {name: raw}}).agents[name]
    config = ServeTool()._build_agent_config_dict(name, agent_config)
    platform = PlatformContext.from_config(lg=MagicMock(), llm_config=DotDict({}))
    return Factory(platform=platform).create(config, variables={"CODEBASE_PATH": str(codebase)})


def test_agents_are_shipped():
    assert [p.stem for p in SHIPPED] == ["codebase-qna", "jokester"]


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_agent_builds(path, tmp_path):
    assert _build(path, tmp_path) is not None


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_scheduled_task_is_the_task_description(path, tmp_path):
    """What ``run_once()`` runs is ``task.description``, with its variables substituted."""
    description = yaml.safe_load(path.read_text())["task"]["description"]

    agent = _build(path, tmp_path)

    assert agent._default_prompt == description.replace("{{CODEBASE_PATH}}", str(tmp_path))
    assert agent._default_prompt.strip()


class TestCodebaseQnaShell:
    """codebase-qna's shell runs its allowed programs in the codebase, and nothing else."""

    @pytest.fixture
    def shell(self, tmp_path):
        (tmp_path / "README.md").write_text("hello\n")
        agent = _build(AGENTS_DIR / "codebase-qna.yaml", tmp_path)
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
