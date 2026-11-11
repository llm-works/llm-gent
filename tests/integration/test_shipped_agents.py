# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A shipped agent's tools run against a real codebase directory."""

import pytest

from llm_gent.core.traits.builtin.tools import ToolsTrait

from ..unit.test_shipped_agents import AGENTS_DIR, _build


pytestmark = pytest.mark.integration


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
