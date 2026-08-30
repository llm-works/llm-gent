# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for server configuration models.

Note: The AgentGateway class has been replaced by AgentRegistry.
See test_registry.py for registry tests.
This file tests configuration models that are still in use.
"""

import pytest

from llm_gent.core.traits.builtin.directive import Directive
from llm_gent.runtime.server.config import (
    AgentConfigYAML,
    AgentServerConfig,
    LearnBackendConfig,
    ScheduleConfigYAML,
    TaskConfigYAML,
)


pytestmark = pytest.mark.unit


class TestAgentServerConfig:
    """Tests for AgentServerConfig."""

    def test_agent_config_yaml(self):
        """Parse agent config from dict."""
        config = AgentConfigYAML(
            directive=Directive(prompt="Test prompt"),
            task=TaskConfigYAML(description="Do things"),
            tools={"shell": {"allowed_commands": ["ls"]}},
            schedule=ScheduleConfigYAML(interval=300),
        )
        assert config.type_ == "prompt"
        assert config.directive.prompt == "Test prompt"
        assert config.schedule.interval == 300

    def test_agent_server_config_from_dict(self):
        """Create AgentServerConfig from raw dict."""
        raw = {
            "server": {"port": 9000},
            "llm": {"base_url": "http://localhost:8000/v1"},
            "agents": {
                "test-agent": {
                    "class": "prompt",
                    "directive": {"prompt": "Be helpful"},
                    "task": {"description": "Help users"},
                }
            },
        }
        config = AgentServerConfig.from_dict(raw)

        assert config.server.port == 9000
        assert config.llm["base_url"] == "http://localhost:8000/v1"
        assert "test-agent" in config.agents
        assert config.agents["test-agent"].directive.prompt == "Be helpful"


class TestScheduleConfigYAML:
    """Tests for ScheduleConfigYAML."""

    def test_interval_required(self):
        """ScheduleConfigYAML requires interval."""
        config = ScheduleConfigYAML(interval=60)
        assert config.interval == 60


class TestLearnBackendConfig:
    """Tests for LearnBackendConfig."""

    def test_defaults(self):
        """LearnBackendConfig has defaults for optional fields."""
        db_config = {"url": "postgresql://localhost/learn"}
        config = LearnBackendConfig(profile_id="1", db=db_config)
        assert config.profile_id == "1"
        assert config.db == db_config
        assert config.embedder_model == "default"
        assert config.embedder_url is None

    def test_custom_values(self):
        """LearnBackendConfig accepts custom values."""
        db_config = {"url": "postgresql://localhost/custom", "extensions": ["vector"]}
        config = LearnBackendConfig(
            profile_id="42",
            db=db_config,
            embedder_url="http://embedder:8000",
            embedder_model="custom-model",
        )
        assert config.profile_id == "42"
        assert config.db == db_config
        assert config.embedder_url == "http://embedder:8000"
        assert config.embedder_model == "custom-model"


class TestTaskConfigYAML:
    """Tests for TaskConfigYAML."""

    def test_description_required(self):
        """TaskConfigYAML requires description."""
        config = TaskConfigYAML(description="Test task")
        assert config.description == "Test task"
        assert config.output_schema is None

    def test_with_output_schema(self):
        """TaskConfigYAML accepts output_schema."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        config = TaskConfigYAML(description="Test task", output_schema=schema)
        assert config.output_schema == schema


class TestAgentConfigYAML:
    """Tests for AgentConfigYAML."""

    def test_default_class(self):
        """AgentConfigYAML defaults to 'prompt' class."""
        config = AgentConfigYAML(
            identity={"name": "test"},
            task=TaskConfigYAML(description="Test"),
        )
        assert config.type_ == "prompt"

    def test_programmatic_class(self):
        """AgentConfigYAML accepts 'programmatic' type."""
        raw = {
            "type": "programmatic",
            "identity": {"name": "test"},
            "task": {"description": "Test"},
        }
        config = AgentConfigYAML(**raw)
        assert config.type_ == "programmatic"

    def test_tools_default_empty(self):
        """AgentConfigYAML defaults tools to empty dict."""
        config = AgentConfigYAML(
            identity={"name": "test"},
            task=TaskConfigYAML(description="Test"),
        )
        assert config.tools == {}

    def test_schedule_optional(self):
        """AgentConfigYAML schedule is optional."""
        config = AgentConfigYAML(
            identity={"name": "test"},
            task=TaskConfigYAML(description="Test"),
        )
        assert config.schedule is None
