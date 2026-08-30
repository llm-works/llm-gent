# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""E2E tests for agent execution with SAIA.

These tests require:
1. An LLM server at localhost:18100
2. SAIA properly configured with tools/executor

Tests are skipped if the environment is not properly configured.
"""

import os
from unittest.mock import MagicMock

import httpx
import pytest

from llm_gent.agents.default import Factory


pytestmark = [pytest.mark.e2e, pytest.mark.slow]

LLM_BASE_URL = "http://localhost:18100/v1"
LLM_MODEL = "default"


def saia_e2e_enabled() -> bool:
    """Check if SAIA E2E tests should run.

    E2E tests require a fully configured environment with LLM server.
    Set SAIA_E2E_TESTS=1 environment variable to enable.
    """
    if os.environ.get("SAIA_E2E_TESTS") != "1":
        return False

    try:
        response = httpx.get(f"{LLM_BASE_URL}/models", timeout=2.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


skip_no_saia = pytest.mark.skipif(
    not saia_e2e_enabled(),
    reason="SAIA E2E tests disabled (set SAIA_E2E_TESTS=1 to enable)",
)


@pytest.mark.expected_skip
@skip_no_saia
class TestAgentExecutionE2E:
    """E2E tests for Agent with real LLM via SAIA."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def llm_config(self):
        """LLM configuration for tests."""
        return {
            "default": "local",
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": LLM_BASE_URL,
                    "model": LLM_MODEL,
                }
            },
        }

    @pytest.fixture
    def agent(self, mock_logger, llm_config):
        """Create agent with SAIATrait connected to LLM server."""
        factory = Factory(lg=mock_logger, llm_config=llm_config)
        config = {
            "name": "test-agent",
            "identity": "You are a helpful assistant.",
            "task": {"description": "Respond briefly."},
        }
        agent = factory.create(config)
        agent.start()
        yield agent
        agent.stop()

    @pytest.fixture
    def agent_with_tools(self, mock_logger, llm_config):
        """Create agent with SAIATrait and tools."""
        factory = Factory(lg=mock_logger, llm_config=llm_config)
        config = {
            "name": "test-agent-tools",
            "identity": "You are a helpful assistant with shell access.",
            "tools": {"shell": {}, "read_file": {}},
            "task": {"description": "Help with tasks using available tools."},
        }
        agent = factory.create(config)
        agent.start()
        yield agent
        agent.stop()

    def test_agent_ask(self, agent):
        """Test agent ask() method."""
        response = agent.ask("Say hello in exactly 3 words.")

        assert len(response) > 0

    def test_agent_run_once(self, agent):
        """Test agent run_once() method."""
        result = agent.run_once()

        assert result.success is True
        assert len(result.content) > 0

    def test_agent_with_tools_ask(self, agent_with_tools):
        """Test agent with tools can answer questions."""
        response = agent_with_tools.ask("What tools do you have available?")

        assert len(response) > 0

    def test_agent_cycle_count(self, agent):
        """Test that cycle count increments."""
        assert agent.cycle_count == 0

        agent.run_once()
        assert agent.cycle_count == 1

        agent.run_once()
        assert agent.cycle_count == 2

    def test_agent_recent_results(self, agent):
        """Test recent results tracking."""
        assert len(agent.get_recent_results()) == 0

        agent.run_once()
        results = agent.get_recent_results()

        assert len(results) == 1
        assert results[0].success is True
