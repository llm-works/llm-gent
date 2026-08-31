# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for MemoryTrait — construction-time DI and immutable-view fluents."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.core.traits.builtin.memory import MemoryConfig, MemoryTrait


pytestmark = pytest.mark.unit


@pytest.fixture
def agent():
    """Minimal Agent stub with logger."""
    stub = MagicMock()
    stub.lg = MagicMock()
    return stub


@pytest.fixture
def config():
    """Minimal MemoryConfig with resolved identity."""
    identity = MagicMock()
    identity.context_key = "test-ctx"
    return MemoryConfig(
        identity=identity,
        schema={"name": "public"},
        llm={},
        db={"url": "postgresql://localhost/test"},
        embedder_url=None,
        embedder_model="default",
        embedder_timeout=30.0,
        training=None,
    )


@pytest.fixture
def stubs():
    """Provide database, chat_client, embedder mocks with a KeltClient patch."""
    with (
        patch("llm_gent.core.traits.builtin.memory.KeltClient") as mock_kelt,
        patch("llm_gent.core.traits.builtin.memory.ContextBuilder"),
    ):
        kelt_instance = MagicMock()
        kelt_instance.atomic = MagicMock()
        kelt_instance.atomic.assertions = MagicMock()
        mock_kelt.return_value = kelt_instance
        yield {
            "database": MagicMock(),
            "chat_client": MagicMock(),
            "embedder": MagicMock(),
            "kelt_class": mock_kelt,
        }


class TestConstruction:
    def test_stores_injected_clients(self, agent, config, stubs):
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            embedder=stubs["embedder"],
        )
        assert trait._database is stubs["database"]
        assert trait._client is stubs["chat_client"]
        assert trait._embedder is stubs["embedder"]

    def test_defaults_to_unowned(self, agent, config, stubs):
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
        )
        assert trait._owns_chat_client is False
        assert trait._owns_embedder is False

    def test_embedder_optional(self, agent, config, stubs):
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
        )
        assert trait._embedder is None
        assert trait.has_embedder is False


class TestLifecycle:
    def test_on_start_resolves_defaults(self, agent, config, stubs):
        config["llm"] = DotDict({"model": "qwen2.5", "temperature": 0.3})
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
        )
        trait.on_start()
        assert trait._llm_defaults["model"] == "qwen2.5"
        assert trait._llm_defaults["temperature"] == 0.3

    def test_on_stop_closes_owned_chat_client(self, agent, config, stubs):
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            owns_chat_client=True,
        )
        trait.on_stop()
        stubs["chat_client"].close.assert_called_once()

    def test_on_stop_leaves_injected_chat_client(self, agent, config, stubs):
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            owns_chat_client=False,
        )
        trait.on_stop()
        stubs["chat_client"].close.assert_not_called()

    def test_on_stop_closes_owned_embedder(self, agent, config, stubs):
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            embedder=stubs["embedder"],
            owns_embedder=True,
        )
        trait.on_stop()
        stubs["embedder"].close.assert_called_once()

    def test_on_stop_leaves_injected_embedder(self, agent, config, stubs):
        trait = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            embedder=stubs["embedder"],
            owns_embedder=False,
        )
        trait.on_stop()
        stubs["embedder"].close.assert_not_called()


class TestWithChatClient:
    def test_returns_new_instance(self, agent, config, stubs):
        original = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            owns_chat_client=True,
        )
        replacement = MagicMock()
        detached = original.with_chat_client(replacement)
        assert detached is not original
        assert detached._client is replacement
        assert original._client is stubs["chat_client"]

    def test_new_instance_is_not_owner(self, agent, config, stubs):
        original = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            owns_chat_client=True,
        )
        detached = original.with_chat_client(MagicMock())
        assert detached._owns_chat_client is False
        assert detached._owns_embedder is False

    def test_shares_database_and_embedder(self, agent, config, stubs):
        original = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            embedder=stubs["embedder"],
        )
        detached = original.with_chat_client(MagicMock())
        assert detached._database is stubs["database"]
        assert detached._embedder is stubs["embedder"]

    def test_not_written_to_registry(self, agent, config, stubs):
        original = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
        )
        original.with_chat_client(MagicMock())
        agent.add_trait.assert_not_called()
        agent.replace_trait.assert_not_called()


class TestWithEmbedder:
    def test_swaps_embedder(self, agent, config, stubs):
        original = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            embedder=stubs["embedder"],
        )
        replacement = MagicMock()
        detached = original.with_embedder(replacement)
        assert detached._embedder is replacement
        assert original._embedder is stubs["embedder"]

    def test_can_swap_to_none(self, agent, config, stubs):
        original = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
            embedder=stubs["embedder"],
        )
        detached = original.with_embedder(None)
        assert detached._embedder is None
        assert detached.has_embedder is False


class TestWithDatabase:
    def test_swaps_database(self, agent, config, stubs):
        original = MemoryTrait(
            agent,
            config,
            database=stubs["database"],
            chat_client=stubs["chat_client"],
        )
        replacement = MagicMock()
        detached = original.with_database(replacement)
        assert detached._database is replacement
        assert original._database is stubs["database"]
