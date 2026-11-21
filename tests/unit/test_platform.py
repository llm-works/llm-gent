# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the platform config and factory."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.core.platform import Config, Context, Factory


pytestmark = pytest.mark.unit


PACKAGED = {
    "default": "local",
    "backends": {
        "local": {"type": "openai_compatible", "base_url": "http://localhost:18000/v1"},
        "anthropic": {"type": "anthropic", "model": "claude"},
    },
}


@pytest.fixture
def packaged():
    """The packaged router config, without reading the file."""
    with patch("llm_gent.core.platform.config.packaged_llm_config", return_value=PACKAGED):
        yield


@pytest.fixture
def clients():
    """llm-infer's client factory as the platform factory uses it."""
    with patch("llm_gent.core.platform.factory.LLMClientFactory") as factory:
        yield factory


class TestConfig:
    def test_defaults(self):
        config = Config()

        assert config.llm == {}
        assert config.memory is None

    def test_empty_llm_is_the_packaged_config(self, packaged):
        assert Config().router_config() == PACKAGED

    def test_llm_merges_over_the_packaged_config(self, packaged):
        config = Config(llm={"backends": {"local": {"model": "mine"}}})

        router_config = config.router_config()

        assert router_config["backends"]["local"]["model"] == "mine"
        assert router_config["backends"]["local"]["base_url"] == "http://localhost:18000/v1"
        assert "anthropic" in router_config["backends"]


class TestFactory:
    def test_builds_and_owns_the_router(self, packaged, clients):
        lg = MagicMock()
        router = clients.return_value.from_config.return_value

        platform = Factory(lg).from_config({"llm": {"default": "anthropic"}})

        assert isinstance(platform, Context)
        assert platform.llm is router
        assert platform.memory is None
        built_from = clients.return_value.from_config.call_args[0][0]
        assert built_from["default"] == "anthropic"
        assert "local" in built_from["backends"]
        platform.close()
        router.close.assert_called_once()

    def test_accepts_a_config_instance(self, packaged, clients):
        config = Config(llm={"default": "local"})

        platform = Factory(MagicMock()).from_config(config)

        assert platform.config is config

    def test_traits_get_the_shared_router(self, packaged, clients):
        platform = Factory(MagicMock()).from_config({})

        assert platform.trait_factory._router is platform.llm

    def test_with_llm_uses_the_client_and_leaves_it_open(self, packaged, clients):
        client = MagicMock()
        base = Factory(MagicMock())

        platform = base.with_llm(client).from_config({})
        platform.close()

        assert platform.llm is client
        clients.return_value.from_config.assert_not_called()
        client.close.assert_not_called()
        assert base._llm is None


@pytest.fixture
def database():
    """The kelt Database the platform builds for memory; no connection is made."""
    with (
        patch("appinfra.db.pg.PG") as pg,
        patch("llm_gent.core.platform.factory.Database") as database,
    ):
        yield {"pg": pg, "database": database}


class TestFactoryMemory:
    def test_no_memory_without_a_database(self, packaged, clients, database):
        platform = Factory(MagicMock()).from_config(
            {"memory": {"adapters": {"lora": {"base_path": "/a"}}}}
        )

        assert platform.memory is None
        database["database"].assert_not_called()

    def test_builds_shared_memory_clients(self, packaged, clients, database):
        lg = MagicMock()
        memory_config = {
            "db": {"url": "postgresql://localhost/test"},
            "llm": {"default": "local"},
            "embedder_url": "http://localhost:9000",
        }

        platform = Factory(lg).from_config({"memory": memory_config})

        memory = platform.memory
        assert memory.database is database["database"].return_value
        database["pg"].assert_called_once_with(lg, memory_config["db"])
        clients.return_value.embeddings.assert_called_once_with(
            base_url="http://localhost:9000", model="default", timeout=30.0
        )
        assert memory.embedder is clients.return_value.embeddings.return_value
        assert platform.trait_factory._memory is memory

    def test_close_disposes_database_and_closes_clients(self, packaged, clients, database):
        router, chat_client = MagicMock(), MagicMock()
        clients.return_value.from_config.side_effect = [router, chat_client]
        platform = Factory(MagicMock()).from_config(
            {"memory": {"db": {"url": "x"}, "embedder_url": "http://localhost:9000"}}
        )
        memory = platform.memory

        platform.close()

        assert memory.chat_client is chat_client
        memory.database.dispose.assert_called_once()
        chat_client.close.assert_called_once()
        memory.embedder.close.assert_called_once()
        router.close.assert_called_once()

    def test_failed_build_closes_what_was_built(self, packaged, clients, database):
        router = clients.return_value.from_config.return_value
        clients.return_value.embeddings.side_effect = RuntimeError("embedder down")

        with pytest.raises(RuntimeError, match="embedder down"):
            Factory(MagicMock()).from_config(
                {"memory": {"db": {"url": "x"}, "embedder_url": "http://localhost:9000"}}
            )

        router.close.assert_called()
        database["database"].return_value.dispose.assert_called_once()
