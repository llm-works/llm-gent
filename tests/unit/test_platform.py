# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The platform: its config, its factory, one SAIA per role and its traits."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.core import Role
from llm_gent.core.platform import Config, Factory, Platform
from llm_gent.core.traits import Registry as TraitRegistry
from llm_gent.core.traits.builtin.memory import MemoryTrait
from tests.helpers import make_platform, make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


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

        assert isinstance(platform, Platform)
        assert platform.llm is router
        assert platform.trait_factory._memory is None
        built_from = clients.return_value.from_config.call_args[0][0]
        assert built_from["default"] == "anthropic"
        assert "local" in built_from["backends"]
        platform.close()
        router.close.assert_called_once()

    def test_accepts_a_config_instance(self, packaged, clients):
        config = Config(llm={"default": "local"})

        Factory(MagicMock()).from_config(config)

        assert clients.return_value.from_config.call_args[0][0]["default"] == "local"

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

    def test_the_registry_starts_empty(self, packaged, clients):
        platform = Factory(MagicMock()).from_config({})

        assert platform.traits.count() == 0

    def test_without_a_saia_factory_the_platform_builds_no_saia(self, packaged, clients):
        platform = Factory(MagicMock()).from_config({})

        with pytest.raises(RuntimeError, match=r"builds no SAIA \(role 'planner'\)"):
            platform.saia(Role(name="planner"))

    def test_with_saia_factory_gives_its_platforms_the_factory(self, packaged, clients):
        factory = _CountingFactory()
        base = Factory(MagicMock())

        platform = base.with_saia_factory(factory).from_config({})
        platform.saia(Role(name="planner"))

        assert platform.saia_factory is factory
        assert factory.built == [Role(name="planner")]
        with pytest.raises(RuntimeError, match="builds no SAIA"):
            base.from_config({}).saia(Role(name="planner"))


@pytest.fixture
def database():
    """The kelt Database and client the platform builds for memory; no connection is made."""
    with (
        patch("appinfra.db.pg.PG") as pg,
        patch("llm_gent.core.platform.factory.Database") as database,
        patch("llm_gent.core.traits.builtin.memory.KeltClient") as kelt,
    ):
        yield {"pg": pg, "database": database, "kelt": kelt}


class TestFactoryMemory:
    def test_no_memory_without_a_database(self, packaged, clients, database):
        platform = Factory(MagicMock()).from_config(
            {"memory": {"adapters": {"lora": {"base_path": "/a"}}}}
        )

        assert platform.trait_factory._memory is None
        database["database"].assert_not_called()

    def test_builds_shared_memory_clients(self, packaged, clients, database):
        lg = MagicMock()
        memory_config = {
            "db": {"url": "postgresql://localhost/test"},
            "llm": {"default": "local"},
            "embedder_url": "http://localhost:9000",
        }

        platform = Factory(lg).from_config({"memory": memory_config})

        memory = platform.trait_factory._memory
        assert memory.database is database["database"].return_value
        database["pg"].assert_called_once_with(lg, memory_config["db"])
        clients.return_value.embeddings.assert_called_once_with(
            base_url="http://localhost:9000", model="default", timeout=30.0
        )
        assert memory.embedder is clients.return_value.embeddings.return_value

    def test_the_memory_trait_is_built_once_into_the_registry(self, packaged, clients, database):
        platform = Factory(MagicMock()).from_config({"memory": {"db": {"url": "x"}}})

        memory = platform.traits.require(MemoryTrait)
        assert memory is platform.trait_factory._memory
        assert memory.identity is None
        database["kelt"].assert_called_once()

    def test_close_disposes_database_and_closes_clients(self, packaged, clients, database):
        router, chat_client = MagicMock(), MagicMock()
        clients.return_value.from_config.side_effect = [router, chat_client]
        platform = Factory(MagicMock()).from_config(
            {"memory": {"db": {"url": "x"}, "embedder_url": "http://localhost:9000"}}
        )
        memory = platform.trait_factory._memory

        platform.close()

        assert memory.client is chat_client
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


class _CountingFactory:
    """Builds a fresh object per call and records the roles it was asked for."""

    def __init__(self) -> None:
        self.built: list[Role] = []

    def build(self, role: Role) -> Any:
        self.built.append(role)
        return object()


class TestSaia:
    def test_a_role_is_built_once_and_kept(self):
        factory = _CountingFactory()
        platform = make_platform(LG, saia_factory=factory)
        role = Role(name="planner")
        first = platform.saia(role)
        assert platform.saia(role) is first
        assert platform.saia(Role(name="planner")) is first
        assert factory.built == [role]

    def test_each_role_gets_its_own_saia(self):
        platform = make_platform(LG, saia_factory=_CountingFactory())
        assert platform.saia(Role(name="planner")) is not platform.saia(Role(name="worker"))

    def test_roles_differing_only_in_params_build_apart(self):
        factory = _CountingFactory()
        platform = make_platform(LG, saia_factory=factory)
        role = Role(name="planner")
        varied = role.with_params(max_iterations=3)
        assert platform.saia(role) is not platform.saia(varied)
        assert factory.built == [role, varied]

    def test_platforms_do_not_share_saias(self):
        factory = _CountingFactory()
        role = Role(name="planner")
        first = make_platform(LG, saia_factory=factory).saia(role)
        assert make_platform(LG, saia_factory=factory).saia(role) is not first


class TestTraits:
    def test_traits_are_the_registry_given(self):
        registry = TraitRegistry(LG)
        assert make_platform(LG, traits=registry).traits is registry

    def test_memory_joins_a_copy_of_the_registry_given(self):
        registry = TraitRegistry(LG)
        with patch("llm_gent.core.traits.builtin.memory.KeltClient"):
            memory = MemoryTrait(LG, DotDict({}), database=MagicMock(), chat_client=MagicMock())

        first = make_platform(LG, traits=registry, memory=memory)
        second = make_platform(LG, traits=registry, memory=memory)

        assert not registry.has(MemoryTrait)
        assert first.traits.require(MemoryTrait) is memory
        assert second.traits.require(MemoryTrait) is memory
