# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""MemoryTrait holds the platform's memory clients; each agent holds its own MemoryBinding."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict
from llm_kelt import SchemaMode

from llm_gent.core.agent import Identity
from llm_gent.core.traits.builtin.memory import MemoryBinding, MemoryTrait


pytestmark = pytest.mark.unit


def _agent(name: str = "a", context_key: str | None = None) -> MagicMock:
    """An agent as bind reads it: its identity."""
    return MagicMock(identity=Identity(name, context_key=context_key or name))


@pytest.fixture
def kelt_class():
    """The kelt Client class MemoryTrait builds; nothing touches a database."""
    with patch("llm_gent.core.traits.builtin.memory.KeltClient") as kelt:
        yield kelt


@pytest.fixture
def clients():
    return {"database": MagicMock(), "chat_client": MagicMock(), "embedder": MagicMock()}


def _memory(clients, **config) -> MemoryTrait:
    return MemoryTrait(MagicMock(), DotDict(config), **clients)


def _bound(memory: MemoryTrait, name: str = "a") -> MemoryBinding:
    return memory.bind(_agent(name))


class TestTrait:
    def test_builds_one_kelt_client_on_the_platform_schema(self, kelt_class, clients):
        _memory(clients, schema={"name": "acme"}, training={"x": 1})

        kelt_class.assert_called_once()
        kwargs = kelt_class.call_args.kwargs
        assert kwargs["database"] is clients["database"]
        assert kwargs["embedder"] is clients["embedder"]
        assert kwargs["llm_client"] is clients["chat_client"]
        assert kwargs["schema_mode"] is SchemaMode.ENSURE
        assert kwargs["context"].schema_name == "acme"
        assert kwargs["training_config"] == {"x": 1}

    def test_is_named_memory(self, kelt_class, clients):
        assert _memory(clients).trait_name == "memory"

    def test_schema_defaults_to_public(self, kelt_class, clients):
        clients["database"].schema = None
        memory = _memory(clients)

        assert memory.default_schema == "public"

    def test_without_a_platform_schema_the_database_schema_is_used(self, kelt_class, clients):
        clients["database"].schema = "tenant"

        memory = _memory(clients)

        assert kelt_class.call_args.kwargs["context"].schema_name == "tenant"
        assert memory.default_schema == "tenant"

    def test_the_platform_schema_wins_over_the_database_schema(self, kelt_class, clients):
        clients["database"].schema = "tenant"

        memory = _memory(clients, schema={"name": "acme"})

        assert kelt_class.call_args.kwargs["context"].schema_name == "acme"
        assert memory.default_schema == "acme"

    def test_the_trait_has_no_memory_operations(self, kelt_class, clients):
        memory = _memory(clients)

        assert not hasattr(memory, "remember")
        assert not hasattr(memory, "kelt")

    def test_lifecycle_closes_no_client(self, kelt_class, clients):
        memory = _memory(clients)

        memory.on_start()
        memory.on_stop()

        clients["chat_client"].close.assert_not_called()
        clients["embedder"].close.assert_not_called()


class TestBind:
    def test_binds_the_agents_context_key_on_the_platform_schema(self, kelt_class, clients):
        memory = _memory(clients, schema={"name": "acme"})

        binding = memory.bind(_agent("jokester", context_key="jokes"))

        root = kelt_class.return_value
        root.view.assert_called_once_with("jokes", schema_name="acme")
        assert binding.kelt is root.view.return_value
        assert binding.context_key == "jokes"
        assert binding.trait is memory
        assert binding.default_schema == "acme"

    def test_every_bind_is_a_new_binding(self, kelt_class, clients):
        memory = _memory(clients)

        assert memory.bind(_agent("a")) is not memory.bind(_agent("a"))

    def test_the_trait_keeps_nothing_per_agent(self, kelt_class, clients):
        memory = _memory(clients)
        before = dict(vars(memory))

        for i in range(5):
            _bound(memory, f"agent-{i}")

        assert vars(memory) == before

    def test_bindings_share_the_platforms_clients(self, kelt_class, clients):
        memory = _memory(clients)

        bindings = [_bound(memory, f"agent-{i}") for i in range(5)]

        kelt_class.assert_called_once()
        assert all(b.client is clients["chat_client"] for b in bindings)
        assert all(b.database is clients["database"] for b in bindings)
        assert all(b.embedder is clients["embedder"] for b in bindings)

    def test_client_for_another_schema_keeps_the_context_key(self, kelt_class, clients):
        binding = _memory(clients).bind(_agent("a", context_key="ka"))
        root = kelt_class.return_value
        root.view.reset_mock()

        client = binding.get_client_for_schema("other")

        root.view.assert_called_once_with("ka", schema_name="other")
        assert client is root.view.return_value


class TestOperations:
    def test_remember_stores_and_embeds(self, kelt_class, clients):
        binding = _bound(_memory(clients))
        kelt = binding.kelt
        # MagicMock refuses `.assertions` as an auto-attribute; assign it.
        kelt.atomic.assertions = MagicMock()
        kelt.atomic.assertions.add.return_value = 7

        fact_id = binding.remember("likes tea", category="taste")

        assert fact_id == 7
        kelt.atomic.assertions.add.assert_called_once_with(
            "likes tea", category="taste", source="user", confidence=1.0
        )
        clients["embedder"].embed.assert_called_once_with("likes tea")
        kelt.atomic.embeddings.set_embedding.assert_called_once()

    def test_recall_in_another_schema(self, kelt_class, clients):
        binding = _bound(_memory(clients))
        other = MagicMock()
        kelt_class.return_value.view.return_value = other

        binding.recall("tea", schema="other")

        other.atomic.embeddings.search_similar.assert_called_once()

    def test_build_prompt_uses_the_bindings_context_builder(self, kelt_class, clients):
        binding = _bound(_memory(clients))
        binding.kelt.context_builder.build_system_prompt.return_value = "prompt"

        assert binding.build_prompt("base") == "prompt"

    def test_complete_uses_the_platform_llm_defaults(self, kelt_class, clients):
        binding = _bound(_memory(clients, llm={"model": "m1", "temperature": 0.2}))
        clients["chat_client"].chat.return_value = MagicMock(content="ok", model="m1", usage=None)

        binding.complete("hi", include_facts=False)

        kwargs = clients["chat_client"].chat.call_args.kwargs
        assert kwargs["model"] == "m1"
        assert kwargs["temperature"] == 0.2
