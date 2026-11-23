# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""MemoryTrait: one shared instance per platform, a cheap view per identity."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict
from llm_kelt import SchemaMode

from llm_gent.core.agent import Identity
from llm_gent.core.traits.builtin.memory import MemoryTrait


pytestmark = pytest.mark.unit


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


class TestSharedInstance:
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

    def test_memory_operations_need_a_view(self, kelt_class, clients):
        memory = _memory(clients)

        with pytest.raises(RuntimeError, match="call view"):
            memory.remember("a fact")
        with pytest.raises(RuntimeError, match="call view"):
            memory.get_client_for_schema("other")
        assert memory.identity is None

    def test_a_completion_needs_a_view_even_without_facts(self, kelt_class, clients):
        memory = _memory(clients)

        with pytest.raises(RuntimeError, match="call view"):
            memory.complete("hi", include_facts=False)
        clients["chat_client"].chat.assert_not_called()

    def test_lifecycle_closes_no_client(self, kelt_class, clients):
        memory = _memory(clients)

        memory.on_start()
        memory.on_stop()

        clients["chat_client"].close.assert_not_called()
        clients["embedder"].close.assert_not_called()


class TestView:
    def test_many_views_share_one_kelt_client(self, kelt_class, clients):
        memory = _memory(clients)

        views = [memory.view(Identity(f"agent-{i}")) for i in range(5)]

        kelt_class.assert_called_once()
        assert all(v.client is clients["chat_client"] for v in views)
        assert all(v.database is clients["database"] for v in views)

    def test_binds_the_identity_context_key_on_the_platform_schema(self, kelt_class, clients):
        memory = _memory(clients, schema={"name": "acme"})

        view = memory.view(Identity("jokester", context_key="jokes"))

        root = kelt_class.return_value
        root.view.assert_called_once_with("jokes", schema_name="acme")
        assert view.kelt is root.view.return_value
        assert view.identity == Identity("jokester", context_key="jokes")
        assert view.default_schema == "acme"

    def test_schema_override(self, kelt_class, clients):
        memory = _memory(clients)

        view = memory.view(Identity("a"), schema="agent_schema")

        kelt_class.return_value.view.assert_called_once_with("a", schema_name="agent_schema")
        assert view.default_schema == "agent_schema"

    def test_a_view_of_a_view_defaults_to_the_platform_schema(self, kelt_class, clients):
        memory = _memory(clients, schema={"name": "acme"})

        view = memory.view(Identity("a"), schema="tenant").view(Identity("b"))

        assert view.default_schema == "acme"
        kelt_class.return_value.view.assert_called_with("b", schema_name="acme")

    def test_leaves_the_shared_instance_unbound(self, kelt_class, clients):
        memory = _memory(clients)

        memory.view(Identity("a"))

        assert memory.identity is None
        with pytest.raises(RuntimeError, match="call view"):
            memory.kelt  # noqa: B018 - the property raises

    def test_views_of_different_identities_are_separate(self, kelt_class, clients):
        root = kelt_class.return_value
        root.view.side_effect = lambda key, **_: MagicMock(name=key)
        memory = _memory(clients)

        first, second = memory.view(Identity("a")), memory.view(Identity("b"))

        assert first.kelt is not second.kelt

    def test_client_for_another_schema_keeps_the_identity(self, kelt_class, clients):
        view = _memory(clients).view(Identity("a", context_key="ka"))
        root = kelt_class.return_value
        root.view.reset_mock()

        client = view.get_client_for_schema("other")

        root.view.assert_called_once_with("ka", schema_name="other")
        assert client is root.view.return_value


class TestOperations:
    def test_remember_stores_and_embeds(self, kelt_class, clients):
        view = _memory(clients).view(Identity("a"))
        kelt = view.kelt
        # MagicMock refuses `.assertions` as an auto-attribute; assign it.
        kelt.atomic.assertions = MagicMock()
        kelt.atomic.assertions.add.return_value = 7

        fact_id = view.remember("likes tea", category="taste")

        assert fact_id == 7
        kelt.atomic.assertions.add.assert_called_once_with(
            "likes tea", category="taste", source="user", confidence=1.0
        )
        clients["embedder"].embed.assert_called_once_with("likes tea")
        kelt.atomic.embeddings.set_embedding.assert_called_once()

    def test_recall_in_another_schema(self, kelt_class, clients):
        view = _memory(clients).view(Identity("a"))
        other = MagicMock()
        kelt_class.return_value.view.return_value = other

        view.recall("tea", schema="other")

        other.atomic.embeddings.search_similar.assert_called_once()

    def test_build_prompt_uses_the_view_context_builder(self, kelt_class, clients):
        view = _memory(clients).view(Identity("a"))
        view.kelt.context_builder.build_system_prompt.return_value = "prompt"

        assert view.build_prompt("base") == "prompt"

    def test_complete_uses_the_platform_llm_defaults(self, kelt_class, clients):
        view = _memory(clients, llm={"model": "m1", "temperature": 0.2}).view(Identity("a"))
        clients["chat_client"].chat.return_value = MagicMock(content="ok", model="m1", usage=None)

        view.complete("hi", include_facts=False)

        kwargs = clients["chat_client"].chat.call_args.kwargs
        assert kwargs["model"] == "m1"
        assert kwargs["temperature"] == 0.2
