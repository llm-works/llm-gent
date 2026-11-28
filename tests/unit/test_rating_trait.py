# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""RatingTrait on the platform and the RatingBinding each agent holds."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.core.memory.rating import ProviderType
from llm_gent.core.traits import ScopedTrait, TraitName
from llm_gent.core.traits.builtin.rating import RatingBinding, RatingTrait


pytestmark = pytest.mark.unit

RATING = {
    "providers": [
        {"type": "llm", "backend": {"model": "judge", "type": "openai"}},
        {"type": "llm", "backend": {"model": "off", "type": "openai"}, "enabled": False},
    ],
    "models": {
        "atomic": {
            "solution": {
                "prompt": "Rate this joke",
                "criteria": [{"name": "humor", "description": "Is it funny?"}],
            }
        }
    },
    "batch_size": 3,
}


@pytest.fixture
def parts():
    """The service and backend the trait builds, as mocks; no model call or SQL runs."""
    with (
        patch("llm_gent.core.traits.builtin.rating.AtomicFactsBackend") as backend,
        patch("llm_gent.core.traits.builtin.rating.RatingService") as service,
        patch("llm_gent.core.traits.builtin.rating.LLMCaller") as caller,
    ):
        yield {"backend": backend, "service": service, "caller": caller}


@pytest.fixture
def memory():
    """The platform's memory trait: its database and schema."""
    memory = MagicMock()
    memory.default_schema = "acme"
    return memory


def _agent(context_key: str = "jokes", rating: dict | None = None) -> MagicMock:
    agent = MagicMock()
    agent.identity.context_key = context_key
    agent.config = DotDict({"rating": rating} if rating is not None else {})
    return agent


def _trait(memory: MagicMock) -> RatingTrait:
    return RatingTrait(MagicMock(), MagicMock(), memory)


class TestTrait:
    def test_is_a_scoped_trait_named_rating(self, parts, memory):
        trait = _trait(memory)

        assert isinstance(trait, ScopedTrait)
        assert trait.trait_name == TraitName.RATING

    def test_builds_one_service_and_backend_on_the_memory_schema(self, parts, memory):
        router = MagicMock()

        trait = RatingTrait(MagicMock(), router, memory)

        assert parts["caller"].call_args.args[1] is router
        parts["backend"].assert_called_once()
        assert parts["backend"].call_args.args[1:] == (memory.database, "acme")
        assert trait.backend is parts["backend"].return_value
        assert trait.service is parts["service"].return_value

    def test_the_trait_has_no_rating_operations(self, parts, memory):
        trait = _trait(memory)

        assert not hasattr(trait, "rate_unrated")
        assert not hasattr(trait, "rate_fact_with_all_providers")


class TestBind:
    def test_binds_the_agents_context_key_and_config(self, parts, memory):
        trait = _trait(memory)

        binding = trait.bind(_agent("jokes", RATING))

        assert isinstance(binding, RatingBinding)
        assert binding.trait is trait
        assert binding.context_key == "jokes"
        assert binding.config["batch_size"] == 3

    def test_each_agent_rates_by_its_own_config(self, parts, memory):
        trait = _trait(memory)
        strict = {**RATING, "models": {"atomic": {"prediction": {"criteria": ["accuracy"]}}}}

        jokes = trait.bind(_agent("jokes", RATING))
        forecasts = trait.bind(_agent("forecasts", strict))

        assert set(jokes._criteria) == {"solution"}
        assert set(forecasts._criteria) == {"prediction"}

    def test_the_trait_keeps_nothing_per_agent(self, parts, memory):
        trait = _trait(memory)
        before = dict(vars(trait))

        for i in range(5):
            trait.bind(_agent(f"agent-{i}", RATING))

        assert vars(trait) == before
        parts["backend"].assert_called_once()

    def test_no_rating_section_binds_with_defaults(self, parts, memory):
        binding = _trait(memory).bind(_agent())

        assert binding.get_fact_rating(1) is parts["backend"].return_value.get_fact_rating(1)
        with pytest.raises(RuntimeError, match="No enabled rating providers"):
            binding.rate_unrated()


class TestOperations:
    def test_counts_only_the_agents_facts(self, parts, memory):
        binding = _trait(memory).bind(_agent("jokes", RATING))
        backend = parts["backend"].return_value
        backend.get_unrated_count.return_value = 4

        assert binding.get_unrated_count("solution", "pun") == 4
        backend.get_unrated_count.assert_called_once_with("jokes", "solution", "pun")

    def test_rate_unrated_rates_and_saves_each_of_the_agents_facts(self, parts, memory):
        binding = _trait(memory).bind(_agent("jokes", RATING))
        backend = parts["backend"].return_value
        service = parts["service"].return_value
        backend.unrated_facts.return_value = [
            {"id": 1, "content": "a", "type": "solution"},
            {"id": 2, "content": "b", "type": "solution"},
        ]

        results = binding.rate_unrated(limit=2)

        backend.unrated_facts.assert_called_once_with("jokes", None, None, 2)
        assert results == [service.rate_content.return_value] * 2
        assert service.rate_content.call_args.args[0].model == "judge"
        assert backend.save_rating.call_count == 2

    def test_rate_with_all_providers_uses_the_enabled_llm_ones(self, parts, memory):
        binding = _trait(memory).bind(_agent("jokes", RATING))
        service = parts["service"].return_value

        results = binding.rate_fact_with_all_providers(7, "a joke")

        assert len(results) == 1
        request = service.rate_content.call_args.args[0]
        assert (request.fact, request.model, request.provider) == (7, "judge", "llm_judge_openai")
        assert [p.provider_type for p in binding._providers] == [ProviderType.LLM] * 2

    def test_a_failed_rating_is_logged_and_skipped(self, parts, memory):
        lg = MagicMock()
        agent = _agent("jokes", RATING)
        agent.lg = lg
        binding = _trait(memory).bind(agent)
        parts["service"].return_value.rate_content.side_effect = RuntimeError("model down")

        assert binding.rate_fact_with_all_providers(7, "a joke") == []
        assert lg.warning.call_args.kwargs["extra"]["exception"].args == ("model down",)

    def test_rate_items_rates_them_in_one_batch(self, parts, memory):
        binding = _trait(memory).bind(_agent("jokes", RATING))
        service = parts["service"].return_value
        service.rate_batch.return_value = [MagicMock(), MagicMock()]

        assert binding.rate_items([(1, "a"), (2, "b")]) == 2
        assert [item.fact for item in service.rate_batch.call_args.args[0].items] == [1, 2]

    def test_rate_batch_splits_the_agents_facts_by_batch_size(self, parts, memory):
        binding = _trait(memory).bind(_agent("jokes", RATING))
        backend = parts["backend"].return_value
        service = parts["service"].return_value
        backend.unrated_facts.return_value = [{"id": i, "content": str(i)} for i in range(5)]
        service.rate_batch.side_effect = lambda request: [MagicMock()] * len(request.items)

        assert binding.rate_batch() == (5, 0)
        assert [len(call.args[0].items) for call in service.rate_batch.call_args_list] == [3, 2]
        assert backend.unrated_facts.call_args.args[0] == "jokes"
