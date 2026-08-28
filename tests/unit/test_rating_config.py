# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for rating configuration parsing."""

from unittest.mock import MagicMock

import pytest
from appinfra import DotDict

from llm_gent.core.memory.rating.config import ConfigParser
from llm_gent.core.memory.rating.models import ProviderType


pytestmark = pytest.mark.unit


@pytest.fixture
def parser():
    return ConfigParser(MagicMock())


# ---------------------------------------------------------------------------
# parse_providers
# ---------------------------------------------------------------------------


class TestParseProviders:
    def test_empty(self, parser):
        assert parser.parse_providers([]) == []
        assert parser.parse_providers({}) == []

    def test_list_format(self, parser):
        config = [{"type": "llm", "backend": {"model": "claude-3"}}]
        result = parser.parse_providers(config)
        assert len(result) == 1
        assert result[0].provider_type == ProviderType.LLM
        assert result[0].model == "claude-3"

    def test_dict_format(self, parser):
        config = {"anthropic": {"type": "llm", "backend": {"model": "claude-3"}}}
        result = parser.parse_providers(config)
        assert len(result) == 1

    def test_missing_backend_skipped(self, parser):
        config = [{"type": "llm"}]
        result = parser.parse_providers(config)
        assert result == []

    def test_missing_backend_with_name(self, parser):
        config = {"provider1": {"type": "llm"}}
        result = parser.parse_providers(config)
        assert result == []

    def test_invalid_type_defaults_to_llm(self, parser):
        config = [{"type": "invalid_type", "backend": {"model": "x"}}]
        result = parser.parse_providers(config)
        assert len(result) == 1
        assert result[0].provider_type == ProviderType.LLM

    def test_defaults(self, parser):
        config = [{"backend": {"model": "test"}}]
        result = parser.parse_providers(config)
        assert len(result) == 1
        assert result[0].provider_type == ProviderType.LLM
        assert result[0].enabled is True

    def test_enabled_false(self, parser):
        config = [{"backend": {"model": "test"}, "enabled": False}]
        result = parser.parse_providers(config)
        assert result[0].enabled is False

    def test_dotdict_input(self, parser):
        config = [DotDict({"type": "llm", "backend": DotDict({"model": "test"})})]
        result = parser.parse_providers(config)
        assert len(result) == 1

    def test_model_default(self, parser):
        config = [{"backend": {"type": "openai"}}]
        result = parser.parse_providers(config)
        assert result[0].model == "auto"

    def test_manual_provider_type(self, parser):
        config = [{"type": "manual", "backend": {"model": "human"}}]
        result = parser.parse_providers(config)
        assert result[0].provider_type == ProviderType.MANUAL


# ---------------------------------------------------------------------------
# parse_criteria
# ---------------------------------------------------------------------------


class TestParseCriteria:
    def test_empty(self, parser):
        assert parser.parse_criteria({}) == {}

    def test_full_criteria(self, parser):
        config = {
            "atomic": {
                "solution": {
                    "prompt": "Rate this joke",
                    "criteria": [{"name": "humor", "description": "Is it funny?", "weight": 2.0}],
                }
            }
        }
        result = parser.parse_criteria(config)
        assert "solution" in result
        assert result["solution"].prompt == "Rate this joke"
        assert len(result["solution"].criteria) == 1
        assert result["solution"].criteria[0].name == "humor"
        assert result["solution"].criteria[0].weight == 2.0

    def test_string_shorthand(self, parser):
        config = {"atomic": {"solution": {"prompt": "Rate", "criteria": ["humor", "quality"]}}}
        result = parser.parse_criteria(config)
        assert len(result["solution"].criteria) == 2
        assert result["solution"].criteria[0].name == "humor"

    def test_missing_name_skipped(self, parser):
        config = {
            "atomic": {
                "solution": {"prompt": "Rate", "criteria": [{"description": "no name field"}]}
            }
        }
        result = parser.parse_criteria(config)
        # No valid criteria -> type skipped
        assert "solution" not in result

    def test_missing_prompt_default(self, parser):
        config = {"atomic": {"solution": {"criteria": [{"name": "quality"}]}}}
        result = parser.parse_criteria(config)
        assert "solution" in result
        assert "solution" in result["solution"].prompt  # default includes fact_type

    def test_default_weight_and_description(self, parser):
        config = {"atomic": {"solution": {"prompt": "Rate", "criteria": [{"name": "humor"}]}}}
        result = parser.parse_criteria(config)
        c = result["solution"].criteria[0]
        assert c.weight == 1.0
        assert "humor" in c.description  # default description includes name

    def test_empty_criteria_list(self, parser):
        config = {"atomic": {"solution": {"prompt": "Rate", "criteria": []}}}
        result = parser.parse_criteria(config)
        assert "solution" not in result  # skipped due to empty criteria

    def test_multiple_fact_types(self, parser):
        config = {
            "atomic": {
                "solution": {"prompt": "Rate sol", "criteria": [{"name": "quality"}]},
                "prediction": {"prompt": "Rate pred", "criteria": [{"name": "accuracy"}]},
            }
        }
        result = parser.parse_criteria(config)
        assert "solution" in result
        assert "prediction" in result

    def test_no_atomic_key(self, parser):
        config = {"other": {"solution": {"prompt": "x", "criteria": [{"name": "y"}]}}}
        result = parser.parse_criteria(config)
        assert result == {}


# ---------------------------------------------------------------------------
# parse_batch
# ---------------------------------------------------------------------------


class TestParseBatch:
    def test_none(self, parser):
        result = parser.parse_batch(None)
        assert result.enabled is False
        assert result.size == 5

    def test_explicit_size(self, parser):
        result = parser.parse_batch(10)
        assert result.enabled is True
        assert result.size == 10

    def test_size_one(self, parser):
        result = parser.parse_batch(1)
        assert result.enabled is False  # batch_size > 1 for enabled
        assert result.size == 1

    def test_size_zero(self, parser):
        result = parser.parse_batch(0)
        assert result.enabled is False
        assert result.size == 5  # falls back to default
