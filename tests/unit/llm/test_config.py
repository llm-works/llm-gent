# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the router-config helpers in ``llm_gent.core.llm.config``."""

import pytest

from llm_gent.core.llm.config import merge_llm_config, resolve_llm_defaults


pytestmark = pytest.mark.unit


class TestMergeLLMConfig:
    """Tests for merge_llm_config()."""

    def test_override_existing_backend(self):
        base = {
            "default": "anthropic",
            "backends": {
                "anthropic": {"adapter": "openai", "base_url": "https://api.anthropic.com"}
            },
        }
        override = {"backends": {"anthropic": {"base_url": "https://custom.endpoint.com"}}}

        result = merge_llm_config(base, override)

        assert result["backends"]["anthropic"]["adapter"] == "openai"
        assert result["backends"]["anthropic"]["base_url"] == "https://custom.endpoint.com"

    def test_add_new_backend(self):
        base = {"default": "anthropic", "backends": {"anthropic": {"adapter": "openai"}}}
        override = {"backends": {"custom": {"adapter": "vllm", "base_url": "http://custom:8000"}}}

        result = merge_llm_config(base, override)

        assert result["backends"]["custom"]["adapter"] == "vllm"
        assert result["backends"]["anthropic"]["adapter"] == "openai"

    def test_top_level_keys_merged(self):
        base = {"default": "anthropic", "backends": {"anthropic": {"adapter": "openai"}}}
        override = {"default": "local", "temperature": 0.7}

        result = merge_llm_config(base, override)

        assert result["default"] == "local"
        assert result["temperature"] == 0.7

    def test_override_backends_only_in_override(self):
        result = merge_llm_config({"default": "anthropic"}, {"backends": {"new": {"adapter": "t"}}})

        assert result["backends"]["new"]["adapter"] == "t"

    def test_none_backend_skipped(self):
        base = {"default": "anthropic", "backends": {"anthropic": {"adapter": "openai"}}}

        result = merge_llm_config(base, {"backends": {"anthropic": None}})

        assert result["backends"]["anthropic"]["adapter"] == "openai"

    def test_none_backend_absent_from_base_skipped(self):
        base = {"default": "anthropic", "backends": {"anthropic": {"adapter": "openai"}}}

        result = merge_llm_config(base, {"backends": {"new": None}})

        assert "new" not in result["backends"]

    def test_empty_override_keeps_base(self):
        base = {"default": "anthropic", "backends": {"anthropic": {"adapter": "openai"}}}

        assert merge_llm_config(base, {}) == base

    def test_does_not_mutate_inputs(self):
        base = {"default": "anthropic", "backends": {"anthropic": {"adapter": "openai"}}}
        override = {"backends": {"anthropic": {"adapter": "custom"}, "new": {"adapter": "n"}}}

        result = merge_llm_config(base, override)
        result["backends"]["new"]["adapter"] = "changed"

        assert base["backends"]["anthropic"]["adapter"] == "openai"
        assert override["backends"]["new"]["adapter"] == "n"


class TestResolveLLMDefaults:
    """Tests for resolve_llm_defaults()."""

    def test_single_backend_config(self):
        result = resolve_llm_defaults({"model": "qwen2.5", "temperature": 0.3, "adapter": "lora1"})

        assert result["model"] == "qwen2.5"
        assert result["temperature"] == 0.3
        assert result["adapter"] == "lora1"

    def test_single_backend_defaults(self):
        result = resolve_llm_defaults({})

        assert result == {
            "model": "default",
            "temperature": 0.7,
            "max_tokens": None,
            "adapter": None,
        }

    def test_multi_backend_with_default(self):
        config = {
            "default": "cloud",
            "backends": {
                "local": {"model": "qwen2.5", "temperature": 0.3},
                "cloud": {"model": "claude-3", "temperature": 0.5},
            },
        }

        result = resolve_llm_defaults(config)

        assert result["model"] == "claude-3"
        assert result["temperature"] == 0.5

    def test_multi_backend_no_default_picks_first(self):
        config = {"backends": {"local": {"model": "qwen2.5"}, "cloud": {"model": "claude-3"}}}

        assert resolve_llm_defaults(config)["model"] == "qwen2.5"

    def test_unknown_default_backend_yields_defaults(self):
        config = {"default": "missing", "backends": {"local": {"model": "qwen2.5"}}}

        assert resolve_llm_defaults(config)["model"] == "default"
