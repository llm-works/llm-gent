# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for TrainingTrait — adapter manifest / schema resolution."""

from unittest.mock import MagicMock

import pytest

from llm_gent.core.traits.builtin.training import (
    ManifestNotFoundError,
    TrainingConfig,
    TrainingTrait,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def agent():
    """Minimal Agent stub with logger."""
    stub = MagicMock()
    stub.lg = MagicMock()
    return stub


@pytest.fixture
def adapter_info():
    """Adapter info with a fake md5 fingerprint."""
    info = MagicMock()
    info.md5 = "abc123"
    return info


class TestDefaults:
    def test_default_schema_falls_back_to_public(self, agent):
        trait = TrainingTrait(agent)
        assert trait.default_schema == "public"

    def test_default_schema_from_config(self, agent):
        trait = TrainingTrait(agent, TrainingConfig(schema={"name": "custom"}))
        assert trait.default_schema == "custom"

    def test_config_defaults_to_empty(self, agent):
        trait = TrainingTrait(agent)
        assert isinstance(trait.config, TrainingConfig)


class TestLifecycle:
    def test_on_start_is_noop(self, agent):
        trait = TrainingTrait(agent)
        trait.on_start()  # must not raise or touch disk
        assert trait._train_factory is None

    def test_on_stop_clears_cache(self, agent):
        trait = TrainingTrait(agent)
        trait._train_factory = MagicMock()
        trait.on_stop()
        assert trait._train_factory is None


class TestResolveSchemaForAdapter:
    def test_enforce_returns_default_schema(self, agent, adapter_info):
        trait = TrainingTrait(agent, TrainingConfig(schema={"name": "forced", "enforce": True}))
        # Enforce short-circuits before the md5 branch — no factory needed.
        assert trait.resolve_schema_for_adapter(adapter_info) == "forced"

    def test_no_md5_returns_default_schema(self, agent):
        trait = TrainingTrait(agent, TrainingConfig(schema={"name": "fallback"}))
        info = MagicMock()
        info.md5 = None
        assert trait.resolve_schema_for_adapter(info) == "fallback"

    def test_no_train_factory_returns_default(self, agent, adapter_info):
        # No adapters config -> _get_train_factory returns None
        trait = TrainingTrait(agent, TrainingConfig(schema={"name": "fallback"}))
        assert trait.resolve_schema_for_adapter(adapter_info) == "fallback"

    def test_manifest_found_returns_schema_name(self, agent, adapter_info):
        trait = TrainingTrait(agent, TrainingConfig(schema={"name": "fallback"}))
        # Prime the cached factory with a fake manifest chain.
        manifest = MagicMock()
        manifest.source = MagicMock(schema_name="resolved")
        fake_factory = MagicMock()
        fake_factory.manifest.get_manifest.return_value = manifest
        trait._train_factory = fake_factory

        assert trait.resolve_schema_for_adapter(adapter_info) == "resolved"
        fake_factory.manifest.get_manifest.assert_called_once_with("abc123")

    def test_manifest_missing_raises(self, agent, adapter_info):
        trait = TrainingTrait(agent, TrainingConfig(schema={"name": "fallback"}))
        fake_factory = MagicMock()
        fake_factory.manifest.get_manifest.return_value = None
        trait._train_factory = fake_factory

        with pytest.raises(ManifestNotFoundError, match="abc123"):
            trait.resolve_schema_for_adapter(adapter_info)

    def test_manifest_without_source_raises(self, agent, adapter_info):
        trait = TrainingTrait(agent, TrainingConfig(schema={"name": "fallback"}))
        manifest = MagicMock()
        manifest.source = None
        fake_factory = MagicMock()
        fake_factory.manifest.get_manifest.return_value = manifest
        trait._train_factory = fake_factory

        with pytest.raises(ManifestNotFoundError):
            trait.resolve_schema_for_adapter(adapter_info)
