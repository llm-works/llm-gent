# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for TrainingTrait — the platform's adapter manifest / schema resolution."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.core.traits import ScopedTrait, Trait, TraitName
from llm_gent.core.traits.builtin.training import (
    ManifestNotFoundError,
    TrainingConfig,
    TrainingTrait,
)


pytestmark = pytest.mark.unit

ADAPTERS = {"lora": {"base_path": "/adapters"}}


@pytest.fixture
def adapter_info():
    """Adapter info with a fake md5 fingerprint."""
    info = MagicMock()
    info.md5 = "abc123"
    return info


@pytest.fixture
def train_factory():
    """``TrainFactory`` patched, and the registry path made to exist; no disk read."""
    with (
        patch("llm_gent.core.traits.builtin.training.TrainFactory") as factory,
        patch("llm_gent.core.traits.builtin.training.Path.exists", return_value=True),
    ):
        yield factory


def _trait(schema: dict | None = None, adapters: dict | None = None) -> TrainingTrait:
    return TrainingTrait(MagicMock(), TrainingConfig(schema=schema, adapters=adapters))


def _manifest(schema_name: str | None) -> MagicMock:
    manifest = MagicMock()
    manifest.source = MagicMock(schema_name=schema_name) if schema_name else None
    return manifest


class TestTrait:
    def test_is_a_shared_trait_that_does_not_bind(self):
        trait = _trait()

        assert trait.trait_name == TraitName.TRAINING
        assert isinstance(trait, Trait)
        assert not isinstance(trait, ScopedTrait)

    def test_start_and_stop_touch_nothing(self, train_factory):
        trait = _trait(adapters=ADAPTERS)

        trait.on_start()
        trait.on_stop()

        train_factory.assert_not_called()


class TestDefaults:
    def test_default_schema_falls_back_to_public(self):
        assert _trait().default_schema == "public"

    def test_default_schema_from_config(self):
        assert _trait(schema={"name": "custom"}).default_schema == "custom"


class TestTrainFactory:
    def test_built_once_on_the_first_lookup(self, train_factory, adapter_info):
        train_factory.return_value.manifest.get_manifest.return_value = _manifest("resolved")
        trait = _trait(adapters=ADAPTERS)

        trait.resolve_schema_for_adapter(adapter_info)
        trait.resolve_schema_for_adapter(adapter_info)

        train_factory.assert_called_once()
        assert str(train_factory.call_args.args[1]) == "/adapters"

    def test_without_a_registry_path_none_is_built(self, train_factory, adapter_info):
        assert _trait(schema={"name": "fallback"}).resolve_schema_for_adapter(adapter_info) == (
            "fallback"
        )
        train_factory.assert_not_called()


class TestResolveSchemaForAdapter:
    def test_enforce_returns_default_schema(self, train_factory, adapter_info):
        trait = _trait(schema={"name": "forced", "enforce": True}, adapters=ADAPTERS)

        assert trait.resolve_schema_for_adapter(adapter_info) == "forced"
        train_factory.assert_not_called()

    def test_no_md5_returns_default_schema(self):
        info = MagicMock()
        info.md5 = None

        assert _trait(schema={"name": "fallback"}).resolve_schema_for_adapter(info) == "fallback"

    def test_manifest_found_returns_schema_name(self, train_factory, adapter_info):
        get_manifest = train_factory.return_value.manifest.get_manifest
        get_manifest.return_value = _manifest("resolved")

        schema = _trait(adapters=ADAPTERS).resolve_schema_for_adapter(adapter_info)

        assert schema == "resolved"
        get_manifest.assert_called_once_with("abc123")

    def test_manifest_missing_raises(self, train_factory, adapter_info):
        train_factory.return_value.manifest.get_manifest.return_value = None

        with pytest.raises(ManifestNotFoundError, match="abc123"):
            _trait(adapters=ADAPTERS).resolve_schema_for_adapter(adapter_info)

    def test_manifest_without_source_raises(self, train_factory, adapter_info):
        train_factory.return_value.manifest.get_manifest.return_value = _manifest(None)

        with pytest.raises(ManifestNotFoundError):
            _trait(adapters=ADAPTERS).resolve_schema_for_adapter(adapter_info)
