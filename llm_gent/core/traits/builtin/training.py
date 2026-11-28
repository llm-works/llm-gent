# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Training trait for kelt-side adapter-manifest lookup and schema resolution."""

from __future__ import annotations

import threading
from pathlib import Path

from appinfra import DotDict
from appinfra.log import Logger
from llm_infer.client.types import AdapterInfo
from llm_kelt.training import Factory as TrainFactory


# Type alias for Training configuration
TrainingConfig = DotDict
"""Training configuration as DotDict.

Expected fields:
    schema: Schema config dict with 'name' (default schema) and 'enforce'
        (when True, always return the default schema and skip manifest lookup).
    adapters: Adapter registry config dict; adapters.lora.base_path points at
        the on-disk manifest registry read by llm_kelt.training.Factory.
"""


class ManifestNotFoundError(Exception):
    """Raised when adapter manifest cannot be found for schema resolution."""


class TrainingTrait:
    """Kelt-training-side capabilities: adapter manifest lookup and schema resolution.

    One per platform, shared by every agent: it holds nothing per agent, so it
    has no binding and ``agent.require(TrainingTrait)`` is the platform's
    instance. The platform builds it from the ``schema`` and ``adapters`` keys
    of its ``memory`` section; without them every lookup falls back to the
    default schema.

    Separated from MemoryTrait so agents that only need memory (facts, embeddings,
    feedback) do not carry the manifest-lookup surface, and agents that need adapter
    provenance do not have to attach the full memory stack.

    Capabilities:
        - resolve_schema_for_adapter(): Look up an adapter's manifest to pick a
          storage schema for downstream memory operations. Falls back to the
          configured default schema when the config's ``schema.enforce`` flag is
          set, when the adapter has no md5, or when the registry is missing.
          Raises :class:`ManifestNotFoundError` when a lookup runs but no matching
          manifest is found.

    Example:
        schema = agent.require(TrainingTrait).resolve_schema_for_adapter(adapter_info)

    The ``TrainFactory`` is built on the first lookup that needs it, from
    ``config.adapters.lora.base_path``, and kept for the platform's life: its
    constructor reads the on-disk manifest registry, so no agent or platform
    start pays for it unless a lookup runs.
    """

    trait_name = "training"

    def __init__(self, lg: Logger, config: TrainingConfig) -> None:
        """Initialize training trait.

        Args:
            lg: Logger instance.
            config: The platform's training configuration (``schema``, ``adapters``).
        """
        self._lg = lg
        self.config = config
        self._train_factory: TrainFactory | None = None
        self._factory_lock = threading.Lock()

    def on_start(self) -> None:
        """No I/O: the TrainFactory is built on the first lookup."""

    def on_stop(self) -> None:
        """Nothing to release: the TrainFactory holds no connection."""

    @property
    def default_schema(self) -> str:
        """Default schema name from config (schema.name), falling back to 'public'."""
        schema_config = self.config.get("schema") or {}
        return str(schema_config.get("name") or "public")

    def _resolve_registry_path(self) -> Path | None:
        """Resolve and validate the adapter registry path from config."""
        adapters_config = self.config.get("adapters") or {}
        lora_config = adapters_config.get("lora") or {}
        base_path = lora_config.get("base_path")

        if not base_path:
            self._lg.warning(
                "no adapters.lora.base_path configured, manifest lookup disabled",
                extra={"adapters_config": adapters_config},
            )
            return None

        registry_path = Path(base_path).expanduser()
        if not registry_path.exists():
            self._lg.warning(
                "adapter registry path does not exist, manifest lookup disabled",
                extra={"path": str(registry_path)},
            )
            return None
        return registry_path

    def _get_train_factory(self) -> TrainFactory | None:
        """The platform's TrainFactory, built once under the lock on first use."""
        if self._train_factory is not None:
            return self._train_factory

        with self._factory_lock:
            if self._train_factory is not None:
                return self._train_factory

            registry_path = self._resolve_registry_path()
            if registry_path is None:
                return None

            self._train_factory = TrainFactory(self._lg, registry_path)
            self._lg.trace("train factory built", extra={"registry_path": str(registry_path)})
            return self._train_factory

    def resolve_schema_for_adapter(self, adapter_info: AdapterInfo) -> str:
        """Resolve the schema for an adapter by looking up its manifest.

        If ``schema.enforce=True`` is set in config, always returns the default
        schema and skips manifest lookup entirely.

        Args:
            adapter_info: Adapter info from LLM response.

        Returns:
            Schema name from the manifest, or the default schema when enforce
            is set, the adapter has no md5, or no registry is configured.

        Raises:
            ManifestNotFoundError: When a lookup runs but no manifest is found
                for the adapter's md5 — refuses to fall back to the default
                schema in that case to prevent data corruption.
        """
        schema_config = self.config.get("schema") or {}
        if schema_config.get("enforce", False):
            return self.default_schema

        md5 = adapter_info.md5
        if not md5:
            self._lg.debug("no md5 in adapter info, using default schema")
            return self.default_schema

        train_factory = self._get_train_factory()
        if train_factory is None:
            self._lg.debug("no train factory, using default schema")
            return self.default_schema

        manifest = train_factory.manifest.get_manifest(md5)
        if manifest and manifest.source and manifest.source.schema_name:
            schema = str(manifest.source.schema_name)
            self._lg.trace("resolved schema from manifest", extra={"schema": schema})
            return schema

        raise ManifestNotFoundError(
            f"Manifest lookup failed for md5={md5}. "
            f"Cannot determine schema - refusing to use default to prevent data corruption."
        )
