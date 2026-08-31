# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Training trait for kelt-side adapter-manifest lookup and schema resolution."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Self

from appinfra import DotDict
from llm_infer.client.types import AdapterInfo
from llm_kelt.training import Factory as TrainFactory

from ..base import BaseTrait


if TYPE_CHECKING:
    from ...agent import Agent


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

    pass


class TrainingTrait(BaseTrait):
    """Kelt-training-side capabilities: adapter manifest lookup and schema resolution.

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
        from llm_gent.core.traits import TrainingTrait, TrainingConfig

        agent.add_trait(TrainingTrait(agent, TrainingConfig(
            schema={"name": "public", "enforce": False},
            adapters={"lora": {"base_path": "~/adapters"}},
        )))
        agent.start()

        schema = agent.get_trait(TrainingTrait).resolve_schema_for_adapter(adapter_info)

    Dependency ownership: the trait can build its own ``TrainFactory``
    lazily from ``config.adapters.lora.base_path``, or accept a pre-built
    one via ``train_factory=`` at construction. Two seams:

    - Lazy default: ``train_factory=None`` at construction. First
      ``resolve_schema_for_adapter`` call builds a ``TrainFactory`` from
      config; trait owns it and drops the reference on ``on_stop``.
    - Direct injection (test / advanced): pass ``train_factory=`` with a
      pre-built factory; ``owns_train_factory=False`` keeps the reference
      across ``on_stop``. See ``.with_train_factory()`` for the
      immutable-view fluent form.

    ``TraitFactory._create_training`` uses the lazy default — the
    ``TrainFactory`` constructor reads the on-disk manifest registry, so
    eager construction at agent-start time is deliberately avoided.

    Lifecycle:
        - ``__init__``: takes an optional injected ``TrainFactory``.
        - ``on_start()``: no I/O; factory is created lazily on first lookup.
        - ``on_stop()``: drops the cached factory iff this trait owns it.
    """

    def __init__(
        self,
        agent: Agent,
        config: TrainingConfig | None = None,
        *,
        train_factory: TrainFactory | None = None,
        owns_train_factory: bool = False,
    ) -> None:
        """Initialize training trait.

        Args:
            agent: The agent this trait belongs to.
            config: Training configuration (None = empty DotDict).
            train_factory: Pre-built ``TrainFactory`` for injection. When None,
                a factory is built lazily from ``config.adapters.lora.base_path``
                on first lookup.
            owns_train_factory: If True, ``on_stop`` drops the factory
                reference. Set automatically to True when the trait builds a
                factory lazily itself. Injected factories default to False so
                the caller retains ownership across stop/start cycles.
        """
        super().__init__(agent)
        self.config = config if config is not None else TrainingConfig()
        self._train_factory: TrainFactory | None = train_factory
        self._owns_train_factory = owns_train_factory
        self._factory_lock = threading.Lock()

    def on_start(self) -> None:
        """No connections; TrainFactory is created lazily on first lookup."""
        self.agent.lg.trace(
            "training trait started",
            extra={
                "agent": self.agent.name,
                "has_factory": self._train_factory is not None,
                "owns_train_factory": self._owns_train_factory,
            },
        )

    def on_stop(self) -> None:
        """Drop the cached TrainFactory iff this trait owns it."""
        self.agent.lg.trace(
            "stopping training trait...",
            extra={
                "agent": self.agent.name,
                "owns_train_factory": self._owns_train_factory,
            },
        )
        dropped = self._owns_train_factory
        if self._owns_train_factory:
            self._train_factory = None
            self._owns_train_factory = False
        self.agent.lg.trace(
            "training trait stopped",
            extra={"agent": self.agent.name, "dropped_factory": dropped},
        )

    def with_train_factory(self, train_factory: TrainFactory) -> Self:
        """Return a new trait bound to ``train_factory``, detached from the registry.

        Immutable-view fluent (mirrors ``LLMTrait.with_router``): ``self``
        stays canonical for ``agent.get_trait(TrainingTrait)`` and its factory
        is unchanged. ``owns_train_factory`` on the returned trait is False;
        the caller retains ownership. For a persistent swap, call
        ``agent.replace_trait(new)``.
        """
        new = type(self)(
            self.agent, self.config, train_factory=train_factory, owns_train_factory=False
        )
        self.agent.lg.debug(
            "training trait detached with new factory",
            extra={"agent": self.agent.name, "detached_from_registry": True},
        )
        return new

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
            self.agent.lg.warning(
                "no adapters.lora.base_path configured, manifest lookup disabled",
                extra={"adapters_config": adapters_config},
            )
            return None

        registry_path = Path(base_path).expanduser()
        if not registry_path.exists():
            self.agent.lg.warning(
                "adapter registry path does not exist, manifest lookup disabled",
                extra={"path": str(registry_path)},
            )
            return None
        return registry_path

    def _get_train_factory(self) -> TrainFactory | None:
        """Get or create training factory for manifest lookups."""
        if self._train_factory is not None:
            return self._train_factory

        with self._factory_lock:
            if self._train_factory is not None:
                return self._train_factory

            registry_path = self._resolve_registry_path()
            if registry_path is None:
                return None

            self.agent.lg.trace(
                "building lazy train factory...",
                extra={"agent": self.agent.name, "registry_path": str(registry_path)},
            )
            self._train_factory = TrainFactory(self.agent.lg, registry_path)
            self._owns_train_factory = True
            self.agent.lg.trace(
                "train factory built",
                extra={"agent": self.agent.name, "owns_train_factory": True},
            )
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
            self.agent.lg.debug("no md5 in adapter info, using default schema")
            return self.default_schema

        train_factory = self._get_train_factory()
        if train_factory is None:
            self.agent.lg.debug("no train factory, using default schema")
            return self.default_schema

        manifest = train_factory.manifest.get_manifest(md5)
        if manifest and manifest.source and manifest.source.schema_name:
            schema = str(manifest.source.schema_name)
            self.agent.lg.trace("resolved schema from manifest", extra={"schema": schema})
            return schema

        raise ManifestNotFoundError(
            f"Manifest lookup failed for md5={md5}. "
            f"Cannot determine schema - refusing to use default to prevent data corruption."
        )
