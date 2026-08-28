# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Configuration parsing for rating system."""

from __future__ import annotations

from typing import Any

from appinfra import DotDict
from appinfra.log import Logger

from .models import (
    BatchConfig,
    Criteria,
    CriteriaConfig,
    ProviderConfig,
    ProviderType,
)


class ConfigParser:
    """Parse rating configuration from dict/YAML."""

    def __init__(self, lg: Logger) -> None:
        """Initialize config parser.

        Args:
            lg: Logger instance.
        """
        self._lg = lg

    def parse_providers(self, providers_config: list[Any] | dict[str, Any]) -> list[ProviderConfig]:
        """Parse provider configurations from config.

        Supports both list and dict formats:

          # List format (legacy)
          providers:
            - type: llm
              backend: ${llm.backends.local}
              enabled: true

          # Dict format (preferred - gives providers names)
          providers:
            anthropic:
              type: llm
              backend: ${llm.backends.anthropic}
              enabled: true

        Args:
            providers_config: List or dict of provider configurations.

        Returns:
            List of parsed ProviderConfig objects.
        """
        if not providers_config:
            return []

        # Handle dict format (name -> config)
        if isinstance(providers_config, dict):
            providers = []
            for name, cfg in providers_config.items():
                provider = self._parse_provider(cfg, name=name)
                if provider:
                    providers.append(provider)
            return providers

        # Handle list format (legacy)
        providers = []
        for provider_cfg in providers_config:
            provider = self._parse_provider(provider_cfg)
            if provider:
                providers.append(provider)

        return providers

    def _parse_provider(
        self, config: dict[str, Any] | DotDict, name: str | None = None
    ) -> ProviderConfig | None:
        """Parse a single provider configuration.

        Args:
            config: Provider config with type, backend, enabled.
            name: Optional provider name (from dict key).

        Returns:
            ProviderConfig or None if invalid.
        """
        config = self._ensure_dotdict(config)
        provider_type = self._parse_provider_type(config.get("type", "llm"))

        # Extract backend config (from ${llm.backends.local} expansion)
        backend = config.get("backend")
        if not backend:
            self._lg.warning(
                "provider missing backend config, skipping",
                extra={"name": name} if name else {},
            )
            return None

        backend = self._ensure_dotdict(backend)

        return ProviderConfig(
            provider_type=provider_type,
            model=str(backend.get("model", "auto")),
            backend=backend,
            enabled=config.get("enabled", True),
        )

    def _ensure_dotdict(self, config: dict[str, Any] | DotDict) -> DotDict:
        """Convert plain dict to DotDict if needed."""
        if isinstance(config, dict) and not isinstance(config, DotDict):
            return DotDict(config)
        return config

    def _parse_provider_type(self, provider_type_str: str) -> ProviderType:
        """Parse provider type string to enum, defaulting to LLM on error."""
        try:
            return ProviderType(provider_type_str)
        except ValueError:
            self._lg.warning(
                "invalid provider type, defaulting to llm",
                extra={"type": provider_type_str},
            )
            return ProviderType.LLM

    def parse_criteria(self, models_config: dict[str, Any]) -> dict[str, CriteriaConfig]:
        """Parse type-specific criteria from config.

        Format:
          models:
            atomic:
              solution:
                prompt: "You are rating a joke..."
                criteria:
                  - name: humor
                    description: "Is it funny?"
                    weight: 1.0

        Args:
            models_config: Models configuration dict.

        Returns:
            Dict mapping fact_type to CriteriaConfig.
        """
        type_criteria = {}

        # Parse atomic fact types
        atomic_config = models_config.get("atomic", {})
        for fact_type, type_config in atomic_config.items():
            parsed = self._parse_single_criteria(fact_type, type_config)
            if parsed:
                type_criteria[fact_type] = parsed

        return type_criteria

    def _parse_single_criteria(
        self, fact_type: str, config: dict[str, Any] | DotDict
    ) -> CriteriaConfig | None:
        """Parse criteria for a single fact type.

        Args:
            fact_type: Type of fact (e.g., "solution", "prediction").
            config: Type-specific config with prompt and criteria.

        Returns:
            CriteriaConfig or None if invalid.
        """
        if isinstance(config, dict) and not isinstance(config, DotDict):
            config = DotDict(config)

        # Get type-specific prompt
        prompt = self._get_or_default_prompt(fact_type, config)

        # Parse criteria list
        criteria = self._parse_criteria_list(config.get("criteria", []))
        if not criteria:
            self._lg.warning(
                "no criteria for fact type, skipping",
                extra={"fact_type": fact_type},
            )
            return None

        return CriteriaConfig(fact_type=fact_type, prompt=prompt, criteria=criteria)

    def _get_or_default_prompt(self, fact_type: str, config: dict[str, Any] | DotDict) -> str:
        """Get prompt from config or return default."""
        prompt = config.get("prompt", "")
        if not prompt:
            self._lg.warning(
                "missing prompt for fact type, using default",
                extra={"fact_type": fact_type},
            )
            prompt = f"Rate the following {fact_type}:"
        return str(prompt)

    def _parse_criteria_list(self, criteria_list: list[Any]) -> list[Criteria]:
        """Parse criteria list into Criteria objects."""
        criteria = []
        for crit_cfg in criteria_list:
            if isinstance(crit_cfg, str):
                criteria.append(Criteria(name=crit_cfg, description=f"Evaluate {crit_cfg}"))
            elif isinstance(crit_cfg, dict):
                name = crit_cfg.get("name")
                if not name:
                    self._lg.warning("criteria entry missing 'name' key, skipping")
                    continue
                criteria.append(
                    Criteria(
                        name=name,
                        description=crit_cfg.get("description", f"Evaluate {name}"),
                        weight=crit_cfg.get("weight", 1.0),
                    )
                )
        return criteria

    def parse_batch(self, batch_size: int | None = None) -> BatchConfig:
        """Parse batch configuration.

        Args:
            batch_size: Batch size from config (None = default).

        Returns:
            BatchConfig with parsed values.
        """
        return BatchConfig(
            enabled=batch_size is not None and batch_size > 1,
            size=batch_size if batch_size and batch_size > 0 else 5,
        )
