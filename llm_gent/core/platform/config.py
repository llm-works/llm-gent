# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Platform configuration: the model and memory settings shared by a worker's agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..llm.config import merge_llm_config


PACKAGED_LLM_CONFIG = Path(__file__).resolve().parents[2] / "etc" / "llm.yaml"


class Config(BaseModel):
    """Settings the platform :class:`~llm_gent.core.platform.Factory` builds a platform from.

    ``llm`` is llm-infer's router config (``default``, ``backends``, ...),
    merged over the packaged ``etc/llm.yaml``: an empty ``llm`` uses the
    packaged backends as they are, and a partial one overrides only what it
    names. ``memory`` holds the memory settings (``db``, ``llm``,
    ``embedder_url``, ``schema``, ``adapters``, ...); without it no
    memory-backed trait or tool can be created.
    """

    llm: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] | None = None

    def router_config(self) -> dict[str, Any]:
        """The router config the platform builds: ``llm`` merged over the packaged one."""
        return merge_llm_config(packaged_llm_config(), self.llm)


def packaged_llm_config() -> dict[str, Any]:
    """The router config shipped with llm-gent (``etc/llm.yaml``)."""
    config: dict[str, Any] = yaml.safe_load(PACKAGED_LLM_CONFIG.read_text())
    return config
