# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The platform: a worker's shared scope, built once and passed to everything that needs it.

- :class:`Config` — the model and memory settings.
- :class:`Factory` — builds the shared clients and factories from a config.
- :class:`Context` — holds them, and closes what it owns.

Consumers usually alias the names: ``from llm_gent.core.platform import Factory as
PlatformFactory``.
"""

from .config import Config
from .context import Context
from .factory import Factory


__all__ = ["Config", "Context", "Factory"]
