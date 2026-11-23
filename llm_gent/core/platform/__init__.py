# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The platform: what a process's agents and runs share, built once and passed to them.

- :class:`Config` — the model and memory settings.
- :class:`Factory` — builds a platform from a config.
- :class:`Platform` — holds the factories, the trait registry and one SAIA per
  role, and closes what it owns. Also exported as :class:`llm_gent.core.Platform`.
- :class:`SAIAFactory` — how a role becomes a SAIA instance.

Consumers usually alias the factory: ``from llm_gent.core.platform import Factory as
PlatformFactory``.
"""

from ._platform import Closeable, Platform, SAIAFactory, close_resources
from .config import Config
from .factory import Factory


__all__ = ["Closeable", "Config", "Factory", "Platform", "SAIAFactory", "close_resources"]
