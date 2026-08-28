# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent identity - name and context_key for isolation.

Identity wraps agent name and context_key (for data isolation).
"""

from __future__ import annotations

from dataclasses import dataclass

from appinfra import DotDict


@dataclass(frozen=True)
class Identity:
    """Agent identity.

    Attributes:
        name: Agent name.
        context_key: Data isolation key (defaults to name if not specified).

    Examples:
        # Simple (context_key defaults to name)
        identity = Identity.from_config({"name": "jokester"})
        # → name = "jokester", context_key = "jokester"

        # Renaming (preserve data by explicit context_key)
        identity = Identity.from_config({"name": "joke-master", "context_key": "jokester"})
        # → name = "joke-master", context_key = "jokester"
    """

    name: str
    context_key: str

    @classmethod
    def from_name(cls, name: str) -> Identity:
        """Create identity from name (context_key = name).

        Args:
            name: Agent name.

        Returns:
            Identity with context_key = name.
        """
        return cls(name=name, context_key=name)

    @classmethod
    def from_config(
        cls, config: DotDict | None = None, defaults: DotDict | None = None
    ) -> Identity:
        """Create identity from configuration dict.

        Args:
            config: Configuration dict with name and optional context_key.
            defaults: Default values if not in config.

        Returns:
            Identity.

        Examples:
            # Simple (context_key defaults to name)
            Identity.from_config({"name": "jokester"})
            # → name = "jokester", context_key = "jokester"

            # Explicit context_key (for renaming without losing data)
            Identity.from_config({"name": "joke-master", "context_key": "jokester"})
            # → name = "joke-master", context_key = "jokester"
        """
        cfg: DotDict = config or DotDict()
        defs: DotDict = defaults or DotDict()

        # Get name (falsy values like "" fall through to defaults)
        name = cfg.get("name") or defs.get("name", "default")

        # Get context_key (defaults to name if not specified)
        context_key = cfg.get("context_key") or defs.get("context_key", name)

        # Validate that name and context_key are not empty
        if not name or not context_key:
            raise ValueError(
                "Identity name and context_key cannot be empty. "
                f"Got name={name!r}, context_key={context_key!r}"
            )

        return cls(name=name, context_key=context_key)
