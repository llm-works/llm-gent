"""Schema validation utilities for database operations."""

from __future__ import annotations

import re


# Pattern for valid PostgreSQL schema names (alphanumeric + underscore, starting with letter/underscore)
_VALID_SCHEMA_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_schema_name(schema: str) -> str:
    """Validate and return schema name to prevent SQL injection.

    Args:
        schema: Schema name to validate.

    Returns:
        The validated schema name.

    Raises:
        ValueError: If schema name contains invalid characters.
    """
    if not _VALID_SCHEMA_PATTERN.match(schema):
        raise ValueError(
            f"Invalid schema name '{schema}': must be alphanumeric with underscores, "
            f"starting with a letter or underscore"
        )
    return schema
