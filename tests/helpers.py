# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shared test utilities."""

import functools
import os
from unittest.mock import MagicMock

from appinfra.cas import FileStore, MemoryStore
from appinfra.log import Logger, quick_console_logger


@functools.cache
def make_test_logger() -> Logger:
    """Return the process-wide test logger (suppressed output).

    appinfra lets a process create a logger name once, so every test shares one instance.
    The session fixture ``test_logger`` creates it before the first test: appinfra's
    per-test cleanup destroys loggers a test creates, and keeps those that existed before.
    """
    return quick_console_logger("test", config={"level": "error"})


def memory_store() -> MemoryStore:
    """Return an empty in-memory checkpoint store."""
    return MemoryStore(make_test_logger())


def file_store(root: str | os.PathLike[str]) -> FileStore:
    """Return a checkpoint store under ``root``, without fsync: tests don't cut the power."""
    return FileStore(make_test_logger(), root, fsync=False)


def create_mock_trait(trait_class, **mock_attrs):
    """Create a mock trait that works with the registry.

    Creates an object whose type() returns the trait_class, allowing
    registry lookups to work correctly.

    Args:
        trait_class: The trait class this mock should impersonate.
        **mock_attrs: Attributes to add to the mock. Values can be:
            - MagicMock: Used directly
            - dict: Passed as **kwargs to MagicMock
            - bool/int/float/str: Stored directly (not as mock)
            - Other: Wrapped in MagicMock(return_value=...)

    Returns:
        Mock object that appears as instance of trait_class.
    """

    class MockHolder:
        pass

    obj = MockHolder()

    # Add mock methods/attributes to __dict__ BEFORE setting __class__
    # This avoids triggering property descriptors
    for attr_name, config in mock_attrs.items():
        if isinstance(config, MagicMock):
            # Already a mock, use it directly
            mock_attr = config
        elif isinstance(config, dict):
            # Dict config for MagicMock
            mock_attr = MagicMock(**config)
        else:
            # For simple scalar values, store them directly (not as MagicMock)
            # This is important for properties like has_embedder that return bool
            if isinstance(config, (bool, int, float, str)):
                mock_attr = config
            else:
                # For lists and other objects, create a MagicMock that returns them
                # Note: accessing this attribute returns a callable mock; call it to get the original value
                mock_attr = MagicMock(return_value=config)
        obj.__dict__[attr_name] = mock_attr

    # Set __class__ AFTER populating __dict__
    obj.__class__ = trait_class

    return obj
