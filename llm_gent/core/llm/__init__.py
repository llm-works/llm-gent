# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""LLM backend abstraction.

The top-level ``llm_gent`` package exports :class:`Backend` and :class:`Error`
as ``LLMBackend`` and ``LLMError``.
"""

from .backend import Backend, Error, HTTPBackend, StructuredOutputError
from .caller import Caller, CallResult
from .types import CompletionResult, Message


__all__ = [
    "Backend",
    "CallResult",
    "Caller",
    "CompletionResult",
    "Error",
    "HTTPBackend",
    "Message",
    "StructuredOutputError",
]
