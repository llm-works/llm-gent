# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""LLM backend abstraction.

The top-level ``llm_gent`` package exports :class:`Caller` as ``LLMCaller``.
"""

from .backend import HTTPBackend, LLMBackend, LLMError, StructuredOutputError
from .caller import Caller, CallResult
from .types import CompletionResult, Message


__all__ = [
    "CallResult",
    "Caller",
    "CompletionResult",
    "HTTPBackend",
    "LLMBackend",
    "LLMError",
    "Message",
    "StructuredOutputError",
]
