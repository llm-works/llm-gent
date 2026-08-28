# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""LLM backend abstraction."""

from .backend import HTTPBackend, LLMBackend, LLMError, StructuredOutputError
from .caller import CallResult, LLMCaller
from .types import CompletionResult, Message


__all__ = [
    "CallResult",
    "CompletionResult",
    "HTTPBackend",
    "LLMBackend",
    "LLMCaller",
    "LLMError",
    "Message",
    "StructuredOutputError",
]
