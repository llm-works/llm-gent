# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Config(BaseModel):
    """Agent configuration."""

    name: str
    default_prompt: str = "You are a helpful assistant."
    model: str = "default"

    # Fact injection (Phase 2)
    fact_injection: Literal["all", "rag", "none"] = "all"
    max_facts: int = 20

    # RAG defaults (Phase 4)
    rag_top_k: int = 5
    rag_min_similarity: float = 0.3

    # Response tracking for feedback (bounded to prevent memory leaks)
    max_tracked_responses: int = 1000
