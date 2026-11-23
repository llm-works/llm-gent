# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""E2E tests for structured output with real LLM.

These tests require an LLM server at localhost:18100 that supports JSON mode.
Tests are skipped if no server is available.
"""

from collections.abc import Iterator

import httpx
import pytest
from appinfra.log import LogConfig, LoggerFactory
from pydantic import BaseModel

from llm_gent.core.llm.caller import Caller


pytestmark = [pytest.mark.e2e, pytest.mark.slow]

LLM_BASE_URL = "http://localhost:18100/v1"
LLM_MODEL = "default"


def llm_available() -> bool:
    """Check if LLM server is running."""
    try:
        response = httpx.get(f"{LLM_BASE_URL}/models", timeout=2.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


skip_no_llm = pytest.mark.skipif(
    not llm_available(),
    reason="LLM server not available at localhost:18100",
)


class MathAnswer(BaseModel):
    """Schema for math problem answers."""

    answer: int
    explanation: str


class Sentiment(BaseModel):
    """Schema for sentiment analysis."""

    sentiment: str
    confidence: float


@skip_no_llm
class TestStructuredOutputE2E:
    """E2E tests for structured output with real LLM responses."""

    @pytest.fixture
    def caller(self) -> Iterator[Caller]:
        """A Caller on a router connected to the LLM server."""
        from llm_infer.client import Factory as LLMClientFactory

        lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))
        config = {"type": "openai_compatible", "base_url": LLM_BASE_URL, "model": LLM_MODEL}
        caller = Caller(lg, LLMClientFactory(lg).from_config(config))
        yield caller
        caller.close()

    def test_math_problem_structured_output(self, caller: Caller):
        """Test structured output with a simple math problem."""
        messages = [{"role": "user", "content": "What is 15 + 27? Give the numeric answer."}]

        result = caller.chat(messages, temperature=0.0, output_schema=MathAnswer)

        # Verify parsing succeeded and types are correct
        assert result.parsed is not None
        assert isinstance(result.parsed, MathAnswer)
        assert isinstance(result.parsed.answer, int)
        assert isinstance(result.parsed.explanation, str)

    def test_sentiment_structured_output(self, caller: Caller):
        """Test structured output with sentiment analysis."""
        messages = [
            {
                "role": "user",
                "content": "Analyze the sentiment of: 'I love this product, it's amazing!'",
            },
        ]

        result = caller.chat(messages, temperature=0.0, output_schema=Sentiment)

        # Verify parsing succeeded and types are correct
        assert result.parsed is not None
        assert isinstance(result.parsed, Sentiment)
        assert isinstance(result.parsed.sentiment, str)
        assert isinstance(result.parsed.confidence, float)
