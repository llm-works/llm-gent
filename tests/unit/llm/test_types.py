# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for LLM types."""

import pytest

from llm_gent.core.llm import CompletionResult, Message


pytestmark = pytest.mark.unit


class TestMessage:
    """Tests for Message type."""

    def test_create_system_message(self):
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."

    def test_create_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_create_tool_message(self):
        msg = Message(role="tool", content="result", tool_call_id="call-1")
        assert msg.role == "tool"
        assert msg.content == "result"
        assert msg.tool_call_id == "call-1"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError):
            Message(role="invalid", content="test")

    def test_serialization(self):
        msg = Message(role="user", content="test")
        data = msg.model_dump(exclude_none=True)
        assert data == {"role": "user", "content": "test"}

    def test_serialization_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "test"}}],
        )
        data = msg.model_dump(exclude_none=True)
        assert data["role"] == "assistant"
        assert data["tool_calls"] == [
            {"id": "call-1", "type": "function", "function": {"name": "test"}}
        ]

    def test_serialization_tool_message(self):
        msg = Message(role="tool", content="result", tool_call_id="call-1")
        data = msg.model_dump(exclude_none=True)
        assert data == {"role": "tool", "content": "result", "tool_call_id": "call-1"}

    def test_deserialization(self):
        data = {"role": "assistant", "content": "response"}
        msg = Message.model_validate(data)
        assert msg.role == "assistant"
        assert msg.content == "response"


class TestCompletionResult:
    """Tests for CompletionResult type."""

    def test_create_result(self):
        result = CompletionResult(
            id="resp-123",
            content="Hello!",
            model="gpt-4",
            tokens_used=50,
            latency_ms=250,
        )
        assert result.id == "resp-123"
        assert result.content == "Hello!"
        assert result.model == "gpt-4"
        assert result.tokens_used == 50
        assert result.latency_ms == 250
        assert result.tool_calls is None

    def test_create_result_with_tool_calls(self):
        tool_calls = [{"id": "call-1", "type": "function", "function": {"name": "test"}}]
        result = CompletionResult(
            id="resp-123",
            content="",
            model="gpt-4",
            tokens_used=50,
            latency_ms=250,
            tool_calls=tool_calls,
        )
        assert result.tool_calls == tool_calls

    def test_serialization(self):
        result = CompletionResult(
            id="resp-123",
            content="Hello!",
            model="gpt-4",
            tokens_used=50,
            latency_ms=250,
        )
        data = result.model_dump(exclude_none=True)
        assert data["id"] == "resp-123"
        assert data["content"] == "Hello!"
        assert data["model"] == "gpt-4"
        assert data["tokens_used"] == 50
        assert data["latency_ms"] == 250
