# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for structured output on a plain call: ``Caller.chat(output_schema=)``."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llm_gent import StructuredOutputError
from llm_gent.core.llm.caller import Caller


pytestmark = pytest.mark.unit


class Answer(BaseModel):
    """The schema replies are validated against."""

    answer: str
    confidence: float


def _caller(content: str) -> Caller:
    """A caller whose router replies with ``content``."""
    response = MagicMock()
    response.content = content
    response.model = "test-model"
    response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
    response.tool_calls = None
    response.adapter = None
    router = MagicMock()
    router.chat.return_value = response
    return Caller(MagicMock(), router)


def _sent(caller: Caller) -> dict:
    """The keyword arguments of the router call."""
    return caller.router.chat.call_args.kwargs


class TestStructuredOutput:
    def test_reply_parsed_into_schema(self):
        caller = _caller('{"answer": "42", "confidence": 0.95}')

        result = caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

        assert result.parsed == Answer(answer="42", confidence=0.95)

    def test_fenced_reply_is_cleaned(self):
        caller = _caller('```json\n{"answer": "42", "confidence": 0.5}\n```')

        result = caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

        assert result.parsed == Answer(answer="42", confidence=0.5)

    def test_instruction_appended_to_system_message(self):
        caller = _caller('{"answer": "a", "confidence": 1.0}')

        caller.chat(
            [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Q?"}],
            output_schema=Answer,
        )

        system = _sent(caller)["messages"][0]
        assert system["role"] == "system"
        assert system["content"].startswith("You are helpful.\n\n")
        assert '"confidence"' in system["content"]

    def test_instruction_becomes_system_message(self):
        caller = _caller('{"answer": "a", "confidence": 1.0}')

        caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

        messages = _sent(caller)["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "JSON" in messages[0]["content"]

    def test_caller_messages_not_mutated(self):
        caller = _caller('{"answer": "a", "confidence": 1.0}')
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]

        caller.chat(messages, output_schema=Answer)

        assert messages[0] == {"role": "system", "content": "sys"}

    def test_json_mode_merged_into_extra_body(self):
        caller = _caller('{"answer": "a", "confidence": 1.0}')

        caller.chat(
            [{"role": "user", "content": "Q?"}], extra_body={"foo": 1}, output_schema=Answer
        )

        assert _sent(caller)["extra_body"] == {
            "foo": 1,
            "response_format": {"type": "json_object"},
        }

    def test_no_schema_leaves_parsed_none(self):
        caller = _caller("plain text")

        result = caller.chat([{"role": "user", "content": "Q?"}])

        assert result.parsed is None
        assert _sent(caller)["messages"] == [{"role": "user", "content": "Q?"}]

    def test_dry_run_skips_parsing(self):
        caller = Caller(MagicMock(), MagicMock(), dry_run=True)

        result = caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

        assert result.dry_run is True
        assert result.parsed is None


class TestStructuredOutputErrors:
    def test_invalid_json_raises(self):
        caller = _caller("not valid json {")

        with pytest.raises(StructuredOutputError, match="Invalid JSON"):
            caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

    def test_missing_field_raises(self):
        caller = _caller('{"answer": "a"}')

        with pytest.raises(StructuredOutputError, match="doesn't match schema"):
            caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

    def test_wrong_type_raises(self):
        caller = _caller('{"answer": "a", "confidence": "high"}')

        with pytest.raises(StructuredOutputError, match="doesn't match schema"):
            caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

    def test_schema_echoed_back_raises(self):
        caller = _caller(json.dumps(Answer.model_json_schema()))

        with pytest.raises(StructuredOutputError, match="schema definition"):
            caller.chat([{"role": "user", "content": "Q?"}], output_schema=Answer)

    def test_tools_and_schema_raises(self):
        caller = _caller("{}")
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]

        with pytest.raises(ValueError, match="Cannot use both tools and output_schema"):
            caller.chat([{"role": "user", "content": "Q?"}], tools=tools, output_schema=Answer)
        caller.router.chat.assert_not_called()

    def test_system_message_past_index_zero_raises(self):
        caller = _caller("{}")
        messages = [{"role": "user", "content": "Q?"}, {"role": "system", "content": "stray"}]

        with pytest.raises(ValueError, match="system message must be at index 0"):
            caller.chat(messages, output_schema=Answer)
