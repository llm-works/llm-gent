# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Structured output for a plain LLM call: the schema instruction, and parsing the reply.

:meth:`~llm_gent.core.llm.caller.Caller.chat` with ``output_schema=`` adds
the instruction to the request's system message, asks for a JSON object, and
validates the reply against the schema after cleaning the usual model
quirks (code fences, trailing text) with :class:`JSONCleaner`.
"""

from __future__ import annotations

import json
from typing import Any

from appinfra.log import Logger
from pydantic import BaseModel, ValidationError

from .backend import StructuredOutputError
from .json_cleaner import JSONCleaner


RESPONSE_FORMAT: dict[str, Any] = {"response_format": {"type": "json_object"}}
"""The request body that asks the backend for a JSON object."""


def with_schema_instruction(
    messages: list[dict[str, Any]], schema: type[BaseModel]
) -> list[dict[str, Any]]:
    """``messages`` with an instruction to answer in JSON matching ``schema``.

    The instruction is appended to the system message at index 0, or becomes
    one. A system message is only allowed at index 0, as Anthropic's Messages
    API and OpenAI's convention require.

    Raises:
        ValueError: A system message is at a position other than 0.
    """
    if any(m.get("role") == "system" for m in messages[1:]):
        raise ValueError(
            "system message must be at index 0; found role='system' at a non-zero position"
        )
    instruction = _instruction(schema)
    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = {**result[0], "content": f"{result[0]['content']}\n\n{instruction}"}
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result


def parse_structured(lg: Logger, content: str, schema: type[BaseModel]) -> BaseModel:
    """``content`` parsed as JSON, cleaned of model quirks, and validated against ``schema``.

    Raises:
        StructuredOutputError: The content is not JSON, is the schema itself
            instead of data, or does not match the schema.
    """
    cleaner = JSONCleaner()
    data = _load(lg, cleaner, content)
    if isinstance(data, dict):
        if cleaner._looks_like_schema(data):
            lg.warning(
                "LLM returned schema definition instead of data", extra={"raw_content": content}
            )
            raise StructuredOutputError(
                "LLM returned JSON schema definition instead of actual data. "
                "The model may not understand the structured output instruction."
            )
        data = cleaner.clean_parsed(data, set(schema.model_fields.keys()))
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        lg.debug(
            "schema validation failed",
            extra={"parsed_data": data, "expected_fields": list(schema.model_fields.keys())},
        )
        raise StructuredOutputError(f"Response doesn't match schema: {e}") from e


def _instruction(schema: type[BaseModel]) -> str:
    """The prompt asking for JSON that matches ``schema``."""
    return (
        "You must respond with valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```\n"
        "Respond ONLY with the JSON object, no other text."
    )


def _load(lg: Logger, cleaner: JSONCleaner, content: str) -> object:
    """``content`` cleaned and parsed as JSON."""
    try:
        return json.loads(cleaner.clean(content))
    except json.JSONDecodeError as e:
        lg.warning(
            "failed to parse LLM JSON output", extra={"raw_content": content, "error": str(e)}
        )
        raise StructuredOutputError(f"Invalid JSON in response: {e}") from e
