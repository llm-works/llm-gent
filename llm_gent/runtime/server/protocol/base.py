# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Base protocol message types."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Base class for all protocol messages."""

    message_type: ClassVar[str]

    model_config = {"extra": "forbid"}


class Request(Message):
    """Base class for request messages.

    All requests must have an `id` field for IPC routing.
    """

    id: str = Field(description="Unique request ID for response routing")


class Response(Message):
    """Base class for response messages.

    All responses must have an `id` field matching the request.
    """

    id: str = Field(description="Request ID this response corresponds to")
    success: bool = Field(default=True, description="Whether the request succeeded")
    error: str | None = Field(default=None, description="Error message if success=False")
