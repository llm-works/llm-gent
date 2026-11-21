# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for LLM backend."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_gent.core.llm import CompletionResult, HTTPBackend, LLMError, Message


pytestmark = pytest.mark.unit


class TestHTTPBackend:
    """Tests for HTTPBackend."""

    @pytest.fixture(autouse=True)
    def mock_httpx_client(self):
        """Mock httpx.Client for all tests to avoid real connections."""
        with patch("llm_gent.core.llm.backend.httpx.Client") as mock_class:
            self.mock_client = MagicMock()
            mock_class.return_value = self.mock_client
            yield mock_class

    def test_init_defaults(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        assert backend._base_url == "http://localhost:8000/v1"
        assert backend._timeout == 30.0
        assert backend._default_model == "default"

    def test_init_strips_trailing_slash(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1/")
        assert backend._base_url == "http://localhost:8000/v1"

    def test_init_custom_values(self):
        backend = HTTPBackend(
            base_url="http://api.example.com/v1",
            timeout=60.0,
            default_model="llama-3",
        )
        assert backend._base_url == "http://api.example.com/v1"
        assert backend._timeout == 60.0
        assert backend._default_model == "llama-3"

    def test_build_payload_basic(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]

        payload = backend._build_payload(messages, None, 0.7, None)

        assert payload["model"] == "default"
        assert payload["temperature"] == 0.7
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert payload["messages"][1] == {"role": "user", "content": "Hello"}
        assert "max_tokens" not in payload

    def test_build_payload_with_options(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        messages = [Message(role="user", content="Hi")]

        payload = backend._build_payload(messages, "gpt-4", 0.5, 100)

        assert payload["model"] == "gpt-4"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100

    def test_parse_response_success(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        result = backend._parse_response(response)

        assert result["id"] == "chatcmpl-123"
        assert result["content"] == "Hello!"
        assert result["model"] == "gpt-4"
        assert result["tokens_used"] == 15

    def test_parse_response_no_id_generates_uuid(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        response = {
            "model": "gpt-4",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {},
        }

        result = backend._parse_response(response)

        assert len(result["id"]) == 36  # UUID format

    def test_parse_response_calculates_tokens(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        result = backend._parse_response(response)

        assert result["tokens_used"] == 15

    def test_parse_response_invalid_format(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")

        with pytest.raises(LLMError, match="Invalid API response"):
            backend._parse_response({"invalid": "response"})

    def test_complete_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "resp-123",
            "model": "gpt-4",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"total_tokens": 20},
        }
        self.mock_client.post.return_value = mock_response

        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        messages = [Message(role="user", content="Hi")]

        result = backend.complete(messages)

        assert isinstance(result, CompletionResult)
        assert result.id == "resp-123"
        assert result.content == "Hello!"
        assert result.model == "gpt-4"
        assert result.tokens_used == 20
        assert result.latency_ms >= 0

    def test_complete_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        self.mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        messages = [Message(role="user", content="Hi")]

        with pytest.raises(LLMError, match="LLM request failed"):
            backend.complete(messages)

    def test_complete_connection_error(self):
        self.mock_client.post.side_effect = httpx.RequestError("Connection refused")

        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        messages = [Message(role="user", content="Hi")]

        with pytest.raises(LLMError, match="LLM connection failed"):
            backend.complete(messages)

    def test_close(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        backend.close()

        self.mock_client.close.assert_called_once()

    def test_load_adapter(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        assert backend._adapter_path is None

        backend.load_adapter("/path/to/adapter")

        assert backend._adapter_path == "/path/to/adapter"

    def test_unload_adapter(self):
        backend = HTTPBackend(base_url="http://localhost:8000/v1")
        backend._adapter_path = "/path/to/adapter"

        backend.unload_adapter()

        assert backend._adapter_path is None
