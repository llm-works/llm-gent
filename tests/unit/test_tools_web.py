# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for WebFetchTool and WebSearchTool."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent import (
    ToolResult,
    WebFetchTool,
    WebSearchTool,
)


pytestmark = pytest.mark.unit


@pytest.fixture()
def mock_lg():
    """Create mock Logger."""
    return MagicMock()


@pytest.fixture()
def web_fetch(mock_lg):
    """Create WebFetchTool for tests that need it."""
    return WebFetchTool(mock_lg)


# ---------------------------------------------------------------------------
# Mock search backend
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = [
    {
        "title": "First Result Title",
        "url": "https://example.com/page1",
        "snippet": "First snippet.",
    },
    {"title": "Second Result", "url": "https://example.com/page2", "snippet": "Second snippet."},
]


class MockSearchBackend:
    """Test backend that returns canned results."""

    def __init__(
        self,
        results: list[dict[str, str]] | None = None,
        side_effects: list[list[dict[str, str]] | None] | None = None,
    ) -> None:
        self._results = results if results is not None else list(_SAMPLE_RESULTS)
        self._side_effects = list(side_effects) if side_effects else None
        self.call_count = 0
        self.last_query: str | None = None
        self.last_max_results: int | None = None
        self.last_offset: int | None = None

    def search(self, query: str, max_results: int, offset: int = 0) -> list[dict[str, str]] | None:
        self.call_count += 1
        self.last_query = query
        self.last_max_results = max_results
        self.last_offset = offset
        if self._side_effects:
            return self._side_effects.pop(0)
        return self._results[offset : offset + max_results]


# ---------------------------------------------------------------------------
# WebFetchTool tests
# ---------------------------------------------------------------------------


class TestWebFetchTool:
    """Tests for WebFetchTool."""

    def _mock_http_result(self, output: str, success: bool = True) -> ToolResult:
        return ToolResult(success=success, output=output, error=None if success else "error")

    def test_tool_properties(self, mock_lg):
        tool = WebFetchTool(mock_lg)
        assert tool.name == "web_fetch"
        assert "web page" in tool.description.lower() or "readable" in tool.description.lower()
        assert tool.parameters["required"] == ["url"]

    def test_protocol_compliance(self, mock_lg):
        from llm_gent import Tool

        tool = WebFetchTool(mock_lg)
        assert isinstance(tool, Tool)

    def test_fetch_html_extracts_content(self, mock_lg):
        """HTML content should be extracted to readable text."""
        tool = WebFetchTool(mock_lg)
        html_content = "<html><body><p>Hello World</p></body></html>"

        with (
            patch.object(tool._http, "execute") as mock_fetch,
            patch("trafilatura.extract", return_value="Hello World") as mock_extract,
        ):
            mock_fetch.return_value = self._mock_http_result(html_content)
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert result.output == "Hello World"
        mock_extract.assert_called_once()

    def test_fetch_json_passthrough(self, mock_lg):
        """JSON content should not be converted."""
        tool = WebFetchTool(mock_lg)
        json_content = '{"key": "value", "count": 42}'

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = self._mock_http_result(json_content)
            result = tool.execute(url="https://api.example.com/data")

        assert result.success is True
        assert result.output == json_content

    def test_fetch_plain_text_passthrough(self, mock_lg):
        """Plain text content should not be converted."""
        tool = WebFetchTool(mock_lg)
        text_content = "Just some plain text content."

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = self._mock_http_result(text_content)
            result = tool.execute(url="https://example.com/file.txt")

        assert result.success is True
        assert result.output == text_content

    def test_fetch_error_propagated(self, mock_lg):
        """Errors from HTTPFetchTool should propagate."""
        tool = WebFetchTool(mock_lg)

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = ToolResult(
                success=False, output="", error="Connection refused"
            )
            result = tool.execute(url="https://example.com")

        assert result.success is False
        assert "Connection refused" in result.error

    def test_truncation(self, mock_lg):
        """Long content should be truncated."""
        tool = WebFetchTool(mock_lg, max_text_length=100)
        long_text = "word " * 200  # ~1000 chars

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = self._mock_http_result(long_text)
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert len(result.output) <= 100
        assert "(truncated" in result.output

    def test_no_truncation_when_within_limit(self, mock_lg):
        """Short content should not be truncated."""
        tool = WebFetchTool(mock_lg, max_text_length=1000)
        short_text = "short content"

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = self._mock_http_result(short_text)
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert result.output == short_text
        assert "(truncated" not in result.output

    def test_delegates_to_http_fetch(self, mock_lg):
        """Verifies HTTPFetchTool is used under the hood."""
        tool = WebFetchTool(
            mock_lg,
            timeout=15.0,
            allowed_domains=["example.com"],
            block_private_ips=True,
        )
        assert tool._http._timeout == 15.0
        assert tool._http._allowed_domains == {"example.com"}
        assert tool._http._block_private_ips is True

    def test_sets_browser_user_agent(self, mock_lg):
        """Default headers should include a User-Agent."""
        tool = WebFetchTool(mock_lg)
        assert "User-Agent" in tool._http._default_headers

    def test_custom_user_agent(self, mock_lg):
        """Custom User-Agent should override the default."""
        tool = WebFetchTool(mock_lg, user_agent="MyBot/1.0")
        assert tool._http._default_headers["User-Agent"] == "MyBot/1.0"

    def test_to_openai_function(self, mock_lg):
        tool = WebFetchTool(mock_lg)
        func = tool.to_openai_function()
        assert func["function"]["name"] == "web_fetch"
        assert "url" in func["function"]["parameters"]["properties"]

    def test_html_extraction_fallback(self, mock_lg):
        """When trafilatura returns None, raw content is returned."""
        tool = WebFetchTool(mock_lg)
        html = "<html><body></body></html>"

        with (
            patch.object(tool._http, "execute") as mock_fetch,
            patch("trafilatura.extract", return_value=None),
        ):
            mock_fetch.return_value = self._mock_http_result(html)
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert result.output == html

    def test_fetch_raw(self, mock_lg):
        """fetch_raw returns raw HTTP response without HTML extraction."""
        tool = WebFetchTool(mock_lg)
        html = "<html><body><p>Raw HTML</p></body></html>"

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = self._mock_http_result(html)
            result = tool.fetch_raw(url="https://example.com")

        assert result.success is True
        assert result.output == html  # No extraction, raw HTML preserved


# ---------------------------------------------------------------------------
# WebSearchTool tests
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_tool_properties(self, mock_lg):
        tool = WebSearchTool(mock_lg, backend=MockSearchBackend())
        assert tool.name == "web_search"
        assert "search" in tool.description.lower()
        assert tool.parameters["required"] == ["query"]

    def test_protocol_compliance(self, mock_lg):
        from llm_gent import Tool

        tool = WebSearchTool(mock_lg, backend=MockSearchBackend())
        assert isinstance(tool, Tool)

    def test_search_success(self, mock_lg):
        """Search returns structured results from backend."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        result = tool.execute(query="test search")

        assert result.success is True
        assert "First Result Title" in result.output
        assert "Second Result" in result.output
        assert "https://example.com/page1" in result.output
        assert "First snippet" in result.output
        assert backend.last_query == "test search"

    def test_search_max_results(self, mock_lg):
        """max_results is forwarded to the backend."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        result = tool.execute(query="test", max_results=1)

        assert result.success is True
        assert "First Result Title" in result.output
        assert "Second Result" not in result.output
        assert backend.last_max_results == 1

    def test_search_no_results(self, mock_lg):
        """Empty results from backend."""
        backend = MockSearchBackend(results=[])
        tool = WebSearchTool(mock_lg, backend=backend)
        result = tool.execute(query="xyzzy nonexistent")

        assert result.success is True
        assert "No results found" in result.output

    def test_search_missing_query(self, mock_lg):
        tool = WebSearchTool(mock_lg, backend=MockSearchBackend())
        result = tool.execute()
        assert result.success is False
        assert "query" in result.error.lower()

    def test_search_empty_query(self, mock_lg):
        tool = WebSearchTool(mock_lg, backend=MockSearchBackend())
        result = tool.execute(query="   ")
        assert result.success is False
        assert "query" in result.error.lower()

    def test_search_max_results_none(self, mock_lg):
        """max_results=None (LLM sends null) should default to 5, not crash."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        result = tool.execute(query="test", max_results=None)

        assert result.success is True
        assert backend.last_max_results == 5

    def test_max_results_floor_clamped(self, mock_lg):
        """max_results below 1 is clamped to 1; no upper cap."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)

        tool.execute(query="test", max_results=100)
        assert backend.last_max_results == 100

        tool.execute(query="test", max_results=0)
        assert backend.last_max_results == 1

    def test_retry_on_retriable_failure(self, mock_lg):
        """Backend returning None triggers backoff + retry; succeeds on second attempt."""
        backend = MockSearchBackend(side_effects=[None, _SAMPLE_RESULTS])
        tool = WebSearchTool(mock_lg, backend=backend, retry_delay=0.0)

        with patch("llm_gent.core.tools.builtin.web.search.time.sleep") as mock_sleep:
            result = tool.execute(query="test query")

        assert result.success is True
        assert "First Result Title" in result.output
        assert backend.call_count == 2
        mock_sleep.assert_called_once_with(0.0)

    def test_retry_exhausted(self, mock_lg):
        """Backend returning None twice should fail with clear error."""
        backend = MockSearchBackend(side_effects=[None, None])
        tool = WebSearchTool(mock_lg, backend=backend, retry_delay=0.0)

        with patch("llm_gent.core.tools.builtin.web.search.time.sleep"):
            result = tool.execute(query="test")

        assert result.success is False
        assert "retriable error twice" in result.error.lower()
        assert backend.call_count == 2

    def test_to_openai_function(self, mock_lg):
        tool = WebSearchTool(mock_lg, backend=MockSearchBackend())
        func = tool.to_openai_function()
        assert func["function"]["name"] == "web_search"
        assert "query" in func["function"]["parameters"]["properties"]

    def test_offset_forwarded(self, mock_lg):
        """offset is forwarded to the backend."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        tool.execute(query="test", offset=8)
        assert backend.last_offset == 8

    def test_offset_default_zero(self, mock_lg):
        """offset defaults to 0 when not provided."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        tool.execute(query="test")
        assert backend.last_offset == 0

    def test_offset_none(self, mock_lg):
        """offset=None (LLM sends null) should default to 0."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        tool.execute(query="test", offset=None)
        assert backend.last_offset == 0

    def test_offset_negative_clamped(self, mock_lg):
        """Negative offset is clamped to 0."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        tool.execute(query="test", offset=-5)
        assert backend.last_offset == 0

    def test_offset_invalid_string(self, mock_lg):
        """Non-numeric offset falls back to 0."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        tool.execute(query="test", offset="abc")
        assert backend.last_offset == 0

    def test_query_is_stripped(self, mock_lg):
        """Leading/trailing whitespace in query should be stripped."""
        backend = MockSearchBackend()
        tool = WebSearchTool(mock_lg, backend=backend)
        tool.execute(query="  hello world  ")
        assert backend.last_query == "hello world"


# ---------------------------------------------------------------------------
# ToolFactory integration tests
# ---------------------------------------------------------------------------


class TestToolFactoryWeb:
    """Tests for web tools in ToolFactory."""

    def test_factory_creates_web_fetch(self, mock_lg):
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        tool = factory.create("web_fetch")
        assert isinstance(tool, WebFetchTool)
        assert tool.name == "web_fetch"

    def test_factory_creates_web_search_with_backend(self, mock_lg):
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        backend = MockSearchBackend()
        tool = factory.create("web_search", {"backend": backend})
        assert isinstance(tool, WebSearchTool)
        assert tool.name == "web_search"

    def test_factory_creates_web_search_via_setter(self, mock_lg):
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        backend = MockSearchBackend()
        factory.set_web_search_backend(backend)
        tool = factory.create("web_search")
        assert isinstance(tool, WebSearchTool)

    def test_factory_web_search_returns_none_without_backend(self, mock_lg):
        """Creating web_search without a backend returns None (skip gracefully)."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        assert factory.create("web_search") is None
        mock_lg.warning.assert_called_once()

    def test_factory_web_fetch_with_config(self, mock_lg):
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        tool = factory.create("web_fetch", {"timeout": 10.0, "max_text_length": 5000})
        assert isinstance(tool, WebFetchTool)
        assert tool._max_text_length == 5000
        assert tool._http._timeout == 10.0

    def test_factory_web_search_with_config(self, mock_lg):
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        backend = MockSearchBackend()
        tool = factory.create("web_search", {"backend": backend, "retry_delay": 1.0})
        assert isinstance(tool, WebSearchTool)
        assert tool._retry_delay == 1.0

    def test_factory_lazy_web_fetch(self, mock_lg):
        """WebFetchTool is not created until needed (lazy init)."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        assert factory._web_fetch is None
        factory.create("shell")
        assert factory._web_fetch is None

    def test_factory_web_search_backend_protocol(self, mock_lg):
        """Backend must satisfy WebSearchBackend protocol."""
        from llm_gent import WebSearchBackend

        backend = MockSearchBackend()
        assert isinstance(backend, WebSearchBackend)

    def test_factory_config_not_mutated(self, mock_lg):
        """create('web_search', config) must not mutate the caller's dict."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        backend = MockSearchBackend()
        config = {"backend": backend, "retry_delay": 1.0}
        factory.create("web_search", config)
        assert "backend" in config, "config dict was mutated by factory.create()"

    def test_factory_backend_dict_missing_factory_key(self, mock_lg):
        """Backend dict without 'factory' key raises ValueError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(ValueError, match="must include 'factory' key"):
            factory.create("web_search", {"backend": {}})

    def test_factory_backend_dict_factory_not_string(self, mock_lg):
        """Backend dict with non-string 'factory' raises ValueError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(ValueError, match="'factory' must be a string"):
            factory.create("web_search", {"backend": {"factory": 123}})

    def test_factory_backend_dict_invalid_factory_path(self, mock_lg):
        """Backend dict with invalid factory path raises ImportError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(ImportError):
            factory.create("web_search", {"backend": {"factory": "nonexistent.module.Factory"}})

    def test_factory_backend_dict_non_factory_class(self, mock_lg):
        """Backend dict pointing to non-WebSearchBackendFactory raises TypeError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(TypeError, match="not a WebSearchBackendFactory subclass"):
            factory.create(
                "web_search",
                {"backend": {"factory": "llm_gent.core.tools.factory.ToolFactory"}},
            )

    def test_factory_backend_dict_valid_factory(self, mock_lg):
        """Backend dict with valid factory path creates tool successfully."""
        from unittest.mock import MagicMock

        from llm_gent import ToolFactory, WebSearchBackendFactory

        # Create a mock factory class
        mock_backend = MockSearchBackend()
        mock_factory_cls = MagicMock(spec=WebSearchBackendFactory)
        mock_factory_cls.create.return_value = mock_backend

        factory = ToolFactory(mock_lg)
        with patch.object(factory, "_import_factory", return_value=mock_factory_cls):
            tool = factory.create(
                "web_search",
                {"backend": {"factory": "some.module.Factory", "config": {"timeout": 5.0}}},
            )

        assert tool is not None
        assert tool.name == "web_search"
        mock_factory_cls.create.assert_called_once()
        # Verify config was passed
        call_args = mock_factory_cls.create.call_args
        assert call_args[0][0] == mock_lg  # Logger
        assert call_args[0][1].get("timeout") == 5.0  # Config
