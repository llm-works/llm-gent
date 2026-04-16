"""Tests for WebFetchTool, WebSearchTool, and BraveSearchBackend."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_gent import (
    BraveSearchBackend,
    SerperSearchBackend,
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


# ---------------------------------------------------------------------------
# BraveSearchBackend tests
# ---------------------------------------------------------------------------

_BRAVE_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Python Tutorial",
                "url": "https://docs.python.org/3/tutorial/",
                "description": "The official Python tutorial.",
            },
            {
                "title": "Real Python",
                "url": "https://realpython.com/",
                "description": "Learn Python programming.",
            },
        ]
    }
}


class TestBraveSearchBackend:
    """Tests for BraveSearchBackend."""

    def test_protocol_compliance(self, mock_lg):
        from llm_gent import WebSearchBackend

        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        assert isinstance(backend, WebSearchBackend)

    def test_api_key_required(self, mock_lg, monkeypatch):
        """Must raise ValueError if no API key provided."""
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            BraveSearchBackend(mock_lg)

    def test_api_key_from_env(self, mock_lg, monkeypatch):
        """API key can come from environment variable."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "env-key")
        backend = BraveSearchBackend(mock_lg)
        assert backend._api_key == "env-key"

    def test_api_key_constructor_overrides_env(self, mock_lg, monkeypatch):
        """Explicit api_key takes precedence over env var."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "env-key")
        backend = BraveSearchBackend(mock_lg, api_key="explicit-key")
        assert backend._api_key == "explicit-key"

    def test_search_success(self, mock_lg):
        """Successful API response returns structured results."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json=_BRAVE_RESPONSE)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("python tutorial", 5)

        assert results is not None
        assert len(results) == 2
        assert results[0]["title"] == "Python Tutorial"
        assert results[0]["url"] == "https://docs.python.org/3/tutorial/"
        assert results[0]["snippet"] == "The official Python tutorial."
        assert results[1]["title"] == "Real Python"

    def test_search_no_results(self, mock_lg):
        """Empty web results returns empty list."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json={"web": {"results": []}})

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("xyzzy", 5)

        assert results == []

    def test_search_no_web_key(self, mock_lg):
        """Response without 'web' key returns empty list."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json={"query": {"original": "test"}})

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results == []

    def test_search_rate_limited(self, mock_lg):
        """429 response returns None (retriable)."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(429)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results is None

    def test_search_server_error(self, mock_lg):
        """5xx responses return None (retriable)."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        for status in (500, 502, 503, 504):
            response = httpx.Response(status)
            with patch.object(backend, "_request", return_value=response):
                assert backend.search("test", 5) is None

    def test_search_client_error(self, mock_lg):
        """4xx (non-429) returns empty list (non-retriable)."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(403)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results == []

    def test_search_timeout(self, mock_lg):
        """Timeout returns None (retriable)."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")

        with patch.object(backend, "_request", side_effect=httpx.TimeoutException("")):
            results = backend.search("test", 5)

        assert results is None

    def test_search_connection_error(self, mock_lg):
        """Connection error returns None (retriable)."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")

        with patch.object(backend, "_request", side_effect=httpx.ConnectError("")):
            results = backend.search("test", 5)

        assert results is None

    def test_search_other_transport_error(self, mock_lg):
        """Non-timeout, non-connect transport errors return None (retriable)."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")

        for exc_cls in (httpx.ReadError, httpx.ProxyError, httpx.ProtocolError):
            with patch.object(backend, "_request", side_effect=exc_cls("")):
                assert backend.search("test", 5) is None

    def test_search_unparseable_json(self, mock_lg):
        """Malformed JSON response returns None instead of crashing."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, text="<html>not json</html>")

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results is None

    def test_search_with_offset(self, mock_lg):
        """Offset parameter is passed to the API request."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json=_BRAVE_RESPONSE)

        with patch.object(backend, "_request", return_value=response) as mock_req:
            backend.search("test", 5, offset=10)

        mock_req.assert_called_once_with("test", 5, 10)

    def test_search_offset_zero_omitted(self, mock_lg):
        """Offset=0 should not add offset param to API request."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")

        with patch.object(
            backend._client, "get", return_value=httpx.Response(200, json=_BRAVE_RESPONSE)
        ) as mock_get:
            backend.search("test", 5, offset=0)

        call_params = mock_get.call_args[1]["params"]
        assert "offset" not in call_params

    def test_search_offset_nonzero_included(self, mock_lg):
        """Offset>0 adds offset param to API request."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")

        with patch.object(
            backend._client, "get", return_value=httpx.Response(200, json=_BRAVE_RESPONSE)
        ) as mock_get:
            backend.search("test", 5, offset=8)

        call_params = mock_get.call_args[1]["params"]
        assert call_params["offset"] == 8

    def test_skips_results_without_title_or_url(self, mock_lg):
        """Results missing title or url should be skipped."""
        backend = BraveSearchBackend(mock_lg, api_key="test-key")
        data = {
            "web": {
                "results": [
                    {"title": "", "url": "https://example.com", "description": "no title"},
                    {"title": "No URL", "url": "", "description": "missing url"},
                    {"title": "Valid", "url": "https://example.com", "description": "ok"},
                ]
            }
        }
        response = httpx.Response(200, json=data)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert len(results) == 1
        assert results[0]["title"] == "Valid"


# ---------------------------------------------------------------------------
# BraveSearchBackend Factory tests
# ---------------------------------------------------------------------------


class TestBraveSearchFactory:
    """Tests for BraveSearchBackend.Factory."""

    def test_create_from_config(self, mock_lg, monkeypatch):
        """Factory.create builds a BraveSearchBackend from DotDict config."""
        from appinfra import DotDict

        from llm_gent.core.tools.builtin.web.brave import Factory

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "factory-key")
        config = DotDict({"timeout": 5.0})
        backend = Factory.create(mock_lg, config)
        assert isinstance(backend, BraveSearchBackend)
        assert backend._timeout == 5.0

    def test_create_with_api_key(self, mock_lg):
        """Factory passes api_key from config to constructor."""
        from appinfra import DotDict

        from llm_gent.core.tools.builtin.web.brave import Factory

        config = DotDict({"api_key": "explicit-key", "timeout": 3.0})
        backend = Factory.create(mock_lg, config)
        assert backend._api_key == "explicit-key"
        assert backend._timeout == 3.0

    def test_factory_is_subclass(self):
        """Factory must be a WebSearchBackendFactory subclass."""
        from llm_gent import WebSearchBackendFactory
        from llm_gent.core.tools.builtin.web.brave import Factory

        assert issubclass(Factory, WebSearchBackendFactory)


# ---------------------------------------------------------------------------
# ToolFactory backend resolution tests
# ---------------------------------------------------------------------------


class TestToolFactoryBackendResolution:
    """Tests for ToolFactory resolving backends from config dicts."""

    def test_resolve_type_brave(self, mock_lg, monkeypatch):
        """type: brave resolves via lazy backend registry."""
        from llm_gent import ToolFactory

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
        factory = ToolFactory(mock_lg)
        tool = factory.create(
            "web_search",
            {
                "backend": {"type": "brave"},
            },
        )
        assert tool is not None
        assert tool.name == "web_search"

    def test_resolve_type_brave_with_config(self, mock_lg):
        """type: brave with nested config passes config to Factory.create."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        tool = factory.create(
            "web_search",
            {
                "backend": {
                    "type": "brave",
                    "config": {"api_key": "from-config", "timeout": 3.0},
                },
            },
        )
        assert tool is not None

    def test_resolve_factory_dynamic_import(self, mock_lg, monkeypatch):
        """factory: dotted.path dynamically imports and calls create()."""
        from llm_gent import ToolFactory

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
        factory = ToolFactory(mock_lg)
        tool = factory.create(
            "web_search",
            {
                "backend": {
                    "factory": "llm_gent.core.tools.builtin.web.brave.Factory",
                },
            },
        )
        assert tool is not None
        assert tool.name == "web_search"

    def test_resolve_unknown_type_raises(self, mock_lg):
        """Unknown type name raises ValueError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(ValueError, match="Unknown web_search backend type"):
            factory.create("web_search", {"backend": {"type": "nonexistent"}})

    def test_resolve_missing_type_and_factory_raises(self, mock_lg):
        """Backend dict without type or factory raises ValueError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(ValueError, match="must include 'type' or 'factory'"):
            factory.create("web_search", {"backend": {}})

    def test_resolve_both_type_and_factory_raises(self, mock_lg):
        """Backend dict with both type and factory raises ValueError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(ValueError, match="not both"):
            factory.create(
                "web_search",
                {"backend": {"type": "brave", "factory": "some.path.Factory"}},
            )

    def test_resolve_non_factory_class_raises(self, mock_lg):
        """Factory path to a non-WebSearchBackendFactory class raises TypeError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(TypeError, match="not a WebSearchBackendFactory subclass"):
            factory.create(
                "web_search",
                {"backend": {"factory": "llm_gent.core.tools.factory.ToolFactory"}},
            )

    def test_resolve_invalid_factory_path_raises(self, mock_lg):
        """Invalid factory dotted path raises ImportError."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        with pytest.raises(ImportError):
            factory.create(
                "web_search",
                {
                    "backend": {"factory": "NoModule"},
                },
            )

    def test_backward_compat_direct_backend(self, mock_lg):
        """Passing a WebSearchBackend instance directly still works."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        backend = MockSearchBackend()
        tool = factory.create("web_search", {"backend": backend})
        assert tool is not None
        assert tool.name == "web_search"

    def test_backward_compat_set_web_search_backend(self, mock_lg):
        """set_web_search_backend() still works for programmatic injection."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        backend = MockSearchBackend()
        factory.set_web_search_backend(backend)
        tool = factory.create("web_search")
        assert tool is not None

    def test_tool_config_passes_through(self, mock_lg, monkeypatch):
        """Tool-level config (retry_delay) passes to WebSearchTool."""
        from llm_gent import ToolFactory

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
        factory = ToolFactory(mock_lg)
        tool = factory.create(
            "web_search",
            {
                "backend": {"type": "brave"},
                "retry_delay": 2.0,
            },
        )
        assert tool is not None
        assert tool._retry_delay == 2.0


# ---------------------------------------------------------------------------
# SerperSearchBackend tests
# ---------------------------------------------------------------------------

_SERPER_RESPONSE = {
    "organic": [
        {
            "title": "Python Tutorial",
            "link": "https://docs.python.org/3/tutorial/",
            "snippet": "The official Python tutorial.",
        },
        {
            "title": "Real Python",
            "link": "https://realpython.com/",
            "snippet": "Learn Python programming.",
        },
    ]
}


class TestSerperSearchBackend:
    """Tests for SerperSearchBackend."""

    def test_protocol_compliance(self, mock_lg):
        from llm_gent import WebSearchBackend

        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        assert isinstance(backend, WebSearchBackend)

    def test_api_key_required(self, mock_lg, monkeypatch):
        """Must raise ValueError if no API key provided."""
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            SerperSearchBackend(mock_lg)

    def test_api_key_from_env(self, mock_lg, monkeypatch):
        """API key can come from environment variable."""
        monkeypatch.setenv("SERPER_API_KEY", "env-key")
        backend = SerperSearchBackend(mock_lg)
        assert backend._api_key == "env-key"

    def test_api_key_constructor_overrides_env(self, mock_lg, monkeypatch):
        """Explicit api_key takes precedence over env var."""
        monkeypatch.setenv("SERPER_API_KEY", "env-key")
        backend = SerperSearchBackend(mock_lg, api_key="explicit-key")
        assert backend._api_key == "explicit-key"

    def test_search_success(self, mock_lg):
        """Successful API response returns structured results."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json=_SERPER_RESPONSE)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("python tutorial", 5)

        assert results is not None
        assert len(results) == 2
        assert results[0]["title"] == "Python Tutorial"
        assert results[0]["url"] == "https://docs.python.org/3/tutorial/"
        assert results[0]["snippet"] == "The official Python tutorial."
        assert results[1]["title"] == "Real Python"

    def test_search_maps_link_to_url(self, mock_lg):
        """Serper 'link' field is mapped to 'url' in results."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        data = {"organic": [{"title": "Test", "link": "https://example.com", "snippet": "s"}]}
        response = httpx.Response(200, json=data)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results[0]["url"] == "https://example.com"
        assert "link" not in results[0]

    def test_search_no_results(self, mock_lg):
        """Empty organic results returns empty list."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json={"organic": []})

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("xyzzy", 5)

        assert results == []

    def test_search_no_organic_key(self, mock_lg):
        """Response without 'organic' key returns empty list."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json={"searchParameters": {"q": "test"}})

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results == []

    def test_search_rate_limited(self, mock_lg):
        """429 response returns None (retriable)."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(429)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results is None

    def test_search_server_error(self, mock_lg):
        """5xx responses return None (retriable)."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        for status in (500, 502, 503, 504):
            response = httpx.Response(status)
            with patch.object(backend, "_request", return_value=response):
                assert backend.search("test", 5) is None

    def test_search_client_error(self, mock_lg):
        """4xx (non-429) returns empty list (non-retriable)."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(403)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results == []

    def test_search_timeout(self, mock_lg):
        """Timeout returns None (retriable)."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(backend, "_request", side_effect=httpx.TimeoutException("")):
            results = backend.search("test", 5)

        assert results is None

    def test_search_connection_error(self, mock_lg):
        """Connection error returns None (retriable)."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(backend, "_request", side_effect=httpx.ConnectError("")):
            results = backend.search("test", 5)

        assert results is None

    def test_search_unparseable_json(self, mock_lg):
        """Malformed JSON response returns None instead of crashing."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, text="<html>not json</html>")

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert results is None

    def test_search_pagination_via_page_param(self, mock_lg):
        """Offset > 0 is mapped to Serper's page parameter."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        response = httpx.Response(200, json=_SERPER_RESPONSE)

        with patch.object(backend, "_request", return_value=response) as mock_req:
            results = backend.search("test", 10, offset=20)

        assert results is not None
        # _request is called with offset, which maps to page=3 (20 // 10 + 1)
        mock_req.assert_called_once_with("test", 10, 20)

    def test_search_pagination_page_calculation(self, mock_lg):
        """Page parameter is correctly calculated from offset and max_results."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(
            backend._client, "post", return_value=httpx.Response(200, json=_SERPER_RESPONSE)
        ) as mock_post:
            backend.search("test", 10, offset=30)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"] == {"q": "test", "num": 10, "page": 4}

    def test_search_no_page_param_when_offset_zero(self, mock_lg):
        """Page parameter is omitted when offset is 0."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(
            backend._client, "post", return_value=httpx.Response(200, json=_SERPER_RESPONSE)
        ) as mock_post:
            backend.search("test", 10, offset=0)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"] == {"q": "test", "num": 10}
        assert "page" not in call_kwargs["json"]

    def test_search_max_results_zero_returns_empty(self, mock_lg):
        """max_results <= 0 returns empty list without making request."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(backend._client, "post") as mock_post:
            result = backend.search("test", 0, offset=10)

        assert result == []
        mock_post.assert_not_called()

    def test_search_non_aligned_offset_logs_warning(self, mock_lg):
        """Non-page-aligned offset logs warning with effective offset."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(
            backend._client, "post", return_value=httpx.Response(200, json=_SERPER_RESPONSE)
        ):
            backend.search("test", 10, offset=15)

        mock_lg.warning.assert_called_once()
        call_args = mock_lg.warning.call_args
        assert "rounds offset to page boundary" in call_args[0][0]
        assert call_args[1]["extra"]["requested_offset"] == 15
        assert call_args[1]["extra"]["effective_offset"] == 10

    def test_search_aligned_offset_no_warning(self, mock_lg):
        """Page-aligned offset does not log warning."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(
            backend._client, "post", return_value=httpx.Response(200, json=_SERPER_RESPONSE)
        ):
            backend.search("test", 10, offset=20)

        mock_lg.warning.assert_not_called()

    def test_search_uses_post(self, mock_lg):
        """Serper API uses POST, not GET."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")

        with patch.object(
            backend._client, "post", return_value=httpx.Response(200, json=_SERPER_RESPONSE)
        ) as mock_post:
            backend.search("test", 5)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"] == {"q": "test", "num": 5}
        assert call_kwargs["headers"]["X-API-KEY"] == "test-key"

    def test_skips_results_without_title_or_link(self, mock_lg):
        """Results missing title or link should be skipped."""
        backend = SerperSearchBackend(mock_lg, api_key="test-key")
        data = {
            "organic": [
                {"title": "", "link": "https://example.com", "snippet": "no title"},
                {"title": "No Link", "link": "", "snippet": "missing link"},
                {"title": "Valid", "link": "https://example.com", "snippet": "ok"},
            ]
        }
        response = httpx.Response(200, json=data)

        with patch.object(backend, "_request", return_value=response):
            results = backend.search("test", 5)

        assert len(results) == 1
        assert results[0]["title"] == "Valid"


# ---------------------------------------------------------------------------
# SerperSearchBackend Factory tests
# ---------------------------------------------------------------------------


class TestSerperSearchFactory:
    """Tests for SerperSearchBackend.Factory."""

    def test_create_from_config(self, mock_lg, monkeypatch):
        """Factory.create builds a SerperSearchBackend from DotDict config."""
        from appinfra import DotDict

        from llm_gent.core.tools.builtin.web.serper import Factory

        monkeypatch.setenv("SERPER_API_KEY", "factory-key")
        config = DotDict({"timeout": 5.0})
        backend = Factory.create(mock_lg, config)
        assert isinstance(backend, SerperSearchBackend)
        assert backend._timeout == 5.0

    def test_create_with_api_key(self, mock_lg):
        """Factory passes api_key from config to constructor."""
        from appinfra import DotDict

        from llm_gent.core.tools.builtin.web.serper import Factory

        config = DotDict({"api_key": "explicit-key", "timeout": 3.0})
        backend = Factory.create(mock_lg, config)
        assert backend._api_key == "explicit-key"
        assert backend._timeout == 3.0

    def test_factory_is_subclass(self):
        """Factory must be a WebSearchBackendFactory subclass."""
        from llm_gent import WebSearchBackendFactory
        from llm_gent.core.tools.builtin.web.serper import Factory

        assert issubclass(Factory, WebSearchBackendFactory)

    def test_create_default_per_minute(self, mock_lg):
        """Default per_minute is 60.0 (much higher than Brave's 3.0)."""
        from appinfra import DotDict

        from llm_gent.core.tools.builtin.web.serper import Factory

        config = DotDict({"api_key": "test-key"})
        backend = Factory.create(mock_lg, config)
        assert backend._rate_limiter.per_minute == 60.0


# ---------------------------------------------------------------------------
# ToolFactory Serper backend resolution tests
# ---------------------------------------------------------------------------


class TestToolFactorySerperResolution:
    """Tests for ToolFactory resolving Serper backend from config."""

    def test_resolve_type_serper(self, mock_lg, monkeypatch):
        """type: serper resolves via lazy backend registry."""
        from llm_gent import ToolFactory

        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        factory = ToolFactory(mock_lg)
        tool = factory.create(
            "web_search",
            {"backend": {"type": "serper"}},
        )
        assert tool is not None
        assert tool.name == "web_search"

    def test_resolve_type_serper_with_config(self, mock_lg):
        """type: serper with nested config passes config to Factory.create."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        tool = factory.create(
            "web_search",
            {
                "backend": {
                    "type": "serper",
                    "config": {"api_key": "from-config", "timeout": 3.0},
                },
            },
        )
        assert tool is not None

    def test_resolve_factory_dynamic_import(self, mock_lg, monkeypatch):
        """factory: dotted.path dynamically imports Serper factory."""
        from llm_gent import ToolFactory

        monkeypatch.setenv("SERPER_API_KEY", "test-key")
        factory = ToolFactory(mock_lg)
        tool = factory.create(
            "web_search",
            {
                "backend": {
                    "factory": "llm_gent.core.tools.builtin.web.serper.Factory",
                },
            },
        )
        assert tool is not None
        assert tool.name == "web_search"
