"""Tests for WebFetchTool and WebSearchTool."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent import ToolResult, WebFetchTool, WebSearchTool


pytestmark = pytest.mark.unit


@pytest.fixture()
def mock_lg():
    """Create mock Logger."""
    return MagicMock()


@pytest.fixture()
def web_fetch(mock_lg):
    """Create WebFetchTool for WebSearchTool tests."""
    return WebFetchTool(mock_lg)


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
        html_content = (
            "<html><head><title>Test</title></head>"
            "<body><nav>Menu</nav>"
            "<article><p>Hello World</p><p>Main content here.</p></article>"
            "<script>bad();</script>"
            "<footer>Copyright</footer></body></html>"
        )

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = self._mock_http_result(html_content)
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert "Hello World" in result.output
        assert "<p>" not in result.output
        assert "bad()" not in result.output

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
        assert len(result.output) < len(long_text)
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
        # Minimal HTML that trafilatura can't extract content from
        html = "<html><body></body></html>"

        with patch.object(tool._http, "execute") as mock_fetch:
            mock_fetch.return_value = self._mock_http_result(html)
            result = tool.execute(url="https://example.com")

        assert result.success is True
        # Should fall back to raw content
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

# Sample DDG HTML with two results
_SAMPLE_DDG_HTML = """
<div class="result results_links results_links_deep web-result">
    <div class="links_main links_deep result__body">
        <a class="result__a" href="https://example.com/page1">
            <b>First</b> Result Title
        </a>
        <a class="result__snippet" href="https://example.com/page1">
            This is the first snippet.
        </a>
    </div>
</div>
<div class="result results_links results_links_deep web-result">
    <div class="links_main links_deep result__body">
        <a class="result__a" href="https://example.com/page2">
            Second Result
        </a>
        <a class="result__snippet" href="https://example.com/page2">
            Second snippet here.
        </a>
    </div>
</div>
"""


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_tool_properties(self, mock_lg, web_fetch):
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch)
        assert tool.name == "web_search"
        assert "search" in tool.description.lower()
        assert tool.parameters["required"] == ["query"]

    def test_protocol_compliance(self, mock_lg, web_fetch):
        from llm_gent import Tool

        tool = WebSearchTool(mock_lg, web_fetch=web_fetch)
        assert isinstance(tool, Tool)

    def test_search_success(self, mock_lg, web_fetch):
        """Search returns structured results."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=_SAMPLE_DDG_HTML)
            result = tool.execute(query="test search")

        assert result.success is True
        assert "First Result Title" in result.output
        assert "Second Result" in result.output
        assert "https://example.com/page1" in result.output
        assert "first snippet" in result.output

    def test_search_max_results(self, mock_lg, web_fetch):
        """max_results limits the number of results."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=_SAMPLE_DDG_HTML)
            result = tool.execute(query="test", max_results=1)

        assert result.success is True
        assert "First Result Title" in result.output
        assert "Second Result" not in result.output

    def test_search_no_results(self, mock_lg, web_fetch):
        """Empty search results."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)
        empty_html = "<html><body>No results</body></html>"

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=empty_html)
            result = tool.execute(query="xyzzy nonexistent")

        assert result.success is True
        assert "No results found" in result.output

    def test_search_fetch_error(self, mock_lg, web_fetch):
        """Search propagates fetch errors."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=False, output="", error="Timeout")
            result = tool.execute(query="test")

        assert result.success is False
        assert "Timeout" in result.error

    def test_search_missing_query(self, mock_lg, web_fetch):
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch)
        result = tool.execute()
        assert result.success is False
        assert "query" in result.error.lower()

    def test_search_empty_query(self, mock_lg, web_fetch):
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch)
        result = tool.execute(query="   ")
        assert result.success is False
        assert "query" in result.error.lower()

    def test_search_max_results_none(self, mock_lg, web_fetch):
        """max_results=None (LLM sends null) should default to 5, not crash."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=_SAMPLE_DDG_HTML)
            result = tool.execute(query="test", max_results=None)

        assert result.success is True

    def test_rate_limiting(self, mock_lg, web_fetch):
        """Rate limiter blocks after exceeding limit."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=2)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=_SAMPLE_DDG_HTML)

            # First two should succeed
            r1 = tool.execute(query="query1")
            r2 = tool.execute(query="query2")
            assert r1.success is True
            assert r2.success is True

            # Third should be rate limited
            r3 = tool.execute(query="query3")
            assert r3.success is False
            assert "Rate limited" in r3.error

    def test_rate_limiting_disabled(self, mock_lg, web_fetch):
        """Rate limiting can be disabled with 0."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=_SAMPLE_DDG_HTML)

            for _ in range(10):
                result = tool.execute(query="query")
                assert result.success is True

    def test_max_results_clamped(self, mock_lg, web_fetch):
        """max_results is clamped to 1-8 range."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=_SAMPLE_DDG_HTML)

            # max_results > 8 should be clamped
            result = tool.execute(query="test", max_results=100)
            assert result.success is True

            # max_results < 1 should be clamped to 1
            result = tool.execute(query="test", max_results=0)
            assert result.success is True

    def test_shared_web_fetch(self, mock_lg, web_fetch):
        """WebSearchTool can share a WebFetchTool instance."""
        web_fetch = WebFetchTool(mock_lg, allowed_domains=["duckduckgo.com"])
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch)
        assert tool._web_fetch is web_fetch

    def test_to_openai_function(self, mock_lg, web_fetch):
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch)
        func = tool.to_openai_function()
        assert func["function"]["name"] == "web_search"
        assert "query" in func["function"]["parameters"]["properties"]

    def test_search_url_encoding(self, mock_lg, web_fetch):
        """Query is URL-encoded in the request."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=_SAMPLE_DDG_HTML)
            tool.execute(query="hello world & more")

            call_url = mock_fetch.call_args.kwargs["url"]
            assert "hello+world" in call_url
            assert " " not in call_url

    def test_parse_warning_on_large_empty_response(self, mock_lg, web_fetch):
        """Large DDG response with 0 parsed results should log a warning."""
        tool = WebSearchTool(mock_lg, web_fetch=web_fetch, max_queries_per_minute=0)
        large_html = "<html><body>" + "x" * 10_000 + "</body></html>"

        with patch.object(tool._web_fetch, "fetch_raw") as mock_fetch:
            mock_fetch.return_value = ToolResult(success=True, output=large_html)
            result = tool.execute(query="test")

        assert result.success is True
        assert "No results found" in result.output
        mock_lg.warning.assert_called_once()


# ---------------------------------------------------------------------------
# URL unwrapping tests
# ---------------------------------------------------------------------------


class TestDdgUrlUnwrap:
    """Tests for DDG redirect URL unwrapping."""

    def test_unwraps_ddg_redirect(self):
        from llm_gent.core.tools.builtin.web_search import _unwrap_ddg_url

        raw = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc123"
        assert _unwrap_ddg_url(raw) == "https://example.com/page"

    def test_passthrough_direct_url(self):
        from llm_gent.core.tools.builtin.web_search import _unwrap_ddg_url

        assert _unwrap_ddg_url("https://example.com/page") == "https://example.com/page"

    def test_decodes_html_entities(self):
        from llm_gent.core.tools.builtin.web_search import _unwrap_ddg_url

        raw = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F%3Fa%3D1%26b%3D2&amp;rut=x"
        result = _unwrap_ddg_url(raw)
        assert result == "https://example.com/?a=1&b=2"


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

    def test_factory_creates_web_search(self, mock_lg):
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        tool = factory.create("web_search")
        assert isinstance(tool, WebSearchTool)
        assert tool.name == "web_search"

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
        tool = factory.create("web_search", {"max_queries_per_minute": 10})
        assert isinstance(tool, WebSearchTool)
        assert tool._rate_limit == 10

    def test_factory_web_search_inherits_web_fetch_config(self, mock_lg):
        """web_search should use the same WebFetchTool instance as web_fetch."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        # Create web_fetch with restricted domains first
        fetch_tool = factory.create("web_fetch", {"allowed_domains": ["docs.python.org"]})
        # web_search should reuse that restricted instance
        search_tool = factory.create("web_search")
        assert search_tool._web_fetch is fetch_tool

    def test_factory_lazy_web_fetch(self, mock_lg):
        """WebFetchTool is not created until needed (lazy init)."""
        from llm_gent import ToolFactory

        factory = ToolFactory(mock_lg)
        assert factory._web_fetch is None
        # Creating a non-web tool should not trigger WebFetchTool creation
        factory.create("shell")
        assert factory._web_fetch is None
        # Creating web_search triggers lazy creation
        factory.create("web_search")
        assert factory._web_fetch is not None
