# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for tool use infrastructure."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_gent import (
    BaseTool,
    FetchResult,
    HTTPFetchTool,
    Registry,
    ShellTool,
    Tool,
    ToolCall,
    ToolResult,
)


pytestmark = pytest.mark.unit


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        result = ToolResult(success=True, output="hello")

        assert result.success is True
        assert result.output == "hello"
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, output="", error="something went wrong")

        assert result.success is False
        assert result.output == ""
        assert result.error == "something went wrong"


class TestToolCall:
    """Tests for ToolCall."""

    def test_tool_call(self):
        call = ToolCall(id="call-1", name="shell", arguments={"command": "ls"})

        assert call.id == "call-1"
        assert call.name == "shell"
        assert call.arguments == {"command": "ls"}

    def test_tool_call_empty_arguments(self):
        call = ToolCall(id="call-1", name="noop")

        assert call.arguments == {}


class TestBaseTool:
    """Tests for BaseTool."""

    def test_protocol_compliance(self):
        class MyTool(BaseTool):
            name = "my_tool"
            description = "A test tool"
            parameters = {"type": "object", "properties": {}}

            def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")

        tool = MyTool()
        assert isinstance(tool, Tool)

    def test_to_openai_function(self):
        class MyTool(BaseTool):
            name = "my_tool"
            description = "A test tool"
            parameters = {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            }

            def execute(self, arg: str) -> ToolResult:
                return ToolResult(success=True, output=arg)

        tool = MyTool()
        func = tool.to_openai_function()

        assert func == {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                },
            },
        }


class TestRegistry:
    """Tests for Registry."""

    def test_register_and_get(self):
        registry = Registry()
        tool = ShellTool()

        registry.register(tool)

        assert registry.get("shell") is tool
        assert "shell" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self):
        registry = Registry()
        registry.register(ShellTool())

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ShellTool())

    def test_unregister(self):
        registry = Registry()
        registry.register(ShellTool())

        registry.unregister("shell")

        assert registry.get("shell") is None
        assert len(registry) == 0

    def test_unregister_unknown_raises(self):
        registry = Registry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("unknown")

    def test_list_names(self):
        registry = Registry()
        registry.register(ShellTool())

        assert registry.list_names() == ["shell"]

    def test_list_tools(self):
        registry = Registry()
        tool = ShellTool()
        registry.register(tool)

        assert registry.list_tools() == [tool]

    def test_to_openai_tools(self):
        registry = Registry()
        registry.register(ShellTool())

        tools = registry.to_openai_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "shell"


class TestHTTPFetchTool:
    """Tests for HTTPFetchTool."""

    def _mock_response(
        self,
        status_code: int = 200,
        text: str = "response body",
        headers: dict | None = None,
        reason_phrase: str = "OK",
    ) -> MagicMock:
        """Create a mock httpx.Response."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        response.text = text
        response.headers = headers or {}
        response.reason_phrase = reason_phrase
        response.is_success = 200 <= status_code < 300
        return response

    def _mock_dns_public(self, ip: str = "1.2.3.4"):
        """Create a mock for socket.getaddrinfo returning a public IP."""
        import socket

        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    def _mock_dns_private(self, ip: str = "192.168.1.1"):
        """Create a mock for socket.getaddrinfo returning a private IP."""
        import socket

        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    def test_fetch_success(self):
        """Fetch URL successfully."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(text="Hello, World!")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert "Hello, World!" in result.output

    def test_fetch_with_headers(self):
        """Fetch with custom headers."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(text="authenticated response")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.get.return_value = mock_response

            result = tool.execute(
                url="https://api.example.com/data",
                headers={"Authorization": "Bearer token123"},
            )

            # Verify headers were passed
            call_args = mock_instance.get.call_args
            assert "Authorization" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Authorization"] == "Bearer token123"

        assert result.success is True

    def test_fetch_with_default_headers(self):
        """Default headers are included in requests."""
        tool = HTTPFetchTool(default_headers={"User-Agent": "TestBot/1.0"})
        mock_response = self._mock_response()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.get.return_value = mock_response

            tool.execute(url="https://example.com")

            call_args = mock_instance.get.call_args
            assert call_args.kwargs["headers"]["User-Agent"] == "TestBot/1.0"

    def test_fetch_http_error(self):
        """Handle HTTP error responses."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(
            status_code=404,
            text="Not Found",
            reason_phrase="Not Found",
        )

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com/missing")

        assert result.success is False
        assert "404" in result.error
        assert "Not Found" in result.output

    def test_fetch_timeout(self):
        """Handle request timeout."""
        tool = HTTPFetchTool(timeout=5.0)

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.side_effect = (
                httpx.TimeoutException("timed out")
            )
            result = tool.execute(url="https://slow.example.com")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_fetch_connection_error(self):
        """Handle connection errors."""
        tool = HTTPFetchTool()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            result = tool.execute(url="https://unreachable.example.com")

        assert result.success is False
        assert "connection" in result.error.lower()

    def test_fetch_response_too_large_header(self):
        """Reject responses that exceed max size (from Content-Length header)."""
        tool = HTTPFetchTool(max_response_size=1000)
        mock_response = self._mock_response(
            text="x" * 100,
            headers={"content-length": "2000"},
        )

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com/large")

        assert result.success is False
        assert "too large" in result.error.lower()

    def test_fetch_invalid_content_length_header(self):
        """Handle malformed Content-Length header gracefully."""
        tool = HTTPFetchTool(max_response_size=1000)
        mock_response = self._mock_response(
            text="valid response",
            headers={"content-length": "not-a-number"},
        )

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        # Should succeed - invalid Content-Length is ignored, response is read normally
        assert result.success is True
        assert "valid response" in result.output

    def test_fetch_response_truncated(self):
        """Truncate responses that exceed max size."""
        tool = HTTPFetchTool(max_response_size=100)
        mock_response = self._mock_response(text="x" * 200)

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        assert result.success is True
        assert len(result.output) < 200
        assert "truncated" in result.output.lower()

    def test_allowed_domains_success(self):
        """Allow requests to allowed domains."""
        tool = HTTPFetchTool(allowed_domains=["api.github.com"])
        mock_response = self._mock_response(text="github data")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://api.github.com/repos")

        assert result.success is True

    def test_allowed_domains_blocked(self):
        """Block requests to non-allowed domains."""
        tool = HTTPFetchTool(allowed_domains=["api.github.com"])

        result = tool.execute(url="https://evil.com/steal-data")

        assert result.success is False
        assert "not in allowed" in result.error.lower()

    def test_allowed_domains_subdomain(self):
        """Subdomains of allowed domains are permitted."""
        tool = HTTPFetchTool(allowed_domains=["github.com"])
        mock_response = self._mock_response()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://api.github.com/repos")

        assert result.success is True

    def test_blocked_domains(self):
        """Block requests to blocked domains."""
        tool = HTTPFetchTool(blocked_domains=["evil.com", "malware.org"])

        result = tool.execute(url="https://evil.com/bad")

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocked_domains_subdomain(self):
        """Subdomains of blocked domains are also blocked."""
        tool = HTTPFetchTool(blocked_domains=["evil.com"])

        result = tool.execute(url="https://api.evil.com/endpoint")

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocked_domains_allowed(self):
        """Non-blocked domains are allowed."""
        tool = HTTPFetchTool(blocked_domains=["evil.com"])
        mock_response = self._mock_response()

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://good.com/api")

        assert result.success is True

    def test_invalid_url_scheme(self):
        """Reject non-HTTP schemes."""
        tool = HTTPFetchTool()

        result = tool.execute(url="ftp://files.example.com/data.txt")

        assert result.success is False
        assert "scheme" in result.error.lower()

    def test_invalid_url_missing_domain(self):
        """Reject URLs without domain."""
        tool = HTTPFetchTool()

        result = tool.execute(url="https:///path/only")

        assert result.success is False
        assert "domain" in result.error.lower() or "invalid" in result.error.lower()

    def test_missing_url_argument(self):
        """Fail when URL argument is missing."""
        tool = HTTPFetchTool()

        result = tool.execute()
        assert result.success is False
        assert "missing" in result.error.lower()

        result = tool.execute(url="")
        assert result.success is False

    def test_invalid_url_type(self):
        """Fail when URL is not a string."""
        tool = HTTPFetchTool()

        result = tool.execute(url=123)
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_invalid_headers_type(self):
        """Fail when headers is not a dict."""
        tool = HTTPFetchTool()

        result = tool.execute(url="https://example.com", headers="not-a-dict")
        assert result.success is False
        assert "headers" in result.error.lower()

    def test_invalid_header_values(self):
        """Fail when header values are not strings."""
        tool = HTTPFetchTool()

        result = tool.execute(url="https://example.com", headers={"count": 123})
        assert result.success is False
        assert "string" in result.error.lower()

    def test_tool_properties(self):
        """Tool has correct properties."""
        tool = HTTPFetchTool()

        assert tool.name == "http_fetch"
        assert "url" in tool.description.lower() or "http" in tool.description.lower()
        assert tool.parameters["type"] == "object"
        assert "url" in tool.parameters["properties"]
        assert "headers" in tool.parameters["properties"]

    def test_to_openai_function(self):
        """Converts to OpenAI function format."""
        tool = HTTPFetchTool()
        func = tool.to_openai_function()

        assert func["type"] == "function"
        assert func["function"]["name"] == "http_fetch"
        assert "url" in func["function"]["parameters"]["properties"]

    def test_domain_with_port(self):
        """Handle domains with port numbers."""
        # Use block_private_ips=False since this test is about port handling, not SSRF
        tool = HTTPFetchTool(allowed_domains=["localhost"], block_private_ips=False)
        mock_response = self._mock_response()

        with patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="http://localhost:8080/api")

        assert result.success is True

    # SSRF Protection Tests

    def test_ssrf_blocks_localhost(self):
        """Block requests to localhost (loopback)."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("127.0.0.1")
            result = tool.execute(url="http://localhost/admin")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_private_ip_10(self):
        """Block requests to 10.x.x.x private network."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("10.0.0.1")
            result = tool.execute(url="http://internal-server/api")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_private_ip_172(self):
        """Block requests to 172.16.x.x private network."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("172.16.0.1")
            result = tool.execute(url="http://internal-server/api")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_private_ip_192(self):
        """Block requests to 192.168.x.x private network."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("192.168.1.1")
            result = tool.execute(url="http://router.local/admin")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_link_local(self):
        """Block requests to link-local addresses (169.254.x.x) including cloud metadata."""
        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = self._mock_dns_private("169.254.169.254")
            result = tool.execute(url="http://169.254.169.254/latest/meta-data/")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_blocks_ipv6_loopback(self):
        """Block requests to IPv6 loopback (::1)."""
        import socket

        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]
            result = tool.execute(url="http://localhost/admin")

        assert result.success is False
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    def test_ssrf_allows_public_ip(self):
        """Allow requests to public IP addresses."""
        tool = HTTPFetchTool()
        mock_response = self._mock_response(text="public content")

        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns,
            patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client,
        ):
            mock_dns.return_value = self._mock_dns_public()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="https://example.com")

        assert result.success is True

    def test_ssrf_protection_disabled(self):
        """Allow private IPs when block_private_ips=False."""
        tool = HTTPFetchTool(block_private_ips=False)
        mock_response = self._mock_response(text="internal content")

        with patch("llm_gent.core.tools.builtin.http.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = tool.execute(url="http://localhost/internal")

        assert result.success is True

    def test_ssrf_dns_resolution_failure(self):
        """Handle DNS resolution failures gracefully."""
        import socket

        tool = HTTPFetchTool()

        with patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror(8, "Name does not resolve")
            result = tool.execute(url="http://nonexistent.invalid/api")

        assert result.success is False
        assert "dns" in result.error.lower() or "resolution" in result.error.lower()


_DNS = {
    "site.example": "93.184.216.10",
    "other.example": "93.184.216.20",
    "www.example": "93.184.216.30",
    "intranet.example": "10.0.0.5",
}


def _resolve(host: str, *_: object) -> list[tuple[object, ...]]:
    """socket.getaddrinfo over the _DNS table."""
    import socket

    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_DNS[host], 0))]


class _Site:
    """Stands in for httpx.Client: answers GETs from a URL -> (status, headers, body)
    table and records every request it was sent, with the Cookie header the
    client's jar would add."""

    def __init__(self, routes: dict[str, tuple[int, dict[str, str], str]]) -> None:
        self.routes = routes
        self.sent: list[httpx.Request] = []

    def client(self, **kwargs: object) -> MagicMock:
        cookies = kwargs.get("cookies")
        client = MagicMock()
        client.__enter__.return_value.get.side_effect = lambda url, headers: self._get(
            url, headers, cookies
        )
        return client

    def _get(self, url: str, headers: dict[str, str], cookies: object) -> httpx.Response:
        request = httpx.Request("GET", url, headers=headers)
        if isinstance(cookies, httpx.Cookies):
            cookies.set_cookie_header(request)
        self.sent.append(request)
        status, response_headers, body = self.routes[url]
        return httpx.Response(status, headers=response_headers, text=body, request=request)


class TestHTTPFetchRedirects:
    """HTTPFetchTool follows redirects; every hop is checked like the requested URL."""

    def _fetch(
        self, site: _Site, url: str, tool: HTTPFetchTool | None = None, **kwargs: object
    ) -> ToolResult:
        with (
            patch("llm_gent.core.tools.builtin.http.socket.getaddrinfo", side_effect=_resolve),
            patch("llm_gent.core.tools.builtin.http.httpx.Client", side_effect=site.client),
        ):
            return (tool or HTTPFetchTool()).execute(url=url, **kwargs)

    def test_a_303_to_a_public_page_returns_that_page_and_its_url(self):
        site = _Site(
            {
                "https://site.example/a": (303, {"location": "https://www.example/a"}, ""),
                "https://www.example/a": (200, {}, "the page"),
            }
        )
        result = self._fetch(site, "https://site.example/a")
        assert isinstance(result, FetchResult)
        assert (result.success, result.output, result.url) == (
            True,
            "the page",
            "https://www.example/a",
        )

    def test_a_response_without_redirects_carries_the_requested_url(self):
        site = _Site({"https://site.example/a": (200, {}, "body")})
        result = self._fetch(site, "https://site.example/a")
        assert isinstance(result, FetchResult)
        assert result.url == "https://site.example/a"

    def test_a_cookie_set_by_a_redirect_is_sent_on_the_next_hop(self):
        """A cookie round trip: the redirect sets a cookie and points back."""
        site = _Site(
            {
                "https://site.example/a": (
                    303,
                    {"location": "/a?check", "set-cookie": "session=s1; Path=/"},
                    "",
                ),
                "https://site.example/a?check": (200, {}, "the page"),
            }
        )
        result = self._fetch(site, "https://site.example/a")
        assert result.output == "the page"
        assert site.sent[1].headers.get("cookie") == "session=s1"

    def test_a_relative_location_resolves_against_the_current_url(self):
        site = _Site(
            {
                "https://site.example/x/a": (302, {"location": "../b"}, ""),
                "https://site.example/b": (200, {}, "b"),
            }
        )
        assert self._fetch(site, "https://site.example/x/a").output == "b"

    def test_a_redirect_to_a_private_ip_is_refused_before_any_request(self):
        site = _Site(
            {"https://site.example/a": (302, {"location": "http://intranet.example/page"}, "")}
        )
        result = self._fetch(site, "https://site.example/a")
        assert result.success is False
        assert "http://intranet.example/page" in result.error
        assert "private" in result.error
        assert [str(r.url) for r in site.sent] == ["https://site.example/a"]

    def test_a_redirect_to_a_blocked_domain_is_refused(self):
        site = _Site({"https://site.example/a": (302, {"location": "https://www.example/"}, "")})
        tool = HTTPFetchTool(blocked_domains=["www.example"])
        result = self._fetch(site, "https://site.example/a", tool)
        assert result.success is False
        assert "blocked" in result.error
        assert len(site.sent) == 1

    def test_a_redirect_outside_the_allowed_domains_is_refused(self):
        site = _Site({"https://site.example/a": (302, {"location": "https://www.example/"}, "")})
        tool = HTTPFetchTool(allowed_domains=["site.example"])
        result = self._fetch(site, "https://site.example/a", tool)
        assert result.success is False
        assert "not in allowed list" in result.error
        assert len(site.sent) == 1

    def test_a_redirect_to_a_non_http_scheme_is_refused(self):
        site = _Site({"https://site.example/a": (302, {"location": "ftp://files.example/a"}, "")})
        result = self._fetch(site, "https://site.example/a")
        assert result.success is False
        assert "scheme" in result.error
        assert len(site.sent) == 1

    def test_more_redirects_than_the_limit_is_an_error(self):
        site = _Site(
            {f"https://site.example/{i}": (302, {"location": f"/{i + 1}"}, "") for i in range(4)}
        )
        result = self._fetch(site, "https://site.example/0", HTTPFetchTool(max_redirects=2))
        assert result.success is False
        assert result.error == "Too many redirects (more than 2)"
        assert len(site.sent) == 3

    def test_max_redirects_zero_returns_the_redirect_as_the_response(self):
        site = _Site({"https://site.example/a": (303, {"location": "/b"}, "see other")})
        result = self._fetch(site, "https://site.example/a", HTTPFetchTool(max_redirects=0))
        assert isinstance(result, FetchResult)
        assert result.success is False
        assert result.error == "HTTP 303: See Other"
        assert result.url == "https://site.example/a"

    def test_a_negative_max_redirects_is_rejected(self):
        with pytest.raises(ValueError, match="max_redirects"):
            HTTPFetchTool(max_redirects=-1)

    def test_a_redirect_without_location_is_the_response(self):
        site = _Site({"https://site.example/a": (302, {}, "nowhere")})
        result = self._fetch(site, "https://site.example/a")
        assert result.success is False
        assert result.error == "HTTP 302: Found"
        assert len(site.sent) == 1

    def test_credentials_stay_with_their_origin(self):
        """Same-origin hops keep Authorization; a hop to another host drops it."""
        site = _Site(
            {
                "https://site.example/a": (302, {"location": "/b"}, ""),
                "https://site.example/b": (302, {"location": "https://other.example/c"}, ""),
                "https://other.example/c": (200, {}, "done"),
            }
        )
        tool = HTTPFetchTool(default_headers={"User-Agent": "TestBot/1.0"})
        result = self._fetch(
            site, "https://site.example/a", tool, headers={"Authorization": "Bearer t"}
        )
        assert result.output == "done"
        auth = [r.headers.get("authorization") for r in site.sent]
        assert auth == ["Bearer t", "Bearer t", None]
        assert site.sent[2].headers["user-agent"] == "TestBot/1.0"


# Note: Tool execution loop tests have been moved to llm-saia.
# SAIA now handles the LLM ↔ tool interaction loop.


class TestRememberTool:
    """Tests for RememberTool."""

    def test_remember_success(self):
        """Store a fact successfully."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.remember.return_value = 42

        tool = RememberTool(mock_memory_trait)
        result = tool.execute(fact="User prefers Python")

        assert result.success is True
        assert "42" in result.output
        assert "User prefers Python" in result.output
        mock_memory_trait.remember.assert_called_once_with(
            fact="User prefers Python",
            category="general",
            source="inferred",
        )

    def test_remember_with_category(self):
        """Store a fact with category."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.remember.return_value = 123

        tool = RememberTool(mock_memory_trait)
        result = tool.execute(fact="Uses vim", category="preferences")

        assert result.success is True
        mock_memory_trait.remember.assert_called_once_with(
            fact="Uses vim",
            category="preferences",
            source="inferred",
        )

    def test_remember_failure(self):
        """Handle store failure gracefully."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.remember.side_effect = RuntimeError("Database error")

        tool = RememberTool(mock_memory_trait)
        result = tool.execute(fact="Some fact")

        assert result.success is False
        assert "Database error" in result.error

    def test_tool_properties(self):
        """Tool has correct properties."""
        from llm_gent.core.tools.builtin.memory import RememberTool

        mock_memory_trait = MagicMock()
        tool = RememberTool(mock_memory_trait)

        assert tool.name == "remember"
        assert "fact" in tool.parameters["properties"]
        assert "category" in tool.parameters["properties"]
        assert "fact" in tool.parameters["required"]


class TestRecallTool:
    """Tests for RecallTool."""

    def test_recall_with_embedder(self):
        """Search facts using semantic search."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_fact = MagicMock()
        mock_fact.content = "User prefers Python"
        mock_fact.category = "preferences"

        mock_scored_entity = MagicMock()
        mock_scored_entity.entity = mock_fact
        mock_scored_entity.score = 0.85

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.return_value = [mock_scored_entity]

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="programming language")

        assert result.success is True
        assert "User prefers Python" in result.output
        assert "0.85" in result.output
        mock_memory_trait.recall.assert_called_once()

    def test_recall_without_embedder(self):
        """Fall back to listing facts without semantic search."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_fact = MagicMock()
        mock_fact.content = "Uses vim"
        mock_fact.category = "tools"

        # Build mock chain avoiding 'assertions' conflict with MagicMock
        mock_assertions = MagicMock()
        mock_assertions.list.return_value = [mock_fact]
        mock_atomic = MagicMock()
        mock_atomic.assertions = mock_assertions
        mock_learn = MagicMock()
        mock_learn.atomic = mock_atomic

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = False
        mock_memory_trait.kelt = mock_learn

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="editor")

        assert result.success is True
        assert "Uses vim" in result.output

    def test_recall_no_results(self):
        """Handle no matching facts."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.return_value = []

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="nonexistent topic")

        assert result.success is True
        assert "No matching facts" in result.output

    def test_recall_with_category_filter(self):
        """Filter recall by category."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.return_value = []

        tool = RecallTool(mock_memory_trait)
        tool.execute(query="test", category="preferences")

        mock_memory_trait.recall.assert_called_once_with(
            query="test",
            top_k=5,
            min_similarity=0.3,
            categories=["preferences"],
        )

    def test_recall_failure(self):
        """Handle recall failure gracefully."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        mock_memory_trait.has_embedder = True
        mock_memory_trait.recall.side_effect = RuntimeError("Embedder error")

        tool = RecallTool(mock_memory_trait)
        result = tool.execute(query="test")

        assert result.success is False
        assert "Embedder error" in result.error

    def test_tool_properties(self):
        """Tool has correct properties."""
        from llm_gent.core.tools.builtin.memory import RecallTool

        mock_memory_trait = MagicMock()
        tool = RecallTool(mock_memory_trait)

        assert tool.name == "recall"
        assert "query" in tool.parameters["properties"]
        assert "limit" in tool.parameters["properties"]
        assert "category" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]


class TestCompleteTaskTool:
    """Tests for CompleteTaskTool."""

    def test_complete_done(self):
        """Complete with done status."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="The answer is 42.")

        assert result.success is True
        assert result.terminal is True
        assert result.terminal_data == {
            "status": "done",
            "conclusion": "The answer is 42.",
        }
        assert "done" in result.output

    def test_complete_stuck(self):
        """Complete with stuck status."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="stuck", conclusion="Need database access.")

        assert result.success is True
        assert result.terminal is True
        assert result.terminal_data == {
            "status": "stuck",
            "conclusion": "Need database access.",
        }
        assert "stuck" in result.output

    def test_invalid_status(self):
        """Invalid status returns error."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="invalid", conclusion="test")

        assert result.success is False
        assert result.terminal is False
        assert "Invalid status" in result.error

    def test_missing_conclusion(self):
        """Missing conclusion returns error."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="")

        assert result.success is False
        assert "required" in result.error.lower()

    def test_whitespace_only_conclusion(self):
        """Whitespace-only conclusion is rejected."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="   \n\t  ")

        assert result.success is False
        assert "required" in result.error.lower()

    def test_conclusion_is_trimmed(self):
        """Conclusion whitespace is trimmed."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()
        result = tool.execute(status="done", conclusion="  The answer is 42.  ")

        assert result.success is True
        assert result.terminal_data["conclusion"] == "The answer is 42."

    def test_tool_properties(self):
        """Tool has correct properties."""
        from llm_gent.core.tools.builtin.complete import CompleteTaskTool

        tool = CompleteTaskTool()

        assert tool.name == "complete_task"
        assert "status" in tool.parameters["properties"]
        assert "conclusion" in tool.parameters["properties"]
        assert "status" in tool.parameters["required"]
        assert "conclusion" in tool.parameters["required"]
        assert tool.parameters["properties"]["status"]["enum"] == ["done", "stuck"]
