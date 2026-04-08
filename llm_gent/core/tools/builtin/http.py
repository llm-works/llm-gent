"""HTTP fetch tool."""

from __future__ import annotations

import ipaddress
import socket
import ssl
from collections.abc import Iterable
from typing import Any
from urllib.parse import ParseResult, urlparse

import httpcore
import httpx
from httpcore._backends.base import SOCKET_OPTION

from ..base import BaseTool, ToolResult


class _PinnedIPBackend(httpcore.SyncBackend):
    """Network backend that connects to a pre-validated IP address.

    This prevents DNS rebinding attacks by ensuring the connection uses
    the same IP that was validated during the SSRF check, rather than
    resolving DNS again (which could return a different IP).

    TLS handshake still uses the original hostname for SNI and certificate
    verification, so HTTPS works correctly.
    """

    def __init__(self, pinned_ip: str) -> None:
        self._pinned_ip = pinned_ip

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        """Connect to the pinned IP instead of resolving 'host' via DNS."""
        backend = httpcore.SyncBackend()
        return backend.connect_tcp(self._pinned_ip, port, timeout, local_address, socket_options)


class _PinnedIPTransport(httpx.BaseTransport):
    """Custom httpx transport that pins connections to a pre-validated IP.

    This transport uses httpcore with a custom network backend to ensure
    all connections go to the specified IP address, preventing DNS rebinding.
    """

    def __init__(
        self,
        pinned_ip: str,
        timeout: float,
        verify: bool = True,
    ) -> None:
        self._pinned_ip = pinned_ip
        self._timeout = timeout
        self._verify = verify
        self._pool: httpcore.ConnectionPool | None = None

    def _get_pool(self) -> httpcore.ConnectionPool:
        """Get or create the connection pool."""
        if self._pool is None:
            backend = _PinnedIPBackend(self._pinned_ip)
            ssl_context = ssl.create_default_context() if self._verify else None
            self._pool = httpcore.ConnectionPool(
                network_backend=backend,
                ssl_context=ssl_context,
            )
        return self._pool

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Send request using the pinned IP connection pool."""
        pool = self._get_pool()
        core_request = self._build_core_request(request)
        core_response = pool.handle_request(core_request)
        return self._build_httpx_response(core_response, request)

    def _build_core_request(self, request: httpx.Request) -> httpcore.Request:
        """Convert httpx Request to httpcore Request."""
        timeout_dict = {
            "connect": self._timeout,
            "read": self._timeout,
            "write": self._timeout,
            "pool": self._timeout,
        }
        return httpcore.Request(
            method=request.method.encode() if isinstance(request.method, str) else request.method,
            url=httpcore.URL(
                scheme=request.url.scheme.encode(),
                host=request.url.host.encode() if request.url.host else b"",
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=[
                (k.encode() if isinstance(k, str) else k, v.encode() if isinstance(v, str) else v)
                for k, v in request.headers.raw
            ],
            content=request.content,
            extensions={"timeout": timeout_dict},
        )

    def _build_httpx_response(
        self, core_response: httpcore.Response, request: httpx.Request
    ) -> httpx.Response:
        """Convert httpcore Response to httpx Response."""
        # Read the stream before wrapping — httpcore responses are streaming
        # and .content raises if the stream hasn't been consumed yet.
        core_response.read()
        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=httpx.ByteStream(core_response.content),
            request=request,
        )

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None


class HTTPFetchTool(BaseTool):
    """Tool for fetching content from URLs.

    Allows the agent to retrieve content from HTTP/HTTPS URLs. Useful for
    fetching API responses, web pages, or other remote content.

    Security Features:
        - SSRF protection: By default, blocks requests to private/internal IPs
          (localhost, RFC 1918 ranges, link-local, cloud metadata endpoints).
        - Domain restrictions: Use allowed_domains or blocked_domains to control
          which domains can be accessed.

    Example:
        tool = HTTPFetchTool(allowed_domains=["api.github.com"])
        result = tool.execute(url="https://api.github.com/repos/owner/repo")
        print(result.output)
    """

    name = "http_fetch"
    description = (
        "Fetch content from a URL via HTTP GET. "
        "Use for: retrieving API responses, fetching web pages, downloading text content. "
        "Returns the response body as text. "
        "Supports custom headers for authentication or content negotiation."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (must be http:// or https://)",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers to include in the request",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_response_size: int = 1000000,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        default_headers: dict[str, str] | None = None,
        block_private_ips: bool = True,
    ) -> None:
        """Initialize HTTP fetch tool.

        Args:
            timeout: Request timeout in seconds. Defaults to 30.
            max_response_size: Maximum response size in bytes. Defaults to 1MB.
            allowed_domains: If set, only these domains can be accessed.
                            Takes precedence over blocked_domains.
            blocked_domains: If set, these domains are blocked.
                            Ignored if allowed_domains is set.
            default_headers: Headers to include in all requests.
            block_private_ips: If True (default), blocks requests to private/internal
                              IP addresses to prevent SSRF attacks. This includes:
                              - Loopback (127.0.0.0/8, ::1)
                              - Private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
                              - Link-local (169.254.0.0/16, fe80::/10) - includes cloud metadata
                              - Other non-routable addresses
        """
        self._timeout = timeout
        self._max_response_size = max_response_size
        self._allowed_domains = set(allowed_domains) if allowed_domains else None
        self._blocked_domains = set(blocked_domains) if blocked_domains else None
        self._default_headers = default_headers or {}
        self._block_private_ips = block_private_ips

    def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch content from a URL.

        Args:
            **kwargs: Must contain 'url'. Optional: 'headers'.

        Returns:
            ToolResult with response body or error.
        """
        # Validate URL argument
        url = kwargs.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult(success=False, output="", error="Missing or invalid 'url' argument")

        # Parse and validate URL
        parsed = self._parse_url(url)
        if isinstance(parsed, ToolResult):
            return parsed

        # Check domain restrictions
        if error := self._check_domain(parsed.netloc):
            return error

        # Resolve and validate IPs (SSRF protection)
        # Returns the validated IP to pin the connection to, preventing DNS rebinding
        ip_result = self._resolve_and_validate_ip(parsed.netloc)
        if isinstance(ip_result, ToolResult):
            return ip_result
        pinned_ip = ip_result  # IP to use for the actual connection

        # Validate headers
        headers = self._build_headers(kwargs.get("headers"))
        if isinstance(headers, ToolResult):
            return headers

        # Make the request using the pre-validated IP
        return self._fetch(url, headers, pinned_ip)

    def _parse_url(self, url: str) -> ParseResult | ToolResult:
        """Parse and validate URL. Returns ParseResult or error."""
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Invalid URL: {e}")

        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid URL scheme '{parsed.scheme}'. Must be http or https.",
            )

        if not parsed.netloc:
            return ToolResult(success=False, output="", error="Invalid URL: missing domain")

        return parsed

    def _check_domain(self, domain: str) -> ToolResult | None:
        """Check if domain is allowed. Returns error if blocked."""
        # Normalize domain (remove port if present)
        domain_only = domain.split(":")[0].lower()

        if self._allowed_domains is not None:
            if not self._matches_domain(domain_only, self._allowed_domains):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Domain '{domain_only}' is not in allowed list",
                )
            return None

        if self._blocked_domains is not None and self._matches_domain(
            domain_only, self._blocked_domains
        ):
            return ToolResult(
                success=False,
                output="",
                error=f"Domain '{domain_only}' is blocked",
            )

        return None

    def _matches_domain(self, domain: str, domain_set: set[str]) -> bool:
        """Check if domain matches any in the set (including subdomains)."""
        for allowed in domain_set:
            allowed_lower = allowed.lower()
            if domain == allowed_lower:
                return True
            # Check subdomain match (e.g., "api.github.com" matches "github.com")
            if domain.endswith("." + allowed_lower):
                return True
        return False

    def _resolve_and_validate_ip(self, netloc: str) -> str | ToolResult:
        """Resolve hostname and validate IPs. Returns first valid IP or error.

        This method resolves DNS once and returns a validated IP address to use
        for the actual HTTP connection. This prevents DNS rebinding attacks where
        an attacker's DNS could return a public IP during validation but a private
        IP during the actual request.

        Args:
            netloc: The network location (hostname:port or just hostname).

        Returns:
            The first non-blocked IP address to use for the connection,
            or a ToolResult error if all IPs are blocked or resolution fails.
        """
        hostname = netloc.split(":")[0]
        resolved = self._resolve_hostname(hostname)
        if isinstance(resolved, ToolResult):
            return resolved

        if not self._block_private_ips:
            # Return first resolved IP without validation
            if resolved:
                return resolved[0]
            return ToolResult(
                success=False,
                output="",
                error=f"No IP addresses found for '{hostname}'",
            )

        # Find first non-blocked IP
        for ip_str in resolved:
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if not self._is_blocked_ip(ip):
                return ip_str

        # All IPs are blocked
        return ToolResult(
            success=False,
            output="",
            error=f"Blocked: '{hostname}' resolves only to private/internal IPs",
        )

    def _resolve_hostname(self, hostname: str) -> list[str] | ToolResult:
        """Resolve hostname to IP addresses. Returns list of IPs or error."""
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            return [str(sockaddr[0]) for _family, _, _, _, sockaddr in addr_info]
        except socket.gaierror as e:
            return ToolResult(
                success=False,
                output="",
                error=f"DNS resolution failed for '{hostname}': {e}",
            )

    def _is_blocked_ip(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """Check if an IP address should be blocked (private, loopback, etc.)."""
        # Check common properties that indicate non-public addresses
        if ip.is_private:
            return True
        if ip.is_loopback:
            return True
        if ip.is_link_local:
            return True
        if ip.is_reserved:
            return True
        if ip.is_multicast:
            return True

        # IPv4-specific: block 0.0.0.0/8 (current network)
        return isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.IPv4Network("0.0.0.0/8")

    def _build_headers(self, custom_headers: Any) -> dict[str, str] | ToolResult:
        """Build request headers. Returns headers dict or error."""
        headers = dict(self._default_headers)

        if custom_headers is None:
            return headers

        if not isinstance(custom_headers, dict):
            return ToolResult(
                success=False,
                output="",
                error="Invalid 'headers' argument: must be an object",
            )

        for key, value in custom_headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return ToolResult(
                    success=False,
                    output="",
                    error="Invalid header: keys and values must be strings",
                )
            headers[key] = value

        return headers

    def _fetch(self, url: str, headers: dict[str, str], pinned_ip: str | None) -> ToolResult:
        """Perform HTTP request using a pre-validated IP to prevent DNS rebinding."""
        try:
            response = self._execute_request(url, headers, pinned_ip)
            return self._build_response(response)
        except (httpx.TimeoutException, httpcore.TimeoutException):
            return ToolResult(
                success=False, output="", error=f"Request timed out after {self._timeout} seconds"
            )
        except (httpx.ConnectError, httpcore.ConnectError) as e:
            return ToolResult(success=False, output="", error=f"Connection failed: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Request failed: {e}")

    def _execute_request(
        self, url: str, headers: dict[str, str], pinned_ip: str | None
    ) -> httpx.Response:
        """Execute the HTTP GET request with optional IP pinning."""
        if pinned_ip:
            transport = _PinnedIPTransport(pinned_ip, self._timeout)
            with httpx.Client(transport=transport) as client:
                return client.get(url, headers=headers)
        with httpx.Client(timeout=self._timeout) as client:
            return client.get(url, headers=headers)

    def _build_response(self, response: httpx.Response) -> ToolResult:
        """Build ToolResult from HTTP response."""
        if error := self._check_content_length(response):
            return error

        content_or_error = self._read_response_content(response)
        if isinstance(content_or_error, ToolResult):
            return content_or_error
        content, truncated = content_or_error

        return self._format_response(response, content, truncated)

    def _check_content_length(self, response: httpx.Response) -> ToolResult | None:
        """Check if Content-Length exceeds limit. Returns error or None."""
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self._max_response_size:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Response too large: {content_length} bytes (max {self._max_response_size})",
                    )
            except ValueError:
                pass  # Invalid Content-Length header, proceed to read and truncate if needed
        return None

    def _read_response_content(self, response: httpx.Response) -> tuple[str, bool] | ToolResult:
        """Read response body, truncating if needed. Returns (content, truncated) or error."""
        try:
            content = response.text
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read response: {e}")

        if len(content) > self._max_response_size:
            return content[: self._max_response_size], True
        return content, False

    def _format_response(
        self, response: httpx.Response, content: str, truncated: bool
    ) -> ToolResult:
        """Format the final response."""
        if not response.is_success:
            output = f"HTTP {response.status_code}\n\n{content}"
            if truncated:
                output += "\n\n(response truncated)"
            return ToolResult(
                success=False,
                output=output,
                error=f"HTTP {response.status_code}: {response.reason_phrase}",
            )

        output = content
        if truncated:
            output += f"\n\n(truncated, max {self._max_response_size} chars)"
        return ToolResult(success=True, output=output)
