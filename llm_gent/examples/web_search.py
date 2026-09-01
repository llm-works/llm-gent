#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Demo: web search with pluggable backend.

Demonstrates the WebSearchTool with a stub backend. Real backends (Brave,
Serper, etc.) must be implemented separately and satisfy the WebSearchBackend
protocol.

Usage:
    python -m llm_gent.examples.web_search "python asyncio tutorial"
    python -m llm_gent.examples.web_search "rust vs go performance" --fetch 1

Flags:
    --max-results N   Number of search results (1-8, default 5)
    --fetch N         Fetch the Nth result page and print readable text
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse

from appinfra.log import create_lg

from llm_gent import (
    WebFetchTool,
    WebSearchBackend,
    WebSearchTool,
)


# ---------------------------------------------------------------------------
# Stub backend (used when no API key is available)
# ---------------------------------------------------------------------------


class StubSearchBackend:
    """Minimal backend that returns hardcoded results for demonstration."""

    def search(self, query: str, max_results: int, offset: int = 0) -> list[dict[str, str]]:
        """Return canned results; replace this stub with your own backend for real search."""
        all_results = [
            {
                "title": f"Result {i} for: {query}",
                "url": f"https://example.com/page{i}",
                "snippet": f"Sample search result snippet #{i}.",
            }
            for i in range(1, 11)
        ]
        return all_results[offset : offset + max_results]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web search + fetch demo")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N results (pagination)")
    parser.add_argument("--fetch", type=int, default=0, help="Fetch the Nth result page (1-based)")
    return parser.parse_args()


def _extract_urls(output: str) -> list[str]:
    """Parse URLs from formatted search output."""
    return [
        line.strip().removeprefix("URL: ")
        for line in output.splitlines()
        if line.strip().startswith("URL: ")
    ]


def _fetch_result_page(web_fetch: WebFetchTool, urls: list[str], index: int) -> None:
    """Fetch and print the Nth result page (1-based index)."""
    if index > len(urls):
        print(f"\nOnly {len(urls)} results, cannot fetch #{index}", file=sys.stderr)
        sys.exit(1)

    url = urls[index - 1]
    print(f"\n{'=' * 60}")
    print(f"Fetching result #{index}: {url}")
    print(f"{'=' * 60}\n")

    result = web_fetch.execute(url=url)
    if not result.success:
        print(f"Fetch failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    print(result.output)


def main() -> None:
    args = _parse_args()

    lg = create_lg("web_search_demo", "info")

    backend: WebSearchBackend = StubSearchBackend()
    print("(using stub backend — implement WebSearchBackend for real results)\n")

    web_fetch = WebFetchTool(lg=lg)
    web_search = WebSearchTool(lg=lg, backend=backend)

    print(f"Searching: {args.query}\n")
    result = web_search.execute(query=args.query, max_results=args.max_results, offset=args.offset)

    if not result.success:
        print(f"Search failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    print(result.output)

    if args.fetch >= 1:
        _fetch_result_page(web_fetch, _extract_urls(result.output), args.fetch)


if __name__ == "__main__":
    main()
