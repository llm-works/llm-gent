#!/usr/bin/env python3
"""Demo: web search with BraveSearchBackend.

Uses ``BraveSearchBackend`` when ``BRAVE_SEARCH_API_KEY`` is set, otherwise
falls back to a stub backend that returns hardcoded results.

Usage:
    export BRAVE_SEARCH_API_KEY="BSA..."
    python examples/web_search.py "python asyncio tutorial"
    python examples/web_search.py "rust vs go performance" --fetch 1

Flags:
    --max-results N   Number of search results (1-8, default 5)
    --fetch N         Fetch the Nth result page and print readable text
"""

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from appinfra.log import create_lg

from llm_gent import BraveSearchBackend, WebFetchTool, WebSearchBackend, WebSearchTool


# ---------------------------------------------------------------------------
# Stub backend (used when no API key is available)
# ---------------------------------------------------------------------------


class StubSearchBackend:
    """Minimal backend that returns hardcoded results for demonstration."""

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Return canned results. Set BRAVE_SEARCH_API_KEY for real search."""
        all_results = [
            {
                "title": f"Result 1 for: {query}",
                "url": "https://example.com/page1",
                "snippet": "This is a sample search result snippet.",
            },
            {
                "title": f"Result 2 for: {query}",
                "url": "https://example.com/page2",
                "snippet": "Another example result with relevant content.",
            },
            {
                "title": f"Result 3 for: {query}",
                "url": "https://example.com/page3",
                "snippet": "Third result demonstrating the search interface.",
            },
        ]
        return all_results[:max_results]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web search + fetch demo")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=5, help="Number of results (1-8)")
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

    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    backend: WebSearchBackend
    if api_key:
        backend = BraveSearchBackend(lg=lg, api_key=api_key)
        print("(using Brave Search API)\n")
    else:
        backend = StubSearchBackend()
        print("(using stub backend — set BRAVE_SEARCH_API_KEY for real results)\n")

    web_fetch = WebFetchTool(lg=lg)
    web_search = WebSearchTool(lg=lg, backend=backend, max_queries_per_minute=0)

    print(f"Searching: {args.query}\n")
    result = web_search.execute(query=args.query, max_results=args.max_results)

    if not result.success:
        print(f"Search failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    print(result.output)

    if args.fetch >= 1:
        _fetch_result_page(web_fetch, _extract_urls(result.output), args.fetch)


if __name__ == "__main__":
    main()
