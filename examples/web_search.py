#!/usr/bin/env python3
"""Demo: search the web and fetch a result page.

Usage:
    python examples/web_search.py "python asyncio tutorial"
    python examples/web_search.py "rust vs go performance" --fetch 1
    python examples/web_search.py "site:github.com llm agent" --max-results 3

Flags:
    --max-results N   Number of search results (1-8, default 5)
    --fetch N         Fetch the Nth result page and print readable text
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from llm_gent import WebFetchTool, WebSearchTool


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

    web_fetch = WebFetchTool()
    web_search = WebSearchTool(max_queries_per_minute=0, web_fetch=web_fetch)

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
