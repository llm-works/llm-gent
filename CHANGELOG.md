# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `WebFetchTool` — fetches web pages and extracts main content as readable text using trafilatura
  (boilerplate removal, nav/footer stripping); non-HTML content passes through as-is
- `WebSearchTool` — web search tool with pluggable `WebSearchBackend` protocol; returns structured
  `{title, url, snippet}` results with automatic retry on retriable failures
- `WebSearchBackend` — protocol for search provider implementations
- `WebSearchBackendFactory` ABC — factory pattern for creating search backends from YAML config;
  supports `factory:` (dynamic import) resolution for external backend implementations
- `examples/web_search.py` — demo showing `WebSearchBackend` protocol with stub implementation
- `trafilatura>=2.0` dependency for web content extraction

### Fixed

- `LearnTrait` now uses `LLMClientFactory.embeddings()` to create embedding clients, compatible
  with the new backend-based `EmbeddingClient` API in llm-infer
- `_PinnedIPTransport` now reads the httpcore response stream before accessing `.content`,
  fixing a crash on real HTTP calls (only surfaced with IP pinning enabled)

### Changed

- **Breaking:** `BraveSearchBackend` and `SerperSearchBackend` removed from llm-gent; search
  backends must now be provided externally via `factory:` config path (e.g.,
  `factory: my_package.websearch.Factory`)
- Unit test coverage from 59% to 75%; coverage threshold raised from 50% to 70%
- **Breaking:** `ToolFactory` now requires a `Logger` argument: `ToolFactory(lg)` instead of
  `ToolFactory()`
- Refactored runtime to use `appinfra.service` for IPC and state management, removing ~780 lines
  of custom transport and state code
- Replaced local conversation layer with `llm-kelt` conversation module, delegating compaction to
  kelt's built-in auto-compaction
- Agent config now supports `enabled: false` to disable agents without removing them from config
- Agent config now supports `execution: "thread"` for in-process execution (default: "process")

### Removed

- Extracted `jokester-p` agent (~4,500 LOC) to standalone `agents` repo; prompt-based jokester
  remains as example agent
- Removed `core/conv` package and `ConversationRunner` (replaced by `llm-kelt` conversation)

## [0.2.0] - 2026-03-16

### Added

- Length-balanced DPO pairing with `--length-balance` and `--length-epsilons` to prevent length
  reward hacking via multi-pass pairing with progressively looser constraints
- rsLoRA scaling with `--rslora` flag (recommended for 32B+ models)
- NEFTune noise injection with `--neftune-alpha` for better generalization on small datasets
- DPO summary statistics showing pair distribution (e.g., 5★-2★: 100) and length metrics

## [0.1.0] - 2026-02-26

### Added

- Agent framework with trait-based architecture (LLM, Storage, Rating, Learn, Directive traits)
- Jokester-P agent: joke generation with novelty checking and quality rating
- LLM trait with multi-backend support via llm-infer
- Storage trait with PostgreSQL backend and schema migrations
- Rating trait for automated LLM-based content evaluation
- Learn trait for training data collection (SFT/DPO)
- Training infrastructure with SFT and DPO support via llm-kelt
- CLI tools for agent management, training, and statistics
- HTTP server for running agents as services
- JSON cleaner for robust LLM output parsing

### Changed

- Renamed package from llm-agent to llm-gent
- Migrated from llm-learn to llm-kelt for training
- Refactored training infrastructure to core modules

[Unreleased]: https://github.com/llm-works/llm-gent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/llm-works/llm-gent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/llm-works/llm-gent/releases/tag/v0.1.0
