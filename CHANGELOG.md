# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Development: `make check` and CI fail on duplicated code blocks, unused private definitions,
  and the ruff rule groups C90, SIM, RET, PIE, PERF and FURB.

### Fixed

- `llm-gent stop` and `feedback` print the gateway's error detail on a 400, as `ask` and `start` do.

## [0.4.0] - 2026-10-26

### Added

- `llm_gent.flow`: compose verbs into flows with `Factory` — `call`, `then`, `conclude`,
  `branch`, `iterate` and `map` steps with `guard`, `rescue`, `on_error`, `after` and
  `on_item_complete` hooks. See "Flow composition" in `docs/index.md`.
- Durable runs: checkpoints are full snapshots of a run, kept as a history in an
  `appinfra.cas` store (`PgStore` for Postgres). `run(resume=...)` continues a halted or
  crashed run where it stopped; `run(restart=...)` starts over from a commit's data.
- One halt per run (`with_halt`; `run()` returns `HALTED`), shortcuts that fast-forward a
  region on a run signal (`with_shortcut`, `drain=`), checkpoint policy and named
  checkpoints, and `History` for reading a run's commits.
- `Loop` / `LoopFactory`: a SAIA turn as a step, with lifecycle hooks. A paused turn is
  saved in the checkpoint and resumes within what is left of its iteration and token caps.
- Resources: app-defined run-scoped objects saved and restored with the run
  (`with_resource`, `Resource`, `resource_method`).
- Cost tracking: `CostTracker` with pluggable `PricingProvider` (`LLMPricingProvider`
  included) and `CostTrait`. In a flow the tracker is a resource under `COST`;
  `with_budget` sets a per-run limit the agent reads through `ctx.cost`.
- `ensure_schema` / `SchemaManager` migrate gent's own `gent_`-prefixed tables.
- `AgentFactory(lg)` and `AgentFactory.from_config(dict)` build an agent without a
  `PlatformContext`.
- `RunnableAgent` carries the runtime methods; `Agent` is a plain container and an async
  context manager.
- Fetch tools follow redirects (`max_redirects`, default 5) and return a `FetchResult`
  with the final URL.
- Flow examples under `llm_gent.examples.flow`, and `prompt_agent`; examples run in CI.

### Changed

- **Breaking:** `LearnTrait` / `LearnConfig` split into `MemoryTrait` / `MemoryConfig` and
  `TrainingTrait` / `TrainingConfig`; `TraitName.LEARN` becomes `MEMORY` and `TRAINING`,
  so manifests declaring `required_traits: [learn]` use `memory`.
- **Breaking:** traits receive their clients by injection, built by `TraitFactory`
  (renamed from `core.traits.factory.Factory`). `AgentFactory` is renamed from
  `core.agent.factory.Factory`.
- **Breaking:** `LLMTrait` raises `ValueError` on a system message anywhere but first.
- Requires appinfra `>= 0.13.0`, llm-kelt `>= 0.5.0` (its `kelt_`-prefixed tables),
  llm-saia `>= 0.6.3` and llm-infer `>= 0.8.0`; adds alembic, cattrs and SQLAlchemy 2.0.

### Fixed

- `Identity("name")` works with the name alone; `context_key` defaults to the name.
- An attached `DirectiveTrait` now reaches the model: `LLMTrait.complete()` prepends it
  as the system message.
- A bus coordinator whose start fails releases the ports it bound.
- Bus channels and migrations work with appinfra's logger name ownership and channel
  send timeout.
- The `codebase-qna` example loads and runs its scheduled task.

### Removed

- **Breaking:** `llm_gent.Config`; use `AgentFactory.from_config`.

## [0.3.3] - 2026-09-08

### Fixed

- Base config `etc/llm-gent.yaml` now ships inside the wheel under
  `llm_gent/etc/`; `llm-gent serve` on an installed package no longer
  raises `FileNotFoundError`.

### Changed

- CLI adopts appinfra 0.11.0 faceted `AppBuilder`: `.config.with_spec`,
  `.cli.with_all_flags`, `.tools.with_tools`. Full standard flag set
  exposed (`--etc-dir`, `--config`, log flags, `-v/--version`).
  Requires appinfra `>= 0.11.0`.
- Removed the `.env.yaml`-alongside-etc convention; user overrides go
  under `$XDG_CONFIG_HOME/llm-works/llm-gent.yaml` per config protocol.
- `llm_gent/etc/pg.yaml` adopts canonical `pgserver.name: llm-works-pg`
  and `port: 25432` (shared llm-works local container); main DB renamed
  `gent_main` (was mis-pointing at `kelt`). Existing local data requires
  env override (`INFRA_PGSERVER_PORT=7632`) or manual migration.

## [0.3.2] - 2026-08-31

### Fixed

- `examples/` scripts now ship inside the wheel as the `llm_gent.examples`
  subpackage. Prior releases excluded them from the sdist/wheel, so
  `python -m llm_gent.examples.quickstart` raised `ModuleNotFoundError`
  against an installed package. Wheel-smoke CI now runs the packaged path
  to guard against regressions of this class.

## [0.3.1] - 2026-08-30

### Added

- `CONTRIBUTING.md` with DCO sign-off requirements for external contributions.
- `LLMTrait.complete()` and `complete_async()` accept OpenAI-style message
  dicts (`{"role": ..., "content": ...}`) in addition to `Message` instances.
- `examples/quickstart.py` — runnable end-to-end example.
- Python 3.14 declared in package metadata.

### Fixed

- README and docs quick start now match the shipped API surface.
- `LearnTrait` no longer passes the removed `ensure_schema` kwarg to
  `llm_kelt.Client` — relies on the default `schema_mode=SchemaMode.ENSURE`,
  which preserves the prior behavior. Restores compatibility with
  `llm-kelt>=0.4`.

## [0.3.0] - 2026-06-19

### Added

- `WebFetchTool` — fetches web pages and extracts main content as readable text using trafilatura
  (boilerplate removal, nav/footer stripping); non-HTML content passes through as-is
- `WebSearchTool` — web search tool with pluggable `WebSearchBackend` protocol; returns structured
  `{title, url, snippet}` results with automatic retry, pagination, and configurable rate limiting
- `WebSearchBackend` — protocol for search provider implementations
- `WebSearchBackendFactory` ABC — factory pattern for creating search backends from YAML config;
  supports `factory:` (dynamic import) resolution for external backend implementations
- `examples/web_search.py` — demo showing `WebSearchBackend` protocol with stub implementation
- `trafilatura>=2.0` dependency for web content extraction
- `llm_gent.bus` and `llm_gent.hub` packages — swarm orchestration: Hub coordinator with ZMQ
  three-socket transport (ROUTER/DEALER + PUB/SUB), envelope-wrapped v1 protocol, heartbeat
  health tracking, membership broadcasts, and message tier classification
- `Handler` protocol and `AgentRunner` external-gateway class so external processes can join the
  swarm without inheriting from `Agent`; internal runner accessible as `ManagedAgentRunner`
- `GET /bus/config` HTTP endpoint for external-agent bus discovery

### Fixed

- `LearnTrait` now uses `LLMClientFactory.embeddings()` to create embedding clients, compatible
  with the new backend-based `EmbeddingClient` API in llm-infer
- `LearnTrait` embedding calls updated to use `model` parameter (renamed from `model_name` in
  llm-kelt)
- `_PinnedIPTransport` now reads the httpcore response stream before accessing `.content`,
  fixing a crash on real HTTP calls (only surfaced with IP pinning enabled)

### Changed

- **Breaking:** `ToolFactory` now requires a `Logger` argument: `ToolFactory(lg)` instead of
  `ToolFactory()`
- **Breaking (config):** `llm.yaml` `backoff` key moved from under `rate_limit` to under `retry`
- Unit test coverage from 59% to 75%; coverage threshold raised from 50% to 70%
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

[Unreleased]: https://github.com/llm-works/llm-gent/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/llm-works/llm-gent/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/llm-works/llm-gent/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/llm-works/llm-gent/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/llm-works/llm-gent/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/llm-works/llm-gent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/llm-works/llm-gent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/llm-works/llm-gent/releases/tag/v0.1.0
