# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Flow hooks (`llm_gent.flow.hooks`, `Flow.with_hooks`): called before and after each run and
  step, and after each Loop model call and checkpoint; a `before_*` hook can deny or pause.
- `llm_gent.provenance.VersionControl` commits a record of each step, model call, checkpoint and
  run to an appinfra CAS repository, as a Flow hook or explicitly (`commit`, `activity`,
  `llm_call`) without Flow.
- `llm_gent.core.llm.testing.DelayedScriptedBackend`: llm-infer's scripted backend with an awaited
  per-call delay and optional request recording, so concurrent runs overlap in experiments.
- `llm_gent.core.Run`, a handle on one execution (await it, `halt()` it, check `done`).
- Streamed asks: `ask(..., on_step=)` reports each model call answering the question as an
  `AskStep`; `Hub.ask_stream` yields the steps, then the answer, and
  `POST /agents/{name}/ask/stream` sends them as server-sent events.

### Changed

- Building from source requires setuptools `>= 77.0.3`; the package declares its license as the
  SPDX expression `Apache-2.0`.
- A paused turn's messages are stored as an appinfra CAS `Vector`, one blob per message, so a turn
  paused again stores only the messages it added. Earlier gent versions cannot read such snapshots.
- `PgStore` writes a batch of objects in one transaction and looks up object types in one query;
  a checkpoint stores its snapshot as one batch.
- `retention="gc_on_success"` deletes only the checkpoint history (`HEAD` and its tags) when the
  repository holds other refs, such as version-control records; otherwise the whole repository.
- Breaking: `core.cost.CostTracker` is `core.cost.Tracker` and `core.llm.LLMCaller` is
  `core.llm.Caller`; the `llm_gent` package exports them as `CostTracker` and `LLMCaller`.
- Breaking: a `Platform` built once per process (`PlatformFactory(lg).from_config(...)`) holds the
  model router, the trait and tool factories, the trait registry and one SAIA per role; agents
  (`AgentFactory(platform)`) and runs (`FlowRunner(lg, platform=)`) share it. Removed:
  `AgentFactory(lg)` and `PlatformContext`. Agent, trait and tool factories are `Factory` in
  their subpackages; `llm_gent` exports them with the prefix.
- Breaking: a Flow is a structure that runs many times at once. State, halt, signals, history and
  retention are `FlowRunner.start()` arguments, and a run given no state gets a fresh one
  (`StateFactory.new()`, else `{}`). Removed: `Flow(state=)` / `Flow.state` / `with_halt` /
  `with_signal` / `with_checkpoint_store`, `Factory(state=, halt=)`, `Factory.with_halt`, and
  `Factory.create()`'s `state=`, `halt=`, `checkpointer=` and `client_flow_id=`.
- Breaking: a flow never runs itself. `FlowRunner.start(flow, ...)` starts a run and returns
  it to await or `halt()`; `start=Fresh() | Resume(at) | Restart(at)` replaces `resume=` /
  `restart=`. A started flow and the flows under it refuse further changes. Removed: `Flow.run()`
  and `ResumeMode`.
- Breaking: a run runs on its runner's platform, `FlowRunner(lg, platform=...)`, read as
  `ctx.platform`; `PlatformFactory.with_saia_factory()` sets how roles become SAIAs. Removed:
  `saia_factory=` / `traits=` on `Flow` and `Factory`, `Factory.with_saia_factory` /
  `with_traits` and `Flow.traits`; `Flow.dispatch` requires `platform=`.
- Breaking: `Role` and `SAIAFactory` move to `llm_gent.core`; `llm_gent.flow` no longer exports
  them.
- Breaking: runs keep their histories in the runner's store, `FlowRunner(lg, platform=, store=)`.
  Removed: `checkpoint_store=` on `Flow` and `Factory`, `Factory.with_checkpoint_store`.
  `build_canonical_flow` takes `checkpointed=` for `store=`; `resume_in_subprocess` flow builders
  take `(lg, **kwargs)`; `assert_resume_determinism` takes a platform.
- Breaking: a run's cost tracker is the run's: `FlowRunner.start(flow, cost=tracker)` runs it on
  a child of `tracker`, and a resumed run restores its spend into its own child, never into
  `tracker`. `CostTracker.exceeded`, `urgent_wrapup` and `remaining` count ancestors' caps.
  Removed: `Flow.with_cost_tracker`, `Factory(cost_tracker=)` / `with_cost_tracker`.
- Breaking: model access is the platform's, not a trait. `LLMTrait`, `LLMConfig` and
  `TraitName.LLM` are removed; a plain call is `Caller(lg, platform.llm).chat(...)`, which takes
  `output_schema=` for structured output. An agent config with an `llm` key raises `ConfigError`.
- Breaking: with memory configured, `PlatformFactory.from_config()` builds one `MemoryTrait`
  (`platform.traits`) and migrates the memory schema then; an agent's memory is its binding of
  it, `agent.require(MemoryTrait)`. Removed: `MemoryTrait(agent, ...)`, its `owns_*` flags and
  `with_*` swaps, `TraitFactory.create_memory_trait`, `MemoryResources`. `AgentStorage` takes
  `(lg, database, context)`.
- Breaking: an agent's asks run in sessions. `ask(question, session=None)` returns the
  `ExecutionResult`, whose `session` id continues the conversation; ids are issued by the agent,
  and an unknown one raises `SessionNotFoundError`. The serve API and `Hub.ask` carry the session.
  `ConversationTrait` is removed.
- Breaking: chat is opt-in. `ask` moves from `Runnable` to the `Chat` mixin; a prompt agent with a
  `chat` section (the sessions' settings, replacing `conversation`) is a `ChatAgent`, and the
  runtime refuses an ask to an agent without chat. `kwargs` on the bus, REST and `Hub`/`Core` ask
  pass the agent's own `ask` keywords; `Handler.on_ask` takes them and `on_step`.
- Breaking: injected agents run as services on appinfra's service `Runtime`; `execution: process`
  spawns them with `isolate=True`. `Core` takes the started `runtime` it spawns them on, and
  `ManagedAgentRunner` takes the service's `stopping` token.
- Breaking: a platform `Trait` is shared by every agent and run; a `ScopedTrait` `bind(agent)`s a
  `TraitBinding` (the old `Trait` protocol) the agent holds, for memory a `MemoryBinding` with
  memory's operations. `Agent(lg, platform, config)`; `get` / `has` / `require` replace `*_trait`;
  an agent `kelt.schema` raises; memory tools take `ToolFactory.create(..., memory=)`.
- Breaking: with memory configured, the platform builds one `RatingTrait`;
  `agent.require(RatingTrait)` is the agent's `RatingBinding` (its context key and `rating`
  config), with the rating operations. Removed: `RatingTrait(agent, router, config)` and
  `TraitFactory`'s router argument.
- Breaking: requests over the bus go through each ZMQ transport's `correlator` (appinfra's
  `Correlator`), in place of wrapping it in `BufferedChannel`. Error replies raise
  `ChanResponseError` (was `ChannelResponseError`), a closed transport `ChanClosedError` and a
  full one `TimeoutError`; the transports no longer have `recv()`.

### Fixed

- `AtomicFactsBackend.get_unrated_count`, and so `llm-gent rate`, failed on Postgres with a
  `GROUP BY` error: the count kept the listing's `ORDER BY`.
- Structured output keeps a reply that already matches the schema; a schema with defaulted fields
  no longer has a valid reply unwrapped into an invalid one.
- The serve API's routes failed every request with "Request must have an 'id' attribute": they
  passed the IPC channel the request id where it takes the request.

## [0.4.1] - 2026-11-01

### Changed

- Requires appinfra `>= 0.14.0`, llm-infer `>= 0.8.1` and llm-kelt `>= 0.5.1`; SQLAlchemy 2.1,
  psycopg 3 and pydantic now come through appinfra's `sql` and `validation` extras. `PgStore`
  works with psycopg 3.
- Development: `make check` and CI fail on duplicated code blocks, unused private definitions,
  and the ruff rule groups C90, SIM, RET, PIE, PERF and FURB.

### Fixed

- `llm-gent ask`, `start`, `stop`, `feedback` and `list` print the gateway's error detail when a
  request fails; before, only `ask` and `start` did, and only on a 400.

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

[Unreleased]: https://github.com/llm-works/llm-gent/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/llm-works/llm-gent/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/llm-works/llm-gent/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/llm-works/llm-gent/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/llm-works/llm-gent/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/llm-works/llm-gent/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/llm-works/llm-gent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/llm-works/llm-gent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/llm-works/llm-gent/releases/tag/v0.1.0
