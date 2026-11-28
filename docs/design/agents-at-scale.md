# Design: agents at scale

How llm-gent runs many agents in one process with nothing duplicated per agent: what is shared,
what each agent holds, and how flows run for agents.

Status: the target design, built in steps. Implemented: the `Platform` (one per process, built
by the platform `Factory` from `Config`) shared by agents and runs, with the shared clients
injected into traits; the scope a flow runs in as a fold of its declarations, the generic run,
flow runs started by a runner, and the run's own cost tracker (section 6). Not yet: a default
`SAIAFactory` the platform builds over its router, traits without per-agent instances, a run's
identity, agents supplying a run's identity and platform, and the API in section 8.

## 1. Goals

- An agent costs its own data and a few references: no per-agent model client, connection pool,
  trait copy, thread, process or socket. Ten thousand agents in one process is a routine case.
- Everything shared exists once per worker process.
- Creating an agent reads the same with and without Flow.
- Defaults live in factory config models (Pydantic), not in code paths.
- The model keeps later distribution across processes and machines, and routing across many
  inference servers, possible: structure is shared, state lives with agents and runs, and every
  run carries an identity.

## 2. The model

| | Holds | Instances |
|---|---|---|
| Platform | The shared clients (model router, database pool, memory clients), the trait and tool factories, the trait registry, the `SAIAFactory` and one SAIA per role | One per process |
| Trait | Shared logic and resources | One per trait type, used by every thread and event loop |
| Flow | Structure: what happens to state | One per definition, shared by every run |
| Run | One execution: its identity, halt, cost, history, signals, live scopes and cursors | Many per process |
| Context | One step of a run: the run, the scope folded so far, the position | One per step |
| Agent | An identity, and the platform its runs use | One per identity |

State is kept apart from the logic that acts on it. The platform, traits and flows are logic,
structure and shared wiring, built once; runs and their contexts are state, cheap and many.
Ownership is a tree of three levels, by lifetime:

```
Platform   one per process: clients, factories, the trait registry, one SAIA per role
   ▼
Run        one execution: identity, halt, cost, history, signals, scopes, cursors
   ▼
Context    one step: the run, the scope, the position (what a verb receives)
```

Only a run and what is under it are checkpointed; the platform holds no run's or agent's data.
An agent is not a level of the tree: it supplies a run's identity and platform.

## 3. Construction

Factories build; the platform they produce is passed around and builds nothing but one SAIA per
role. A class is named without its package's name inside the package (`platform.Factory`,
`agent.Factory`, `traits.Factory`); flat namespaces such as the top-level package export it with
the prefix (`PlatformFactory`, `AgentFactory`, `TraitFactory`). `Platform` is exported by
`llm_gent.core` and `llm_gent`.

| Object | Role |
|---|---|
| platform `Factory(lg).from_config(config)` | Builds the shared clients, the trait factory (through `TraitFactory`) and the tool factory, and injects them into a `Platform` with ownership, with an empty trait registry and the `SAIAFactory` given by `with_saia_factory()`. Reads the platform `Config`. |
| `Platform` | Passed to agents and runners. Holds references, builds one SAIA per role on first use, and closes what it owns on `close()` or at the end of a `with` or `async with` block. Its constructor takes every part, so code with parts of its own builds one without the factory. |
| `TraitFactory` | Builds traits from injected clients; holds no reference to the platform. |
| `AgentFactory(platform)` | Builds agents that reference the platform. Subclassed per agent type (`agent_class`, `required_traits`, `default_tools`). |
| flow `Factory(lg)` | Builds flows: structure only (state factory, flow class, declared resources). |
| `FlowRunner(lg, platform=, store=)` | Executes flows. Holds what its runs share: the platform and the checkpoint store. `start()` takes one run's inputs and returns the run. |

A platform requires its factories, not instances: its trait registry may be empty, and a SAIA
factory builds nothing until a step asks for a role.

`AgentFactory` does not build the platform: several agent types share one worker, and the
platform's lifecycle belongs to the worker. Factories build clients and inject them; a receiver
closes only what it is told it owns.

Models are wired in two ways:

- Config: the platform `Config.llm` is llm-infer's router config, merged over the packaged
  `etc/llm.yaml`. An empty config uses the packaged backends.
- Injection: `PlatformFactory(lg).with_llm(client)` takes a client the caller built, for example
  with llm-infer's `Factory(lg).openai(...)`. The caller keeps ownership.

The target default `SAIAFactory` builds SAIA over the platform's llm-infer router and reads the
`Role` fields: `style` becomes the system prompt, `backend` and `model` select the model. Until it
exists, a platform built without `with_saia_factory()` builds no SAIA: a step reading `ctx.saia`
raises.

Platform `Config` defaults: `llm` is the packaged `etc/llm.yaml`; `memory` is optional, and without
it no memory-backed trait or tool is available. An agent's only required field is its name.

## 4. Traits

Every capability is a trait. What a trait holds splits in two:

- **The trait (shared):** clients, pools, caches, config. One instance per trait type per
  process, in the platform's registry, built by `TraitFactory`.
- **Per-agent state:** what differs per agent: directive text, budget, Memory's identity-scoped
  view. The trait defines its state type and creates it per agent; the agent holds these state
  objects keyed by trait. Calls pass the agent: `llm.complete(agent, messages)`. An agent's
  conversations are not trait state: they are its sessions (see Conversations).

An agent costs only the state objects of the traits it uses.

| Trait | Shared part | Per-agent state or input |
|---|---|---|
| LLM | The router | Directive text; backend, model and adapter passed per call |
| Memory | One kelt `Client` | A kelt `Client.view(context_key)`: references only at creation, stores built on first use; embedding stores and the knowledge graph shared per client |
| Storage | On the Memory view | Identity, through the view |
| Rating | On the router and the Memory view | Identity, through the view |
| Training | Adapter lookup | None |
| Tools | Tool registry, tools keyed by name and config | The tool names the agent may use; memory tools take the Memory view |
| Directive, Method | Prompt assembly | Directive and method text |
| Cost | Pricing | Budget and tracker |

SAIA is not a trait: the platform's `SAIAFactory` builds one SAIA per role, with the role taken
from the agent's directive and the tools from the agent's tool names. HTTP is not a trait: the
runtime runs one server per worker, routed by agent name.

### Conversations

A SAIA holds configuration only (backend, tools and their executor, system prompt, guards) and is
shared by every run that uses its role. A SAIA call keeps its messages in the conversation it is
passed, or in a throwaway one; nothing of the call stays on the SAIA. Its tool executor is shared
by every run too, so it holds no run's data: it is stateless, or keys per-run data by the ids the
call carries, as a trait does.

A conversation belongs to a call site in a run, not to the SAIA or the agent: a `map` over 100
items in one run holds 100. A `Loop` takes the conversation from its step or, with a
`ConversationFactory` wired, creates one per dispatch; a turn the run's halt paused is saved in
the run's snapshot at the call's path and continued from there on resume. A conversation that
continues across steps is kept in the run's state, so it is checkpointed with the run.

An agent's own chats are its sessions. A session is one conversation under an id the agent
issues: `agent.ask(question)` opens one, and `agent.ask(question, session=id)` continues it; an
id the agent did not issue is refused. An agent holds any number of sessions, none of them
special; asks in one session run one at a time, asks in different sessions side by side. With a
memory database, sessions persist in kelt's session storage under the agent's `context_key`,
name and the session id, so an id is valid only at the agent that issued it, and an idle session
can leave memory and reload on its next ask. The agent config's `conversation` section sets the
conversations' limits.

### Thread-safety contract

A trait instance is shared by every thread and event loop in its process:

- No per-agent or per-call data on the trait; it lives in per-agent state or call arguments.
- Mutable shared state is synchronized; lazy initialization is guarded by a lock.
- Resources are safe across threads and event loops. Sync clients (httpx `Client`, a SQLAlchemy
  engine with per-use sessions) are. An async HTTP client is bound to the event loop that first
  used it, so llm-infer keeps one async client per event loop and one router serves every loop.
- Blocking I/O (kelt and Postgres are synchronous) runs off the event loop (`asyncio.to_thread`);
  otherwise it stalls every agent on that loop.
- Per-agent state is shared by the agent's concurrent runs. An agent and its runs stay on one
  event loop, so an update with no `await` inside is atomic and needs no lock. A check-then-act
  across an `await` can interleave with another run of the same agent.
- The budget is soft: charged after each call, halting once exceeded, it can overrun by the calls
  in flight across the agent's runs. A run's budget is a `CostTracker.child()` of the agent's.

## 5. Agents and registries

An agent's main job is to hold an identity: who acts, for per-agent state, provenance and
authorization. A run does not need an agent: an agent is one way to supply a run's identity and
platform.

An agent holds:

- its identity;
- per-agent state objects, keyed by trait, created only for the traits it uses;
- a reference to its `Platform`;
- the trait types it uses (`required_traits`), not trait instances.

No trait instance lives on an agent. `agent.get(T)` and `require(T)` resolve through
the platform's registry. A custom trait is registered with the platform like a built-in one, even
when one agent uses it: the platform owns its lifecycle and it follows the thread-safety contract.
A resource an agent manages itself is plain agent code, an attribute of the agent subclass that
the agent closes, not a trait.

A process keeps as many copies of a trait as it must, which is one: the platform's registry holds
one instance per trait type, and differences between agents or runs go in per-agent state, call
arguments or data keyed by the ids a call carries, not in a second instance. There is no API for
a second registry. Code that needs other instances builds a registry and a `Platform` of its own:
nothing prevents it, nothing encourages it. A registry is built whole by `TraitFactory`, which
wires the dependencies between traits (Rating and Storage take Memory's client and the LLM
router). Platform components (`SAIAFactory`, tools) hold no trait instances; they resolve traits
through the platform on each call. Copies that exist for a client's own reasons stay inside the
client: llm-infer keeps one async HTTP client per event loop behind one router.

## 6. Flow: structure and runs

A flow is defined once and runs many times at once, for many agents. It never runs itself: a
runner executes it, and each execution is a run.

- `Flow` is shared and immutable once started: nodes, verbs, checkpointer and checkpoint policy,
  hooks, budget and resource declarations, the shortcut declaration, `state_factory` and the
  structure hash. Its first start freezes it and every flow under it; a builder method on a
  frozen flow raises.
- `FlowRunner(lg, platform=, store=).start(flow, input, ...)` checks the run's inputs, starts the
  run in a task of its own and returns it. `ctx.platform`, `ctx.traits` and `ctx.saia` come from
  the runner's platform. In the target design an agent supplies a run's identity and platform; an
  agent may start any number of runs, of different flows or of the same flow.
- State is supplied per run, never held by the flow. `state_factory` stays with the flow: how a
  run's state is made and restored is structure.
- A checkpoint is a run's durable record: scopes, cursors, resource accounting and the structure
  hash. Halt, cost tracker, SAIA, traits and logger are never serialized; they reattach from
  whoever starts the resumed run, and from its platform.
- The walk keeps its position in per-execution objects (chain step, iterate pass, map items), not
  on the flow's nodes, so the same nodes serve every run.

Cloning a flow per run would keep structure and state coupled: every field added to `Flow` would
need a copy-or-share decision, and nothing would stop two runs from sharing a clone.

Settings live where their lifetime is:

| Setting | Where |
|---|---|
| `state_factory`, flow class, resource methods | The flow `Factory` |
| The `Platform` (trait registry, `SAIAFactory`, SAIAs), CAS store | The runner |
| Halt, signals, history name, start, cost tracker | `start()` |
| Declarations about part of the tree: checkpointer, checkpoint policy, shortcut, budget, resource instances (`with_resource`) | The flow |

A run's history is named by the caller (`history=`); in the target design the name joins the
agent's identity, so runs of a shared flow keep separate histories and `start=Resume("latest")`
continues the caller's own run.

Hooks follow the trait model: one shared instance, per-run data keyed by the event's ids, the
identity read from the event. Events carry `ctx`, so a hook reads `ctx.agent.identity`: the "who"
for provenance and authorization. Hooks are attached to the flow (`Flow.with_hooks`).

### Platform and the generic run

The `Platform` is what every run on it uses: the `SAIAFactory`, the trait registry and the shared
clients. It builds one SAIA per role on first use and keeps it, so every run on one platform
shares its SAIAs. That is safe because a SAIA holds configuration only (section 4,
Conversations). Differences between roles go through `Role` and `SAIAFactory.build(role)`, which
may choose tools and directive by role. The runner holds a platform; every run it starts, and
every step of every flow in it, uses that one.

The generic `Run` is what any execution has, with or without Flow: its logger, its platform,
its cost tracker, its halt event and the task that executes it. It is awaited for its result,
halted with `halt()`, and reports `done`. A flow run extends it with what exists only because a
flow nests and checkpoints its scopes: the flow, its signals, its history writer, its live scopes
and cursors, and its hook bookkeeping. Without Flow there is one scope, the root, and nothing to
fold.

### Runner and run

A runner holds the wiring its runs share and executes them; a run is one execution:

```python
platform = PlatformFactory(lg).with_saia_factory(factory).from_config({})
runner = FlowRunner(lg, platform=platform, store=store)
run = runner.start(
    flow,
    "input",  # the first node's arguments
    state=s,  # the root scope's state
    cost=tracker,  # the run's spend rolls up here
    halt=h,  # the run's halt
    history="k",  # the repository of the runner's store the run's commits go to
    start=Resume("latest"),  # Fresh() (default), Resume(at) or Restart(at)
)
result = await run
```

`start()` runs every check before anything is read or written, so a misconfigured run raises
there and leaves no history behind. `Start` is used once, to produce the run's initial state and
snapshot: `Fresh()` from `state`, `Resume(at)` continuing where the run was at a commit,
`Restart(at)` from the commit's root state with every position cleared; `at` is `"latest"`, a
commit hash or a checkpoint name. A run is awaited for its result (`HALTED` when its halt stopped
it), halted with `run.halt()`, and reports `done`; cancelling the task that awaits it cancels it.

### Cost

A cost tracker is the run's, never the flow's: one on the structure would be shared by every run
of it. `start(flow, cost=tracker)` gives the run a tracker of its own, `tracker.child()`, as the
root's `COST` resource; `with_budget(limit)` gives each run of a flow a child of the enclosing
tracker with that budget. Spend rolls up the chain, and `exceeded`, `urgent_wrapup` and
`remaining` read every budget on it, so a run under an exhausted agent or session tracker sees it.

The run's own tracker is saved with the top-level flow's resources, through the completion commit,
and a resumed run restores it into a new tracker of its own. The caller's tracker is never
restored into: it sees only what the runs it is given to spend, while the run's tracker holds the
history's total. A budgeted run's child is saved while that run is in progress.

### Scopes: the fold

The scope in effect at a node (state, resources, checkpointer, checkpoint policy, hooks, shortcut
regions) is the root scope folded along the path to it. Each descent has two parts: the node the
walk goes through (a subflow call, a branch arm, an iterate pass, a map item) takes a step, then
the flow it enters applies its declarations:

```
scope(node) = root · step(n₁) ⊕ decl(f₁) · step(n₂) ⊕ decl(f₂) · … · step(nₖ) ⊕ decl(fₖ)
```

`n₁ … nₖ` are the nodes and `f₁ … fₖ` the flows on the path from the root to `node`. A step moves
the walk's position (snapshot path, chain context, ancestor chain) and may change the scope: it
hands the flow its state, projected or the enclosing one, and a map item leaves its drain regions
(a started item runs to its end).

A declaration (`Decl`, read off a flow by `Flow.decl`) is a partial scope: what one flow declares
for its subtree, any field unset. Each field defines its own `⊕`:

| Combine | Fields | Rule |
|---|---|---|
| Override | checkpointer, checkpoint policy | the inner declaration wins; unset inherits |
| Accumulate | shortcuts | the inner regions add to the enclosing ones |
| Derive | resources, hooks | the inner value is the enclosing value's `child()`, then the flow's own |

Two fields combine rules. Resources are overridden by key with the flow's own, then derived: the
per-run children the flow asks for are made from what is in scope. Hooks derive, unless the flow
replaces them: then its own hooks override the inherited ones.

State is the exception to `⊕`: a subflow projects its state from the enclosing state on the way in
and merges it back on the way out.

The values are pure: the scope at a node is a function of the root scope, the structure and the
path, the same rule at every depth, and rebuilt from the structure on resume. Entering a scope
also has effects, kept apart from the values: the resources a flow runs with are restored from the
checked-out snapshot and registered with the run while the flow walks, so its checkpoints hold
their accounting. A field that fits none of the rules is either fixed for the whole run (as
`extra` is) or needs a rule of its own named here.

The walk carries a frame from node to node: the run (fixed for the whole run), the scope (the
fold so far) and the position (snapshot path, chain context, ancestor chain). A step sees it as
its `Context`.

### Packages

```
core/       what any run has and runs on: Run, Platform, Role, SAIAFactory
flow/       structure: the builder and its read-only tree, nodes, verbs, state, the declaration
            type (Decl), Context and Loop
flow/run/   execution: the runner (FlowRunner) and Start, Scope and Position, the frame, the
            walk (chain, branch, iterate, map), the flow run, start and resume, history writing,
            hook events, shortcut regions
```

`flow/run` imports the structure; the structure never imports `flow/run`. Context and Loop reach
the run through the frame they are handed.

## 7. Runtime

Each agent is an appinfra `AsyncService` on its worker's `LoopHost`: one task per agent, no thread
held while it waits. All agents in a worker share one `Platform`; the hub maps an agent name to
its runner.

Measured with `tests/performance/test_flow_runs.py` at 10,000 and 100,000 concurrent runs of a
small flow (a scripted model answering in 50-500 ms): one event loop handles about 2,200-2,800
model calls per second at about 50 KB per run in flight, linear in the number of runs. SAIA and
llm-infer take about three quarters of the CPU, the flow walk about an eighth. More event loops on
threads of one process add no throughput under the GIL; more processes do, so a host scales by
running one worker process per core.

## 8. API

Without Flow:

```python
async with PlatformFactory(lg).from_config({}) as platform:
    agent = AgentFactory(platform).create("hello", directive="You are a concise assistant.")
    print(await agent.ask("Say hello."))
```

With Flow:

```python
@verb(role=Role("assistant", style="You are a concise assistant."))
async def hello(ctx, text: str) -> str:
    return (await ctx.saia.instruct(text)).value


async with PlatformFactory(lg).from_config({}) as platform:
    flow = FlowFactory(lg).create("hello").call(hello)
    print(await FlowRunner(lg, platform=platform).start(flow, "Say hello."))
```

A model of the caller's own, in code:

```python
llm = LLMFactory(lg).openai(base_url="http://localhost:8000/v1", default_model="my-model")
platform = PlatformFactory(lg).with_llm(llm).from_config({})
```
