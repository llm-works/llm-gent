# Design: agents at scale

How llm-gent runs many agents in one process with nothing duplicated per agent: what is shared,
what each agent holds, and how flows run for agents.

Status: the target design, built in steps. Implemented: the platform package (`Config`,
`Factory`, `Context`) with the shared clients injected into traits, and agents built on a
platform; the scope a flow runs in as a fold of its declarations, the generic run and its
environment, and flow runs started by a runner (section 6). Not yet: the platform's `SAIAFactory`,
shared trait instances with per-agent state, `platform.with_trait_registry`, agents supplying a
run's environment and identity, and the API in section 8.

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
| Platform | The shared clients (model router, database pool, memory clients), the trait registry, the tool registry, the `SAIAFactory` | One per worker |
| Trait | Shared logic and resources | One per trait type per registry, used by every thread and event loop |
| Agent | Identity, per-agent trait state, its runs | One per identity |
| Flow | Structure: what happens to state | One per definition, shared by every agent |
| Run | One run's state: flow state, halt, signals, run info, checkpoint context | Many per agent, held by it |

State is kept apart from the logic that acts on it. Traits and flows are logic and structure,
shared; agents and runs are state, cheap and many.

## 3. Construction

Factories build; the context they produce is passed around and builds nothing. A class is named
without its package's name inside the package (`platform.Factory`, `platform.Context`,
`agent.Factory`, `traits.Factory`); flat namespaces such as the top-level package export it with
the prefix (`PlatformFactory`, `AgentFactory`, `TraitFactory`).

| Object | Role |
|---|---|
| platform `Factory(lg).from_config(config)` | Builds the shared clients, the traits (through `TraitFactory`), the tool registry and the `SAIAFactory`, and injects them into a platform `Context` with ownership. Reads the platform `Config`. |
| platform `Context` | Container passed around. Holds references and closes what it owns on `close()` or at the end of a `with` or `async with` block. |
| `TraitFactory` | Builds traits from injected clients; holds no reference to the context. |
| `AgentFactory(platform)` | Builds agents that reference the context. Subclassed per agent type (`agent_class`, `required_traits`, `default_tools`). |
| flow `Factory(lg)` | Builds flows: structure only (state factory, flow class, declared resources). |
| `FlowRunner(lg, env=, store=)` | Executes flows. Holds what its runs share: the `Environment` (trait registry, `SAIAFactory`) and the checkpoint store. `start()` takes one run's inputs and returns the run. |

`AgentFactory` does not build the platform: several agent types share one worker, and the
platform's lifecycle belongs to the worker. Factories build clients and inject them; a receiver
closes only what it is told it owns.

Models are wired in two ways:

- Config: the platform `Config.llm` is llm-infer's router config, merged over the packaged
  `etc/llm.yaml`. An empty config uses the packaged backends.
- Injection: `PlatformFactory(lg).with_llm(client)` takes a client the caller built, for example
  with llm-infer's `Factory(lg).openai(...)`. The caller keeps ownership.

The platform's `SAIAFactory` builds SAIA over the platform's llm-infer router and reads the `Role`
fields: `style` becomes the system prompt, `backend` and `model` select the model.

Platform `Config` defaults: `llm` is the packaged `etc/llm.yaml`; `memory` is optional, and without
it no memory-backed trait or tool is available. An agent's only required field is its name.

## 4. Traits

Every capability is a trait. What a trait holds splits in two:

- **The trait (shared):** clients, pools, caches, config. One instance per trait type per
  registry, built by `TraitFactory`.
- **Per-agent state:** what differs per agent: directive text, conversation history, budget,
  Memory's identity-scoped view. The trait defines its state type and creates it per agent; the
  agent holds these state objects keyed by trait. Calls pass the agent:
  `llm.complete(agent, messages)`, `conversation.add_turn(agent, user, reply)`.

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
| Conversation | Compaction config and compactor | Message history |
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

The Conversation trait's per-agent history is the agent's own chat: one sequence per agent, for
an agent that holds one conversation at a time.

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
authorization. Every run has an agent, with and without Flow.

An agent holds:

- its identity;
- per-agent state objects, keyed by trait, created only for the traits it uses;
- a reference to its platform `Context`;
- the trait types it uses (`required_traits`), not trait instances;
- its active flow runs, keyed by run id.

No trait instance lives on an agent. `agent.get_trait(T)` and `require_trait(T)` resolve through
the platform's registry. A custom trait is registered with the platform like a built-in one, even
when one agent uses it: the platform owns its lifecycle and it follows the thread-safety contract.
A resource an agent manages itself is plain agent code, an attribute of the agent subclass that
the agent closes, not a trait.

A registry holds one instance per trait type; differences between agents go in per-agent state or
call arguments, not in a second instance. Agents or a swarm that need other instances get their
own registry through `platform.with_trait_registry(registry)`: a shallow copy of the context that
shares the clients, the tool registry and the `SAIAFactory`, and closes only the registry it was
given. A registry is built whole by `TraitFactory`, which wires the dependencies between traits
(Rating and Storage take Memory's client and the LLM router); swapping one trait into a copy of
another registry would leave its dependents on the old one. Platform components (`SAIAFactory`,
tools) hold no trait instances; they resolve traits through the agent's context on each call.

## 6. Flow: structure and runs

A flow is defined once and runs many times at once, for many agents. It never runs itself: a
runner executes it, and each execution is a run.

- `Flow` is shared and immutable once started: nodes, verbs, checkpointer and checkpoint policy,
  hooks, budget and resource declarations, the shortcut declaration, `state_factory` and the
  structure hash. Its first start freezes it and every flow under it; a builder method on a
  frozen flow raises.
- `FlowRunner(lg, env=, store=).start(flow, input, ...)` checks the run's inputs, starts the run in
  a task of its own and returns it. `ctx.env`, `ctx.traits` and `ctx.saia` come from the runner's
  environment. The target design takes the environment and the run's identity from the agent,
  which holds its runs keyed by run id; an agent may have any number of runs, of different flows
  or of the same flow.
- State is supplied per run, never held by the flow. `state_factory` stays with the flow: how a
  run's state is made and restored is structure.
- A run leaves the agent when it ends. A checkpoint is its durable record: scopes, cursors,
  resource accounting and the structure hash. Halt, cost tracker, SAIA, traits and logger are
  never serialized; they reattach from the agent and the run at resume.
- The walk keeps its position in per-execution objects (chain step, iterate pass, map items), not
  on the flow's nodes, so the same nodes serve every run.

Cloning a flow per run would keep structure and state coupled: every field added to `Flow` would
need a copy-or-share decision, and nothing would stop two runs from sharing a clone.

Settings live where their lifetime is:

| Setting | Where |
|---|---|
| `state_factory`, flow class, resource methods | The flow `Factory` |
| Trait registry, `SAIAFactory` (the `Environment`), CAS store | The runner; in the target design the environment comes from the agent |
| Halt, signals, resource instances (cost tracker), history name, start | `start()` |
| Declarations about part of the tree: checkpointer, checkpoint policy, shortcut, budget | The flow |

A run's history is named by the caller (`history=`); in the target design the name joins the
agent's identity, so runs of a shared flow keep separate histories and `start=Resume("latest")`
continues the caller's own run.

Hooks follow the trait model: one shared instance, per-run data keyed by the event's ids, the
identity read from the event. Events carry `ctx`, so a hook reads `ctx.agent.identity`: the "who"
for provenance and authorization. Hooks are attached to the flow (`Flow.with_hooks`).

### Environment and the generic run

An `Environment` is what a run may use: the `SAIAFactory` and the trait registry. It owns nothing;
it selects what a run uses of the platform's clients. It builds one SAIA per role on first use and
keeps it, so everything that uses one environment shares its SAIAs. That is safe because a SAIA
holds configuration only (section 4, Conversations). Work that needs SAIAs of its own (its own
tools, its own directive) uses an environment of its own. The runner holds one; every run it
starts, and every step of every flow in it, uses that one.

The generic `Run` is what any execution has, with or without Flow: its logger, its environment,
its halt event and the task that executes it. It is awaited for its result, halted with `halt()`,
and reports `done`. A flow run extends it with what exists only because a flow nests and
checkpoints its scopes: the flow, its signals, its history writer, its live scopes and cursors,
and its hook bookkeeping. Without Flow there is one scope, the root, and nothing to fold.

### Runner and run

A runner holds the wiring its runs share and executes them; a run is one execution:

```python
runner = FlowRunner(lg, env=Environment(saia_factory=factory, traits=registry), store=store)
run = runner.start(
    flow,
    "input",  # the first node's arguments
    state=s,  # the root scope's state
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
fold so far) and the position (snapshot path, chain context, ancestor chain).

### Packages

```
core/       what any run has and may use: Run, Environment, Role, SAIAFactory
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
held while it waits. All agents in a worker share one platform `Context`; the hub maps an agent
name to its runner.

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
    agent = AgentFactory(platform).create("hello")
    print(await FlowFactory(lg).create("hello").call(hello).run("Say hello.", agent=agent))
```

A model of the caller's own, in code:

```python
llm = LLMFactory(lg).openai(base_url="http://localhost:8000/v1", default_model="my-model")
platform = PlatformFactory(lg).with_llm(llm).from_config({})
```
