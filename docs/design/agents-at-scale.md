# Design: agents at scale

How llm-gent runs many agents in one process with nothing duplicated per agent: what is shared,
what each agent holds, and how flows run for agents.

Status: the target design, built in steps. Implemented: the platform package (`Config`,
`Factory`, `Context`) with the shared clients injected into traits, and agents built on a
platform. Not yet: the platform's `SAIAFactory`, shared trait instances with per-agent state,
`platform.with_trait_registry`, the flow/run split, and the API in section 8.

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
| flow `Factory(lg)` | Builds flows. A run takes its agent; the trait registry and `SAIAFactory` come from the agent's context. |

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

A flow is defined once and runs for many agents.

- `Flow` is immutable after build and shared: nodes, verbs, checkpointer and checkpoint policy,
  hooks, budget and resource declarations, the shortcut declaration, `state_factory`, the
  structure hash, and the per-role SAIA cache (keyed by SAIA factory and role, since agents of
  different platforms may run the same flow).
- `flow.run(input, agent=agent)` requires the agent and creates a Run held by it, keyed by run id.
  An agent may have any number of runs, of different flows or of the same flow. `ctx` is built
  from the agent and the run; `ctx.traits` and `ctx.saia` come from the agent's context.
- State is supplied per run, never held by the flow. `state_factory` stays with the flow: how a
  run's state is made and restored is structure.
- A run leaves the agent when it ends. A checkpoint is its durable record: scopes, cursors,
  resource accounting and the structure hash. Halt, cost tracker, SAIA, traits and logger are
  never serialized; they reattach from the agent and the run at resume.
- The walk keeps its position in per-execution objects (chain step, iterate pass, map items), not
  on the flow's nodes, so the same nodes serve every run.

Cloning a flow per run would keep structure and state coupled: every field added to `Flow` would
need a copy-or-share decision, each clone would rebuild the SAIA cache, and nothing would stop two
runs from sharing a clone.

Settings live where their lifetime is:

| Setting | Where |
|---|---|
| CAS store, `state_factory`, flow class, resource methods | The flow `Factory` |
| Trait registry, `SAIAFactory` | The run's agent |
| Halt, signals, resource instances (cost tracker), history key | `run()` |
| Declarations about part of the tree: checkpointer, checkpoint policy, shortcut, budget | The flow |

A run names its checkpoint history from the agent's identity and a key the caller gives, so runs
of a shared flow keep separate histories and `resume="latest"` continues the caller's own run.

Hooks follow the trait model: one shared instance, per-run data keyed by the event's ids, the
identity read from the event. Events carry `ctx`, so a hook reads `ctx.agent.identity`: the "who"
for provenance and authorization. Hooks are attached to the flow (`Flow.with_hooks`).

## 7. Runtime

Each agent is an appinfra `AsyncService` on its worker's `LoopHost`: one task per agent, no thread
held while it waits. All agents in a worker share one platform `Context`; the hub maps an agent
name to its runner.

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
