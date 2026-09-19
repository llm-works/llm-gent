# llm-gent Documentation

Agent framework with trait-based architecture and learning capabilities.

## Quick Links

- [README](../README.md) - Overview, installation, quick start
- [Flow composition](#flow-composition) - Nodes-are-verbs contract
- [CHANGELOG](../CHANGELOG.md) - Release history

## Core Concepts

### Agents

An `Agent` is the central unit - a container for traits with lifecycle management. Each agent has:

- **Identity** - `name` plus optional `context_key` for namespacing
- **Config** - agent-config dict passed to `AgentFactory.from_config`
- **Traits** - pluggable capabilities

### Traits

Traits provide specific capabilities to agents:

| Trait | Purpose |
|-------|---------|
| `LLMTrait` | LLM completions with multi-backend routing |
| `DirectiveTrait` | System prompts and agent instructions |
| `StorageTrait` | PostgreSQL persistence with schema migrations |
| `RatingTrait` | Automated LLM-based content evaluation |
| `LearnTrait` | Training data collection (SFT/DPO) |
| `ToolsTrait` | Tool/function calling support |

### Lifecycle

```python
from llm_gent import AgentFactory, LLMTrait

agent = AgentFactory(lg).from_config(
    {
        "identity": {"name": "my-agent"},
        "llm": llm_config,
        "traits": {"required": ["llm"]},
    }
)
agent.start()  # Initialize all traits
llm = agent.require_trait(LLMTrait)
result = llm.complete([{"role": "user", "content": "Hello!"}])
agent.stop()  # Cleanup all traits
```

For cycle-driven agents (`agent.run_once()`), use `RunnableAgent` — set
`agent_class = RunnableAgent` on an `AgentFactory` subclass, or subclass
`RunnableAgent` directly.

See the [README quick start](../README.md#quick-start) or
`llm_gent/examples/quickstart.py` for a runnable end-to-end example.

## Flow composition

**Nodes are verbs; state flows via `ctx.state.data`.** A Flow is an ordered
chain of verb calls: each verb receives one input (the previous step's return),
produces one output, and reads or writes shared state through `ctx.state.data`.
There is no implicit multi-input handoff and no per-call verb context — those
are the composition contract, not omissions.

Two consequences follow directly:

- **Multi-input handoff is state-mediated.** A verb that needs data from two
  earlier steps reads both from `ctx.state.data`; earlier steps write there.
  The single-input chain shape stays intact.
- **Per-call verb configuration is state-mediated.** A verb parameterized per
  call reads its parameters from `ctx.state.data` (or from its input). The
  verb's `@verb` decoration stays configuration-free.

### Worked example

Two verbs handing off through state — `plan` writes a target; `execute` reads
it back and returns the count of steps it produced.

```python
from typing import Any
from llm_gent import verb, Context
from llm_gent.flow import FlowFactory
from llm_gent.role import Role

planner = Role(name="planner", backend="anthropic", model="claude-sonnet-4-20250514")
worker = Role(name="worker", backend="anthropic", model="claude-sonnet-4-20250514")


@verb(role=planner)
async def plan(ctx: Context, goal: str) -> str:
    """Compute a plan and stash the target on state for downstream verbs."""
    ctx.state.data["target"] = 3  # e.g. derived from goal
    return goal


@verb(role=worker)
async def execute(ctx: Context, _prev: Any) -> int:
    """Read the target from state, do the work, return step count."""
    target = ctx.state.data["target"]
    return target  # pretend we ran `target` steps


ff = FlowFactory(lg)  # lg: Logger — elided for brevity
flow = ff.create("run", state={}).call(plan).call(execute)
assert await flow.run("ship it") == 3
```

`execute` did not need a two-argument signature; it read what it needed from
`ctx.state.data`. Adding a third downstream verb that also needs `target` is
free — the state is already there.

### Return values vs state

A verb has two output channels — its return value and `ctx.state.data`.
Each feeds a different consumer, and the framework never treats them as
alternatives; a verb may use either or both.

**Return values** thread single-hop to the next node:

- `.call(a).call(b)` — `b` receives `a`'s return as its sole positional,
  bound by signature-aware dispatch. A `b` declared as `async def b(ctx)`
  drops the value; `async def b(ctx, prev)` (or `*args`) consumes it. The
  verb signature is the opt-in — there is no separate builder-level marker.
- `.map(body)` — each per-item result is collected into the result list
  (order preserved). `aggregate=` folds the list; omitted, the list is the
  map node's own return.
- `.branch(when=...)` — the predicate receives the previous node's return
  as `prev_result`; the chosen arm's return is the branch node's own return.
- `.iterate(body, until=...)` — each body return becomes the next
  iteration's input; the last iteration's return is the iterate node's
  own return. `until=(result, ctx) -> bool` sees that return as
  `result`.

**State** persists for the whole run and is the channel for anything
other than a single-hop hand-off:

- Multi-hop handoffs. Two verbs that both need a value produced three
  steps earlier read it from `ctx.data.<field>`; the intermediate steps
  don't need to know it exists.
- Loop and branch predicates. `until=` and `when=` receive `ctx` and
  read `ctx.data.<field>` when the deciding signal isn't in the just-
  returned value.
- Checkpoint / resume. Only `ctx.state.data` (the payload the flow was
  constructed with, `state_type=` on the factory) is serialized;
  return values are transient and don't survive resume.
- Aggregation across concurrent items. Under `.map(state=proj, merge=fn)`
  each item projects an isolated child payload and folds back through
  `merge` — cross-item accumulation lives in state, not in the returned
  list.

**"Return AND mutate" is idiomatic when a value has two consumers.**
Verifier's `judge` returns the verdict (so the chain can carry it) and
also writes `ctx.data.reviews_agree` because the enclosing
`.iterate(until=lambda _r, ctx: ctx.data.reviews_agree, ...)` reads state,
not the return. Structured-agent's `triage` returns the `TriageResponse`
(so `.run()` yields it as the flow's final value) and also writes
`ctx.data.final_triage` because the state slot is what the checkpoint
carries across resume. Neither pattern is redundant — each channel has a
different consumer.

Rubric for picking a channel when only one consumer needs the value:

- Next verb is the only consumer → return; skip the state slot.
- The consumer is >1 step downstream, is a loop/branch predicate, or the
  value must survive resume → state.
- Two consumers of different kinds → both.

## Related Projects

- [llm-infer](https://github.com/llm-works/llm-infer) - LLM inference server and client
- [llm-kelt](https://github.com/llm-works/llm-kelt) - Training infrastructure (SFT/DPO)
- [appinfra](https://github.com/llm-works/appinfra) - Application infrastructure utilities

---

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
