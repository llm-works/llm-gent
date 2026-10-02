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
- Checkpoint / resume. A checkpoint holds `ctx.state.data` and the
  values in flight between steps: the input of the step a chain is at,
  the value an iterate carries into its pass. Those values must be plain
  JSON, pydantic models, or objects with `to_dict()` and a classmethod
  `from_dict()`. The state type is declared via `state_type=` on the
  factory; the initial payload comes from construction or from
  `run(state=...)` override.
- Aggregation across concurrent items. Under `.map(state=proj, merge=fn)`
  each item projects an isolated child payload and folds back through
  `merge` — cross-item accumulation lives in state, not in the returned
  list. Items run concurrently; `merge` fires in completion order, not
  input order. For input-ordered folding, use `aggregate=` instead.

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
- The consumer is >1 step downstream or is a loop/branch predicate →
  state.
- Two consumers of different kinds → both.

### Halt, interruption and resume

A checkpoint is a snapshot of the whole run, as a git commit is of a
repository: every live scope, and the cursor of every running
structure — the step a chain is at and that step's input, the pass an
iterate is in and the value it carries, the arm a branch took, a Loop
call's paused SAIA turn. Resume checks the snapshot out and continues
every structure at its cursor, so **only the step that was running when
the checkpoint was taken runs again**; completed steps and passes do not.

The halt is cooperative. Under it, **a step either completes or is
interrupted**:

- A step that returns has completed; its work is kept.
- A step that stops before finishing its work raises
  `llm_gent.flow.Interrupted` (a `BaseException`, so rescue policies and
  non-strict maps don't treat it as a failure). A Loop call that leaves
  a paused SAIA turn makes its step interrupted without raising.
  `Interrupted` raised while no halt is set is a `RuntimeError`.

When the halt is set, every part of the run stops at the next place the
framework looks — a chain after its step, an iterate before its next
pass, a map item before it starts — however deeply nested and however
many map items run at once; an LLM call in flight finishes (or its SAIA
turn pauses) first. A chain's cursor stays on an interrupted step and
moves past a completed one; a completed last step does not stop the
chain. Chains, iterates and maps that stop before their end raise
`Interrupted` to the step running them, and each part stays registered
where it stopped. Once everything has stopped, `run()` writes one
`halted` commit holding every position and returns `None`: the halted
run's state is in that commit. A run stopped by its own budget ends
the interruption at its boundary, without a commit (see Cost and budgets). A
halted history is never marked complete.

A run has one halt, set with `.with_halt(event)` on its top-level flow;
every subflow observes it. `run()` raises when a nested flow sets a
different event.

A map's cursor is its item list — resolved once, never evaluated again
on resume — and its completed items with their results. On resume a
completed item does not run again and its merge is not applied again, a
running item continues where it was, and items that had not started,
failed or were skipped by the guard run. Items and their results must be
plain JSON, pydantic models, or objects with `to_dict()` and a
classmethod `from_dict()`.

A step that runs again gets the input it had. Consequence: **verbs must
be idempotent-in-effects** at the step level. Reading state, mutating
state, and returning a value are safe to repeat only when the repeated
work is idempotent. Checkpoints preserve mutations already made. Side
effects that are not idempotent — outbound HTTP writes, message sends,
ledger appends — must be
guarded by the verb itself (idempotency keys, "did I already do this"
checks against state or an external record), or split into their own
step so a rerun of a later step does not repeat them.

A `Panel` run from a step is a map over its verbs, at
`<step>/panel/<k>` (the step's `k`-th Panel): a halted Panel keeps its
finished verbs and their results, and resume continues the rest — a
paused Loop turn in a verb resumes mid-turn. Its verbs' results follow
the map items' rule above. Like everything in an interrupted step, a
Panel that finished before the step stopped runs again with it.

Current limits:

- Positions are recorded by node id. A deploy that inserts steps keeps
  them valid; resuming into a flow that no longer has the saved step
  raises, naming its path. Reordering steps can make a step run again.
- Adding `name=` to a step changes its node id (and its descendants').
  A checkpoint at that step fails to resume until the run completes.

### Shortcuts

A halt pauses the run. A shortcut moves it forward: on a signal, a flow
stops exploring and continues at a later step with what it has, and the
run still ends with a result.

Signals are named events the app sets, declared on the top-level flow
with `with_signal(name, event)`; any flow declares a shortcut on one with
`with_shortcut(name, to=None)`, where `to` is the `name=` of a later step
of its own chain (`.call`, `.then`, `.iterate`, `.map` and `.branch` take
`name=`) and `None` its end:

```python
item = (
    ff.create()
    .call(query)
    .then(explore)
    .then(extract, name="extract")
    .then(digest)
    .with_shortcut("cut", to="extract")
)
wave_body = ff.create().call(plan).map(item).then(revise).with_shortcut("cut")
waves = ff.create().iterate(wave_body, max_iters=20).with_shortcut("cut")
campaign = ff.create(...).with_halt(pause).with_signal("cut", cut).call(waves).then(synthesis)
```

When the signal is set, the flow stops exactly as the halt stops it —
every part at its next boundary, a Loop turn paused and held — and then
continues at once from where it stopped, in shortcut mode:

- The step that was running runs again from its positions. A Loop call
  holding a paused turn does not continue it: it returns the result
  SAIA paused it with (SAIA's `TaskResult`, `paused=True`) and the step
  carries on.
- An iterate in the flow's chain starts no new pass and returns its
  carried value; a map in it starts no new item (those are `Skipped`,
  `on_item_complete` fires) and its started items finish.
- The chain then continues at `to`, skipping the steps before it (`to`
  gets the last completed result), or the flow ends with it.

Flows under it run normally unless they declare a shortcut of their own;
in the example above an item stopped before `extract` jumps there, one
stopped inside `explore` finishes its turn with the paused result and
goes on to `extract`, and one past `extract` finishes normally. A flow
that starts while its signal is set — a later step, the next iterate
pass, a map item — starts in shortcut mode; a signal set once a flow is
at or past its `to` does nothing there.

Pausing works at any point of a shortcut. Which signals are set is in
every checkpoint, and so is a flow being in shortcut mode (its chain's
cursor); resume sets the signals again and continues every shortcut
where it was. A finished run records no signal: the next session starts
with none set.

### Cost and budgets

Cost is what calls and operations cost; a cost tracker
(`llm_gent.core.cost.CostTracker`) records it, and a budget is the limit
it is checked against. Verbs reach the run's tracker as `ctx.cost`:
`ctx.cost.spent` is the cost so far, `ctx.cost.budget` the limit.

- `with_cost_tracker(tracker)` — every run of the flow runs on that
  tracker.
- `with_budget(limit)` — each run of the flow runs on a child of its
  tracker (its own, else the enclosing flow's) with that budget; spend
  rolls up, and every budget on the chain applies. A map body runs once
  per item, an iterate body once per pass, a subflow once per `.call`, so
  a budget on the body is a per-item, per-pass or per-call budget. A
  budget needs a tracker on the flow or an enclosing one; `run()` raises
  otherwise.
- Neither — the run shares the enclosing tracker: map items compete for
  it.

```python
flow = (
    ff.create(...)
    .with_cost_tracker(session_tracker)  # the run's tracker
    .map(lambda b: b.with_budget(0.5).call(research), items=topics)  # each item: 0.5
    .call(ff.create().with_budget(3.0).map(...))  # this map: 3.0 in total
)
```

Crossing a budget stops that run with halt semantics — LLM calls in
flight finish, everything under it stops — and it ends with no result
(`None`); the enclosing flow carries on. A budgeted run also stops when
the run's halt is set; then the interruption carries on up as usual.

Every run's own tracker is in each checkpoint taken while the run is in
progress — its spend and spend by op — and is restored when it resumes,
before its first step: a run halted at 9.0 of a 10.0 budget resumes with
1.0 left. A tracker whose restored spend is at or over its budget fires
its halt at once, so a run that ran out stays out until its budget is
raised. A finished run keeps no tracker in its final commit: the next
session starts from the tracker as given.

### One repo per run

A run has one repo: the checkpoint store and history name set once, on
its top-level flow, with `with_checkpoint_store(store, client_flow_id)`
(or `FlowFactory(checkpoint_store=store)` with `create(client_flow_id=...)`).
Every commit the run writes holds the whole run and goes there, wherever
in the flow tree it was taken — committing in a subdirectory commits the
repo. A flow inside a run cannot set a store of its own; `run()` raises.

Saves are declared with `with_checkpointer()`, on any flow: the top
level, a subflow, a map or iterate body. Inside a flow with a
checkpointer on it or above it, `ctx.checkpoint()` and the checkpoint
policy write commits; elsewhere they write nothing. A save belongs to the
innermost checkpointer enclosing the step; `with_checkpointer("research")`
also moves the tag `tags/research` to each of its saves, so
`run(resume="research")` goes back to that part's latest checkpoint. Tag
names are repo-global, shared with `ctx.checkpoint(name)`. A checkpointer
in a run without a store makes `run()` raise.

```python
flow = (
    ff.create(state=...)
    .with_checkpoint_store(store, "campaign-42")  # the run's repo
    .call(plan)
    .map(lambda b: b.with_checkpointer("research").call(item), items=...)
)
```

### Checkpoint cadence

Three save triggers govern when the framework writes commits:

- **Halt observation** — always on with a store. Setting the ambient
  halt event causes the run to save a `halted` commit before returning.
  This is the durability guarantee for pause/resume across process
  restart. A run whose halt is set is never marked complete and never
  deleted under `gc_on_success`.
- **Explicit `ctx.checkpoint()`** — under a `with_checkpointer()`. Verbs
  invoke the async method to force a save; the chain running the verb is
  at its step, so resuming from that save runs the step again.
  `ctx.checkpoint(name)` also tags the save as a named checkpoint (see
  below).
- **Implicit multi-execution boundary saves** — off by default, and
  under a `with_checkpointer()`. Governed by `CheckpointPolicy`:
    - `on_iterate: bool` — save after every iterate body iteration.
    - `on_map_item: bool` — save after every successful map item
      (body-plus-merge). Failed / skipped / cancelled items never
      save regardless of this flag.

Rationale: writes are cheap in aggregate but not free. Long chain
flows with expensive state don't want a commit after every step, and
iterate / map bodies with cheap per-item work shouldn't pay for a
save every pass. The default (halt-only + explicit) minimizes writes
while preserving the pause/resume promise. Single-execution
primitives (`.call`, `.branch`, `Panel`) have no natural per-item
cadence; verbs at those spots use `ctx.checkpoint()` if a save is
wanted.

```python
flow = (
    ff.create(state=...)
    .with_checkpoint_store(store, "history-42")
    .with_checkpointer()
    .with_checkpoint_policy(on_iterate=True, on_map_item=True)
    .iterate(body, max_iters=10)
)
```

Halt-save and `ctx.checkpoint()` are unaffected by the policy — the
policy governs only implicit auto-saves.

On a clean exit under the default `retain` retention, the framework also
commits the run's final state and moves the `complete` tag to it. A
history whose head is that commit is complete; the tag stays on the last
finished run's final state when a later run halts past it. A final state
that cannot be serialized is committed without state (and a warning is
logged) rather than failing the finished run.

A run that raises — or is cancelled — writes no commit: the state at a
failure may be half-updated, so the history's head stays at its last
save, which is where `resume="latest"` continues. The steps that were
running then run again.

### Resume

`run(resume=...)` selects how a run starts from the history:

- `"off"` (default) — start from `state=` as given; commits still append
  to the history.
- `"latest"` — check out the newest commit with state and continue every
  structure at its cursor (see above). Commits without state (a final
  state that could not be serialized) are skipped; walking past them is
  logged at warning level. On a
  finished history the run starts from its first step with the final
  state — the same call starts the next session of a long-lived agent.
  Iteration bounds count across runs.
- `"<name>"` — check out the named checkpoint `ctx.checkpoint("<name>")`
  took, the same way, and move `HEAD` back to it: the run's commits
  continue from there, and the commits written after the checkpoint leave
  the history's line (`"latest"` no longer sees them). `"off"`,
  `"latest"` and `"complete"` cannot name a checkpoint, nor can a string
  of a commit hash's form; an unknown name raises `ValueError` listing
  the names the history has.
- `"<commit hash>"` — the same for any commit of the history.
  `ctx.checkpoint()` returns the hash of the commit it wrote, named or
  not. A commit off the history's line after a reset stays resumable by
  hash until `collect_unreachable` deletes it.

A corrupt history (a commit, tree or blob missing from the store) raises
`HistoryCorrupt` rather than starting over.

Per-session adjustments to the restored state belong in the flow's first
step.

### Reading a history

A history is the chain of commits one `client_flow_id` accumulates across
runs: each commit's parent is the previous head, in time order. A run
ends in a `halted` commit (halt), the `$end` final-state commit (clean
exit), or at its last save when it raised; the next run's commits pick
up there. `History` reads it:

```python
from llm_gent.flow import History, TypeStateFactory

history = History(store, "history-42")
head = await history.head()  # latest commit, or None
if await history.is_complete():  # last run finished
    state = await history.root_state(head, TypeStateFactory(MyState))
done = await history.last_complete()  # final state of the last finished run
mark = await history.checkpoint("before-review")  # a named checkpoint, or None
names = await history.checkpoint_names()  # every checkpoint name, sorted
commit = await history.commit(head.content_hash)  # any commit by hash, or None
snapshot = await history.snapshot(head)  # root scope, child scopes, cursors
async for commit in history.commits():  # newest first, via parent links
    print(commit.meta.node_path, commit.meta.outcome, commit.meta.timestamp_iso)
```

Every commit's meta carries the internal `flow_id` (a UUID the store maps
`client_flow_id` to), the `node_path` of its save point, and
`flow_root_hash` — the structure hash of the flow that wrote it
(`Flow.root_hash()`). Equal root hashes mean identical node ids, so a
history written by one flow can be resumed by the other.

`llm_gent/examples/flow/durable_resume.md` walks through a halted and
resumed run and the store it leaves on disk: refs, commits, and the
snapshot tree with its cursors.

### Cleaning up a history

Objects stay in the store when nothing reaches them any more: the commits
written after a checkpoint once `resume=<name>` or `resume=<hash>` moves
`HEAD` back to it (resumable by hash until collected), and the objects of
a commit whose process died before moving `HEAD` to it. The framework never deletes them on its own;
`collect_unreachable` does, keeping every ref and everything a ref
reaches:

```python
from llm_gent.flow import collect_unreachable

removed = await collect_unreachable(store, "history-42")  # objects deleted
```

Call it while no run writes the history: a run puts a commit's objects
before `HEAD` moves to it, so they would count as unreachable. A history
with a missing object raises `HistoryCorrupt` and nothing is deleted.
`retention="gc_on_success"` still removes a whole history on success.

## Related Projects

- [llm-infer](https://github.com/llm-works/llm-infer) - LLM inference server and client
- [llm-kelt](https://github.com/llm-works/llm-kelt) - Training infrastructure (SFT/DPO)
- [appinfra](https://github.com/llm-works/appinfra) - Application infrastructure utilities

---

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
