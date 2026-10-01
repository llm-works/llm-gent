# Durable pause / resume with real inference

`durable_resume.py` halts a SAIA turn between a model tool call and the model's follow-up
completion, exits the process, and resumes that same turn from disk in a second process. This
walkthrough follows one run and shows what lands in the checkpoint store at each step.

## Running it

```bash
# Local OpenAI-compatible server (/v1 is appended if missing; model defaults to the first listed)
python -m llm_gent.examples.flow.durable_resume --base-url http://localhost:18300 --reset
python -m llm_gent.examples.flow.durable_resume --base-url http://localhost:18300

# Anthropic instead: omit --base-url, export ANTHROPIC_API_KEY, optionally pass --model
python -m llm_gent.examples.flow.durable_resume --reset
python -m llm_gent.examples.flow.durable_resume

# No network: both phases in one process on a temp store, non-zero exit on a broken round-trip
python -m llm_gent.examples.flow.durable_resume --smoke
```

The two real invocations must be separate processes — that is the point. The store lives at
`~/.cache/llm-gent-durable-resume`. An invocation resumes with `flow.run(resume="latest")` when
the history's head is a halt commit, and starts fresh (`resume="off"`) on an empty history or a
finished one. A third invocation therefore starts a new cycle on the same history; `--reset` wipes
the store first.

## The flow

```
flow ─ .iterate(until=not pending, max_iters=2)     node e62e0596ecc27e3c
        └─ body ─ .call(summarize)                  node 88299db1fcb56568
                     └─ Loop(summarizer) ─ saia.complete(task)
                          tools: lookup_reference, submit_summary (terminal)
```

- State is a `Digest` dataclass: `pending` (topics left) and `summaries`.
- Each pass summarizes `pending[0]`. On a completed turn the verb pops the topic and appends the
  summary; on a paused turn it returns without touching state.
- The tool executor sets the flow's halt event when the first `lookup_reference` returns. SAIA's
  next `Backend.chat` sees the event through `abort_signal`, raises `PauseRequested`, and
  `saia.complete` returns `TaskResult(paused=True)`.

Node ids are blake2b hashes of each node's position, kind, and target within the flow, so they are
stable across processes and runs of the same script — which is what lets run 2 find where run 1
stopped. A checkpoint records positions by node id: resume into a flow that no longer has the saved
step raises, naming the path.

## Run 1 — fresh, halts mid-turn

```
--- Run (fresh, smoke) ---
  tool: lookup_reference('content-addressed storage')
  paused mid-turn on 'content-addressed storage'
  pending: ['content-addressed storage', 'async cancellation']
  summaries: []
  ref files: ['histories/61815f06-…/refs/HEAD']
  halted mid-turn (paused turn saved) — invoke again to resume
```

Sequence inside the process:

1. `Loop` dispatches with no caller conversation, so it hands SAIA a fresh one from its
   `conversation_factory`. SAIA appends each message of the turn to it.
2. The model calls `lookup_reference`; the executor returns the blurb and sets the halt.
3. SAIA's follow-up `chat` raises `PauseRequested`; `saia.complete` returns paused.
4. `Loop` holds `{"task", "conversation"}` in the run's snapshots at its call's path — the first
   Loop call of the `summarize` step in pass 0.
5. `summarize` sees `result.paused` and returns with state untouched. A Loop call left a paused
   turn, so the step counts as interrupted: it stopped before finishing its work.
6. The body chain sees the halt after that step and stops, its cursor on the interrupted step; the
   iterate stops in pass 0. Each stays registered where it stopped. Once the run has unwound,
   `run()` writes the halt commit from those positions and returns `None`: the halted run's state
   is in the halt commit.

## Run 2 — resumes the paused turn

```
--- Run (resume, smoke) ---
  summarized 'content-addressed storage' -> 'A concise concept from systems research.'
  tool: lookup_reference('async cancellation')
  summarized 'async cancellation' -> 'A concise concept from systems research.'
  pending: []
  complete — final state committed and tagged; next invocation starts fresh
```

The first topic is summarized with no `lookup_reference` line: the model's first completion in
run 2 is the one run 1 never got to make. Sequence:

1. `resume="latest"` checks out the newest commit with usable state — the halt commit — and
   restores `Digest` from its snapshot through `TypeStateFactory(Digest)`.
2. The top-level chain continues at the iterate; the iterate continues in pass 0 with its carried
   value; the body chain continues at `summarize` with the input it had.
3. `summarize` runs for `pending[0]` — still the halted topic, because run 1 left state untouched.
4. Its Loop call takes the turn saved at its path, rebuilds the conversation with
   `conversation_factory.create_from_state`, restores the saved task, and calls
   `saia.complete(..., resume=True)`. SAIA continues from the tool result.
5. Pass 1 handles the second topic normally; `until` fires on the empty queue.
6. Clean exit under the default `retain` retention commits the final state and moves the
   `complete` tag to it.

## The on-disk store

After both runs (`JsonFileCheckpointStore`):

```
names/
└── durable-resume-demo                  # client_flow_id -> flow_id
histories/
└── 61815f06-…/                          # flow_id: UUID generated on the first save
    ├── _client_flow_id                  # "durable-resume-demo"
    ├── refs/
    │   ├── HEAD                         # -> final-state commit
    │   └── tags%2Fcomplete              # "tags/complete" -> final-state commit
    └── objects/                         # content-addressed, file name = blake2b hash
        ├── commit/7e55f868…             # halt commit
        ├── commit/0be5929d…             # final-state commit
        ├── tree/…                       # snapshot trees
        └── blob/…                       # values
```

`client_flow_id` is the agent's name for the history; the store keys everything by the internal
`flow_id` it maps to. The UUID and every hash differ from one history to the next, so the ones
shown here will not match a local run.

Only `refs/` is mutable. Everything under `objects/` is immutable and named by the hash of its
bytes; identical content is stored once, so a value that did not change between commits is not
written again. If resume finds an object missing, it raises `HistoryCorrupt`. Objects no ref
reaches any more are deleted only by an explicit `collect_unreachable(store, client_flow_id)`.

### Refs

Refs are named pointers to commits, as in git, moved by compare-and-set: a write lands only while
the ref still points where the writer expects, so a second writer on the same history raises
`ConcurrentWriteError` instead of forking it.

- `HEAD` — the newest commit. Every new commit is parented on it; `resume="latest"` starts its
  walk from it.
- `tags/complete` — the final state of the last run that finished. It stays put while later runs
  append past it (`History.last_complete()`).
- `tags/<name>` — a named checkpoint: `ctx.checkpoint("name")` tags the commit it writes, and
  `run(resume="name")` checks it out; `run(resume=<commit hash>)` checks out any commit (see
  [Named checkpoints](#named-checkpoints)).

The script reads the history through `History(store, CLIENT_FLOW_ID)` rather than the store:
`head()` to decide whether the next run resumes, `snapshot(head)` for the halted state.

### Halt commit

```json
{
  "meta": {
    "flow_id": "61815f06-…",
    "node_path": "e62e0596ecc27e3c/88299db1fcb56568",
    "iteration": 0,
    "outcome": "halted",
    "produced_by": {"node_id": "88299db1fcb56568", "...": null},
    "trace_ref": []
  },
  "parent_hashes": [],
  "root_tree_hash": "208196f5…"
}
```

- `flow_id` — the history's internal id. Commits never carry the agent's `client_flow_id`.
- `parent_hashes: []` — the first commit of the history has no parent. Every later commit's
  parent is the commit that was `HEAD` when it was written.
- `node_path` — where the halt was observed: the `summarize` step inside the iterate. Metadata
  only; where the run continues is in the snapshot's cursors.
- `outcome: "halted"` — written when the halt stopped the run. `resume="latest"` does not branch on it:
  it checks out the newest commit that has state.

### Snapshot

The commit's tree is a snapshot of the whole run: every live scope and the cursor of every running
structure, at paths built from node ids.

```
state/                                   root scope: one blob per top-level key
  pending      ["content-addressed storage", "async cancellation"]
  summaries    []
chain/                                   top-level chain: at the iterate step
  step "e62e0596ecc27e3c"   args []   kwargs {}
n/e62e0596ecc27e3c/
  pass    0                              iterate: in pass 0 ...
  carry   null                           ... with this value carried into it
  until   false                          ... which until did not stop on
  p/0/chain/                             pass 0's body chain: at summarize
    step "88299db1fcb56568"   args [null]   kwargs {}
  p/0/n/88299db1fcb56568/t/0/turn/       the step's first Loop call: its paused turn
    task           "Summarize the following. term: content-addressed storage"
    conversation   {"messages": [user task, assistant tool call, tool result], ...}
```

| Path | Holds |
|---|---|
| `state` | a scope's payload (root, or a `state=` scope at its block's path) |
| `chain` | a running chain's step and that step's input |
| `pass`, `carry`, `until` | a running iterate's pass, the value carried into it, and `until`'s verdict on that value |
| `arm` | the arm a running branch took |
| `t/<k>/turn` | the paused SAIA turn of a step's `k`-th Loop call |
| `items`, `done` | a running map's items and its completed items with their results |
| `p/<pass>/…`, `i/<index>/…` | positions inside an iterate pass, a map item |

The paused turn is the turn exactly at the pause: task, the model's tool call, and the tool
result — nothing after. `conversation` is the `to_dict()` payload of the conversation the Loop's
`ConversationFactory` produces (`llm_kelt.conversation.Conversation` here); the same factory's
`create_from_state` rebuilds it on resume.

A dict value is stored as a tree with one blob per key, so keys that did not change keep their
hash from one commit to the next.

### Final-state commit and the `complete` tag

```json
{"meta": {"node_path": "$end", "iteration": 0, "outcome": "ok", "trace_ref": []},
 "parent_hashes": ["7e55f868…"],   // the halt commit
 "root_tree_hash": "fa7c8a49…"}    // state/ only: Digest with pending [] and both summaries
```

The history is the chain final-state commit → halt commit, written by two different processes:
run 2 read `HEAD` before its first commit and parented on it.

Written on clean exit when the store's retention is `retain` (the default). A finished run has no
running structure, so the snapshot holds the root scope alone. `HEAD` always holds the state the
last run ended with, and `resume="latest"` on a finished history starts from its first step with
that state. If the final state cannot be serialized, the commit is written with an empty tree and
a warning is logged: the history is still complete, but carries no final state. With
`retention="gc_on_success"` the history is deleted instead.

## Where a run stops and continues

**A step either completes or is interrupted.** Under the halt, a step that returns has completed;
a step that stops before finishing its work raises `Interrupted` (a Loop call that leaves a paused
turn counts as interrupted without raising). A chain that sees the halt after a step stops: its
cursor stays on an interrupted step and moves past a completed one. Chains, iterates and maps that
stop early raise `Interrupted` to the step running them, each staying registered where it stopped.
Once everything has stopped — every map item, at any depth — `run()` writes the one halt commit
from those positions and returns `None`.

**Only the interrupted step runs again.** Resume continues every structure at its cursor, so
completed steps and passes do not run again. The step that was interrupted runs again with the
input it had, and a Loop call in it resumes its paused turn.

**Save points.**

| Save point | Fires here? | Why |
|---|---|---|
| Halt checkpoint (once everything stopped) | Yes — run 1 | The halt arrived during `summarize`. |
| Iterate boundary, `outcome="ok"` | No | No `with_checkpointer()`, and `on_iterate` is off by default; enable with `flow.with_checkpointer().with_checkpoint_policy(on_iterate=True)`. |
| Map item, `outcome="ok"` | No | No map here; `on_map_item` is off by default. |
| Explicit `ctx.checkpoint()` / `ctx.checkpoint(name)` | No | The verb does not call it, and it would need a `with_checkpointer()`. |
| Final-state commit (`$end`, tagged `complete`) | Yes — run 2 | Clean exit, `retain` retention. |
| A run that raises | — | Writes nothing: the head stays at the last save, where `latest` continues. |

**Maps.** A running map's cursor (`n/<map>/items`, `n/<map>/done`) holds its item list and its
completed items with their results; each running item keeps its own positions under
`n/<map>/i/<index>/`. On resume a completed item does not run again, a running one continues where
it was, and the rest run.

## Named checkpoints

```python
@verb
async def review(ctx, draft):
    await ctx.checkpoint("before-review")  # tags this commit tags/before-review
    ...


await flow.run(resume="before-review")
```

`ctx.checkpoint(name)` writes a checkpoint like any other — the chain running the step is at that
step — and tags it `tags/<name>`; taking it again moves the tag. `run(resume=name)` checks the
tagged commit out as `latest` checks out the newest one, so the step that took the checkpoint runs
again with the input it had. It also moves `HEAD` back to that commit: the run's commits continue
from there, and the commits written after the checkpoint leave the history's line — `latest` no
longer sees them.

Every commit can be checked out the same way by its hash. `ctx.checkpoint()` returns the hash of
the commit it wrote, named or not, and `History.head()` gives the newest one:

```python
halted = (await History(store, "history-42").head()).content_hash
await flow.run(resume="before-review")  # HEAD moves back; the halted commit leaves the line
await flow.run(resume=halted)  # HEAD moves to the halted commit; the run continues there
```

Moving `HEAD` deletes nothing: a commit off the line stays resumable by hash, and a named one by
its name, until `collect_unreachable` deletes what no ref reaches.

`"off"`, `"latest"` and `"complete"` cannot name a checkpoint, nor can a string of 64 lowercase hex
characters, the form of a commit hash. `run(resume=...)` with a hash or a name the history does
not have raises `ValueError`; for a name, the error lists the checkpoint names the history has
(`History.checkpoint_names()`).

## Budgets across resume

The example wires no budget. With one — `with_budget(tracker)` on the flow, or a cap per item with
`with_budget(cap)` on a map body — each run's tracker is in the halt commit at that run's path
(`budget`: spend so far, and spend by op), and run 2 restores it before the run continues: run 2
spends what is left of the cap, not the whole cap again. A finished run's `$end` commit holds no
tracker, so a fresh cycle starts from the tracker as given. See `docs/index.md`, "Budgets".

## Two contracts the example depends on

**Wire a `ConversationFactory` on the Loop.** It supplies the conversation SAIA appends the turn
to, and rebuilds it on resume. Without one, a paused turn is not captured: the step still counts
as interrupted and runs again, and the turn starts over from the task.

**Do not apply the step's work on a paused result.** The halt commit snapshots state after the
verb returns. A verb that pops its input or appends a placeholder on a paused `TaskResult`
checkpoints a half-applied step; on resume the Loop restores the saved task while the verb reads
the next item, and the halted item is skipped. `summarize` checks `result.paused` before touching
`ctx.data`. Recording the turn's own bookkeeping is fine and is how it survives the halt: a field
of state such as an iteration count, written on the paused result, is in the halt commit and in
`ctx.data` when the step runs again.

## What `--smoke` verifies

A scripted `saia.Backend` stands in for the model; SAIA, Loop, Flow, and the store are real. Each
phase rebuilds store, factory, backend, and flow, so only disk carries over. The run fails unless:

- phase 1 halted with state untouched,
- the halt commit holds the paused turn,
- phase 2 issued exactly one `lookup_reference` (the second topic only — a restarted first turn
  would make it two),
- phase 2 drained every topic with non-empty summaries.
