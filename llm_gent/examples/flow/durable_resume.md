# Durable pause / resume with real inference

`durable_resume.py` halts a SAIA turn between a model tool call and the model's follow-up
completion, exits the process, and resumes that same turn from disk in a second process. This
walkthrough follows one real run against a local OpenAI-compatible server and shows what lands in
the checkpoint store at each step.

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
`~/.cache/llm-gent-durable-resume`. Every invocation calls `flow.run(resume=True)`; the framework
starts fresh on an empty store or when the latest commit is the `$complete` marker, and otherwise
resumes from the latest commit (here always the halt commit). A third invocation therefore starts
a new cycle; `--reset` wipes the store first.

## The flow

```
flow ─ .iterate(until=not pending, max_iters=3)     node e62e0596ecc27e3c
        └─ body ─ .call(summarize)                  node 0b799bf8f1f3498c
                     └─ Loop(summarizer) ─ saia.complete(task)
                          tools: lookup_reference, submit_summary (terminal)
```

- State is a `Digest` dataclass: `pending` (topics left) and `summaries`.
- Each iteration summarizes `pending[0]`. On a completed turn the verb pops the topic and appends
  the summary; on a paused turn it returns without touching state.
- The tool executor sets the flow's halt event when the first `lookup_reference` returns. SAIA's
  next `Backend.chat` sees the event through `abort_signal`, raises `PauseRequested`, and
  `saia.complete` returns `TaskResult(paused=True)`.

Node ids are blake2b hashes of each node's position, kind, and target within the flow, so they are
stable across processes and runs of the same script — which is what lets run 2 find run 1's
commits. Changing the flow's shape between runs changes the ids and orphans the old commits.

## Run 1 — fresh, halts mid-turn

```
  model: qwen3.5-27b-gptq-int4 @ http://127.0.0.1:18300/v1
--- Run (fresh, real) ---
  store: /home/ubuntu/.cache/llm-gent-durable-resume
  tool: lookup_reference('content-addressed storage')
  paused mid-turn on 'content-addressed storage'
  pending: ['content-addressed storage', 'async cancellation']
  summaries: []
  ref files: ['durable-resume-demo/refs/e62e0596ecc27e3c/1.json']
  halted mid-turn (paused_turn saved) — invoke again to resume
```

Sequence inside the process:

1. `Loop` dispatches with no caller conversation, so it hands SAIA a fresh one from its
   `conversation_factory`. SAIA appends each message of the turn to it.
2. The model calls `lookup_reference`; the executor returns the blurb and sets halt.
3. SAIA's follow-up `chat` raises `PauseRequested`; `saia.complete` returns paused.
4. `Loop` serializes `{"task", "conversation"}` and stashes it on the runtime, keyed by the node
   id of the step that dispatched it (`0b799bf8f1f3498c`, the `.call(summarize)` step).
5. `summarize` sees `result.paused` and returns with state untouched.
6. The iterate body finishes; `IterateRunner` increments its counter to 1, checks `until` (topics
   remain), then observes halt at the top of the next pass and writes the halt commit — draining
   the stashed turn into a `paused_turn` blob referenced from that commit.

## Run 2 — resumes the paused turn

```
--- Run (resume, real) ---
  summarized 'content-addressed storage' -> 'Content-addressed storage is a system where data is
    retrieved using a unique identifier derived from its own content, rather than its physical
    location.'
  tool: lookup_reference('async cancellation')
  summarized 'async cancellation' -> 'Async cancellation is a foundational concept in modern
    systems software that allows for the interruption and cleanup of ongoing asynchronous
    operations.'
  pending: []
  complete — $complete marker stamped; next invocation starts fresh
```

The first topic is summarized with no `lookup_reference` line: the model's first completion in
run 2 is the one run 1 never got to make. Sequence:

1. `Resume.hydrate` resolves the latest commit (the halt commit), restores `Digest` from its tree
   through `TypeStateFactory(Digest)`, and loads each `paused_turn` blob as a resume entry keyed by
   the dispatching step's node id.
2. `IterateRunner` sees it is the save-point leaf and fast-forwards its counter to 1.
3. `summarize` runs for `pending[0]` — still the halted topic, because run 1 left state untouched.
4. `Loop` finds its resume entry, rebuilds the conversation with
   `conversation_factory.create_from_state`, restores the saved task, and calls
   `saia.complete(..., resume=True)`. SAIA continues from the tool result; the model calls
   `submit_summary`.
5. The next pass handles the second topic normally (counter 2 → 3, within `max_iters=3`). `until`
   fires on the empty queue.
6. Clean exit under the default `retain` retention stamps a `$complete` marker.

## The on-disk store

After both runs (`~/.cache/llm-gent-durable-resume/`):

```
durable-resume-demo/                       # URL-quoted client_flow_id
├── _seq                                   # "2": monotonic ref counter; newest ref wins
├── refs/
│   ├── e62e0596ecc27e3c/1.json            # node_path / iteration -> halt commit (seq 1)
│   └── %24complete/0.json                 # "$complete" / 0 -> completion marker (seq 2)
└── objects/                               # content-addressed, file name = blake2b hash
    ├── commit/8f3ce969…                   # halt commit
    ├── commit/bafb01ab…                   # completion marker commit
    ├── tree/614ab218…                     # halt commit's tree
    ├── tree/c2c013c8…                     # empty tree (completion marker)
    ├── blob/e6e66548…                     # Digest state at halt
    └── blob/8f339b74…                     # paused_turn envelope
```

Only `refs/` is mutable. Everything under `objects/` is immutable and named by the hash of its
bytes; identical content is stored once. If resume finds the commit, tree, or a state blob
missing, it falls back to a fresh run; a missing `paused_turn` blob makes only that Loop restart its
turn from the task.

### Refs

```json
// refs/e62e0596ecc27e3c/1.json
{"commit_hash": "8f3ce969…", "seq": 1}
```

A ref maps `(node_path, iteration)` to a commit. `node_path` is the `/`-joined chain of node ids
from the run root to the save site; here the save site is the top-level iterate, so it is one id.
`resolve_ref(client_flow_id)` returns the ref with the highest `seq`, which is how resume picks the
latest commit.

### Halt commit

```json
{
  "meta": {
    "client_flow_id": "durable-resume-demo",
    "node_path": "e62e0596ecc27e3c",
    "iteration": 1,
    "outcome": "halted",
    "produced_by": {"node_id": "e62e0596ecc27e3c", "...": null},
    "trace_ref": [
      {"kind": "paused_turn", "id": "0b799bf8f1f3498c:8f339b74…"}
    ]
  },
  "parent_hashes": [],
  "root_tree_hash": "614ab218…"
}
```

- `iteration: 1` — the iterate counter at halt. The paused pass counted as an iteration, which is
  why the flow bounds `max_iters` at `len(TOPICS) + 1` and terminates on `until` instead.
- `outcome: "halted"` — written by the halt-observation site. Resume does not branch on it: any
  latest commit other than `$complete` is resumed, and the script applies the same rule
  (`node_path != "$complete"`) to decide whether to arm the halt on the next invocation.
- `trace_ref` — one entry per paused dispatch, `"<step node id>:<blob hash>"`. Resume hands the
  blob back to the Loop called from that step. The key is per step, not per Loop: a verb that
  calls two Loops that can pause would have them overwrite each other's entry, so keep one
  resumable Loop per step.

### Tree and state blob

```json
// tree/614ab218…
[["00", "blob", "e6e66548…"]]

// blob/e6e66548…
{"pending": ["content-addressed storage", "async cancellation"], "summaries": []}
```

The tree holds one blob per state scope, keyed by two-digit depth. This flow has only the root
scope. The state blob is `Digest.to_dict()` as canonical JSON — both topics pending, no summaries:
the paused pass did not mutate state.

### Paused-turn blob

```json
{
  "task": "Summarize the following. term: content-addressed storage",
  "conversation": {
    "messages": [
      {"role": "user",      "content": "Summarize the following. term: content-addressed storage"},
      {"role": "assistant", "content": "",
       "tool_calls": [{"id": "…", "name": "lookup_reference",
                       "arguments": {"term": "content-addressed storage"}}]},
      {"role": "tool",      "tool_call_id": "…",
       "content": "'content-addressed storage' is a foundational concept in modern systems software."}
    ],
    "...": "…"
  }
}
```

This is the turn exactly at the pause: task, the model's tool call, and the tool result — nothing
after. `conversation` is the `to_dict()` payload of the conversation class the Loop's
`ConversationFactory` produces (`llm_kelt.conversation.Conversation` here); the same factory's
`create_from_state` rebuilds it on resume.

The state blob and the paused-turn blob are separate on purpose: state is the flow's data at the
save site; the paused-turn blob is the in-flight model turn. Resume needs both.

### Completion marker

```json
{"meta": {"node_path": "$complete", "iteration": 0, "outcome": "ok", "trace_ref": []},
 "root_tree_hash": "c2c013c8…"}   // empty tree
```

Stamped on clean exit when the store's retention is `retain` (the `JsonFileCheckpointStore`
default). `run(resume=True)` treats a history whose latest commit is this marker as a fresh
start instead of replaying the old halt commit. With `retention="gc_on_success"` the history is
deleted instead.

## Save sites

| Save site | Fires here? | Why |
|---|---|---|
| Iterate halt observation (top of each pass) | Yes — run 1 | Halt set during pass 1; observed before pass 2. |
| Iterate boundary, `outcome="ok"` | No | `CheckpointPolicy.on_iterate` is off by default; enable with `flow.with_checkpoint_policy(CheckpointPolicy(on_iterate=True))`. |
| Chain between-step / trailing halt | No | Only the top-level chain saves; the body chain is nested, so halt propagates to the iterate boundary where iteration state is consistent. |
| Completion marker | Yes — run 2 | Clean exit, `retain` retention. |
| Explicit `ctx.checkpoint()` | No | The verb does not call it. |

## Two contracts the example depends on

**Wire a `ConversationFactory` on the Loop.** It supplies the conversation SAIA appends the turn
to, and rebuilds it on resume. Without one, a pause still writes a halt commit, but with no
`paused_turn` ref, and resume re-runs the turn from the task.

**Do not mutate state on a paused result.** The halt commit snapshots state after the verb
returns. A verb that pops its input or appends a placeholder on a paused `TaskResult` checkpoints a
half-applied iteration; on resume the Loop restores the saved task while the verb reads the next
item, and the halted item is skipped. `summarize` checks `result.paused` before touching
`ctx.data`.

## What `--smoke` verifies

A scripted `saia.Backend` stands in for the model; SAIA, Loop, Flow, and the store are real. Each
phase rebuilds store, factory, backend, and flow, so only disk carries over. The run fails unless:

- phase 1 halted with state untouched,
- the halt commit carries a `paused_turn` ref,
- phase 2 issued exactly one `lookup_reference` (the second topic only — a restarted first turn
  would make it two),
- phase 2 drained every topic with non-empty summaries.
