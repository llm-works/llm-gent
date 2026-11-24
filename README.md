# llm-gent

Agent framework with trait-based architecture and learning capabilities.

## Overview

llm-gent provides a composable framework for building LLM-powered agents. Agents are composed of
**traits** that provide specific capabilities (LLM access, storage, learning, etc.) and can be
run standalone or as services via the included HTTP runtime.

Key features:

- **Trait-based composition** - Mix and match capabilities via traits (LLM, Storage, Rating, Learn)
- **Multi-backend LLM support** - OpenAI-compatible, Anthropic, and custom backends via llm-infer
- **Built-in learning** - Collect training data (SFT/DPO) and fine-tune via llm-kelt
- **Structured output** - Pydantic schema validation with automatic JSON cleanup for small models
- **Production ready** - HTTP server, PostgreSQL storage, schema migrations

## Installation

```bash
pip install llm-gent
```

For HTTP server support:

```bash
pip install llm-gent[http]
```

## Supported Python versions

CI tests against Python **3.11**, **3.12**, **3.13**, and **3.14** on every
push. `requires-python = ">=3.11"`.

## Quick Start

```python
from appinfra.log import create_lg

from llm_gent import PlatformFactory
from llm_gent.agents.default import Factory as PromptAgentFactory

lg = create_lg("my-agent", "info")

llm = {"backends": {"local": {"base_url": "http://localhost:8000/v1", "model": "default"}}}
with PlatformFactory(lg).from_config({"llm": llm}) as platform:
    agent = PromptAgentFactory(platform).from_config(
        {
            "identity": {"name": "my-agent"},
            "directive": "You are a helpful assistant.",
            "tools": {"complete_task": {}},
            "chat": {},
        }
    )
    agent.start()
    print(agent.ask("Hello!").content)
    agent.stop()
```

The platform builds the model router once from the packaged `etc/llm.yaml`, with the `llm`
settings merged on top, and every agent built on it shares that router. The agent's directive is
the system prompt of its SAIA, which answers by calling `complete_task`. The `chat` section
makes the agent answer questions; without it the agent has no `ask`. A runnable version of
this example lives at `llm_gent/examples/quickstart.py`. Pass `--smoke` to run it on llm-infer's
scripted model (no real backend contacted).

## Core Concepts

### Agents

An `Agent` is a container for traits with lifecycle management. Agents have an identity
(`name` plus optional `context_key`) and can be started, stopped, and — via
`RunnableAgent` — run in cycles.

### Traits

Traits provide specific capabilities to agents:

| Trait | Purpose |
|-------|---------|
| `DirectiveTrait` | System prompts and agent instructions |
| `StorageTrait` | PostgreSQL persistence with migrations |
| `RatingTrait` | Automated LLM-based content evaluation |
| `LearnTrait` | Training data collection (SFT/DPO) |
| `ToolsTrait` | Tool/function calling support |

### Tools

Built-in tools for agentic workflows:

- `ShellTool` - Execute shell commands
- `FileReadTool` / `FileWriteTool` - File operations
- `HTTPFetchTool` - HTTP requests
- `RecallTool` / `RememberTool` - Memory operations

## Running as a Service

```bash
# Start agent server (uses packaged default config)
llm-gent serve

# With custom config
llm-gent --config path/to/config.yaml serve
```

## Related Projects

- [llm-infer](https://github.com/llm-works/llm-infer) - LLM inference server and client
- [llm-kelt](https://github.com/llm-works/llm-kelt) - Training infrastructure (SFT/DPO)
- [appinfra](https://github.com/llm-works/appinfra) - Application infrastructure utilities

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
