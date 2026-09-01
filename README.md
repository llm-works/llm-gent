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

from llm_gent import AgentFactory, LLMTrait

lg = create_lg("my-agent", "info")

agent = AgentFactory(lg).from_config(
    {
        "identity": {"name": "my-agent"},
        "llm": {
            "default": "local",
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                    "model": "default",
                }
            },
        },
        "directive": "You are a helpful assistant.",
        "traits": {"required": ["llm", "directive"]},
    }
)

agent.start()
llm = agent.require_trait(LLMTrait)
result = llm.complete([{"role": "user", "content": "Hello!"}])
print(result.content)
agent.stop()
```

A runnable version of this example lives at `llm_gent/examples/quickstart.py`. Set
`LLM_GENT_SMOKE=1` to run it against a stub LLM router (used by CI's
wheel-smoke job).

## Core Concepts

### Agents

An `Agent` is a container for traits with lifecycle management. Agents have an identity
(`name` plus optional `context_key`) and can be started, stopped, and — via
`RunnableAgent` — run in cycles.

### Traits

Traits provide specific capabilities to agents:

| Trait | Purpose |
|-------|---------|
| `LLMTrait` | LLM completions with multi-backend routing |
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
# Start agent server
llm-gent serve

# Or with specific config
llm-gent -c etc/llm-gent.yaml serve
```

## Related Projects

- [llm-infer](https://github.com/llm-works/llm-infer) - LLM inference server and client
- [llm-kelt](https://github.com/llm-works/llm-kelt) - Training infrastructure (SFT/DPO)
- [appinfra](https://github.com/llm-works/appinfra) - Application infrastructure utilities

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
