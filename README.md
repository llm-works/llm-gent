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

## Quick Start

```python
from appinfra import DotDict
from appinfra.log import LogConfig, LoggerFactory

from llm_gent import Agent, Config, Identity, LLMTrait, DirectiveTrait

# Setup logging
log_config = LogConfig.from_params(level="info", handlers={"console": {"type": "console"}})
lg = LoggerFactory.create_root(log_config)

# Configure LLM backend
llm_config = DotDict({
    "default": "local",
    "backends": {
        "local": {
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "model": "default",
        }
    },
})

# Create agent with traits
identity = Identity(domain=None, workspace="demo", name="my-agent")
config = Config(identity=identity)
agent = Agent(lg, config)

# Add capabilities via traits
agent.add_trait(DirectiveTrait(agent, directive="You are a helpful assistant."))
agent.add_trait(LLMTrait(agent, llm_config))

# Start and use agent
agent.start()
llm = agent.require_trait(LLMTrait)
result = llm.complete([{"role": "user", "content": "Hello!"}])
print(result.content)
agent.stop()
```

## Core Concepts

### Agents

An `Agent` is a container for traits with lifecycle management. Agents have an identity
(domain/workspace/name) and can be started, stopped, and run in cycles.

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

Apache-2.0
