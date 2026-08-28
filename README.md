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
from appinfra import DotDict
from appinfra.log import create_lg

from llm_gent import Agent, LLMTrait
from llm_gent.core.agent.types import ExecutionResult


# The public Agent class is abstract — a real application defines a small
# concrete subclass. Trivial stubs suffice when the workflow only uses
# LLMTrait.complete() directly.
class HelloAgent(Agent):
    def start(self) -> None:
        self._start_traits()

    def stop(self) -> None:
        self._stop_traits()

    def run_once(self) -> ExecutionResult:
        return ExecutionResult(success=True, content="")

    def ask(self, question: str) -> str:
        return ""

    def record_feedback(self, message: str) -> None:
        pass

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        return []


lg = create_lg("hello-agent", "info")

# Agent reads config.identity.name internally.
config = {"identity": {"name": "hello-agent"}}

llm_config = DotDict(
    {
        "default": "local",
        "backends": {
            "local": {
                "type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "model": "default",
            }
        },
    }
)

agent = HelloAgent(lg, config)
agent.add_trait(LLMTrait(agent, llm_config))
agent.start()

# LLMTrait.complete() accepts OpenAI-style message dicts.
llm = agent.require_trait(LLMTrait)
result = llm.complete(
    [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Say hello."},
    ]
)
print(result.content)

agent.stop()
```

A runnable version of this example lives at `examples/quickstart.py`. Set
`LLM_GENT_SMOKE=1` to run it against a stub LLM router (used by CI's
wheel-smoke job).

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

Apache License 2.0 - see [LICENSE](LICENSE) for details.

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
