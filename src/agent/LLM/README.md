# Agent/LLM — LLM Client & Configuration

**What it is:** LLM integration layer — client, configuration, lifecycle management, and interactive model picker. Powers API-based inference for Darwin agents.

**Why it exists:** Darwin needs to call external LLM APIs (OpenAI, Anthropic, etc.) for agent reasoning, RAG generation, and model orchestration (API-based models like GPT, Claude). This module wraps litellm for multi-provider support.

## Files

| File | Purpose |
|---|---|
| `LLMClient.py` | LLM inference client (litellm-based, multi-provider) |
| `LLMConfig.py` | LLM configuration dataclasses |
| `LLMManager.py` | Lifecycle management (init, load, unload) |
| `LLMTypes.py` | Type definitions for LLM interactions |
| `ModelPicker.py` | Interactive model picker (Rich-based) |
| `main.py` | CLI entry for LLM operations |
| `test.py` / `testllm.py` | Tests |

## Usage

```python
from src.agent.LLM.LLMClient import LLMClient
from src.agent.LLM.LLMConfig import LLMConfig

config = LLMConfig(
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=4096,
)

client = LLMClient(config)
response = client.chat_completion([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this: ..."},
])
print(response.content)
```

## Configuration

```python
from src.agent.LLM.LLMManager import LLMManager

manager = LLMManager()
manager.initialize(config)
manager.switch_model("claude-sonnet-4-20250514")
manager.unload()
```

## Integration with Darwin

The LLM module is used by:
- **API-based models**: When a user selects an API model (e.g., GPT-4o) via `darwin admin models switch`, the `ModelOrchestrator` uses `LLMManager` for inference.
- **Prompts**: The `src/lib/Prompts/IntentAnalyzer.md` prompt template is used for intent routing.
