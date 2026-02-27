# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run CLI
aivilization
# or: python -m aivilization.main

# Run all tests
pytest

# Run a single test file
pytest tests/test_tools.py

# Run a single test
pytest tests/test_tools.py::test_registry_has_builtins -v

# Build the Docker sandbox image (required for sandbox features)
docker build -f docker/Dockerfile.agent -t aivilization-sandbox:latest docker/
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` — async test functions run automatically without decorators. The Claude client is mocked in tests (`MagicMock()`) to avoid real API calls.

## Architecture

AIvilization is a multi-agent system where AI agents autonomously collaborate, create tools, and evolve. Source lives in `src/aivilization/`.

### Core Loop

`Civilization` is the top-level orchestrator. It owns all subsystems:

```
Civilization
├── ClaudeClient        (llm/claude.py)      — Anthropic API wrapper with cost tracking
├── ToolRegistry        (tools/registry.py)   — Central tool store + execution engine
├── EventBus            (core/events.py)      — Sync pub/sub for civilization-wide events
├── SandboxManager      (core/sandbox.py)     — Docker container per agent
├── JsonStore           (storage/json_store.py)— File-based persistence (data/civilizations/)
├── Agents              (core/agent.py)       — Agent instances with state + runtime behavior
└── Alliances           (core/civilization.py) — Persistent collaboration groups
```

The agent loop (`Agent.process_message`) calls Claude in a loop, executing tools until `stop_reason != "tool_use"`. The system prompt is rebuilt dynamically before each API call to inject current memory, agent directory, alliance info, and meta-instructions.

### State vs Runtime Split

- **`AgentState`** (Pydantic model): serializable data — identity, memory, stats, tool IDs
- **`Agent`**: wraps `AgentState` with runtime behavior — API calls, tool execution, prompt building

This same pattern applies at civilization level: `CivilizationState` vs `Civilization`.

### Tool System

Three-tier scope: `BUILTIN` (10 tools, all agents), `SHARED` (agent-created, civilization-wide), `PRIVATE` (creator-only).

Built-in tools are defined as `(definition_dict, async_handler)` pairs in `tools/builtin.py`. The handler signature is `async def handler(agent: Agent, input: dict) -> str`. Custom tools execute Python source code inside the agent's Docker sandbox.

`ToolRegistry` records every execution as `ToolExecutionRecord`. The `ToolificationEngine` (`tools/toolification.py`) analyzes these records to detect repeated patterns (skeleton matching for code, bigram detection for tool sequences) and proposes turning them into reusable tools.

### 10 Built-in Tools

`create_agent`, `delegate_task`, `broadcast`, `create_tool`, `edit_tool`, `delete_tool`, `sandbox`, `evolve`, `query_civilization`, `form_alliance`

### Memory System (core/memory.py)

Each agent has `AgentMemory` with four layers:
- **Conversation history**: bounded sliding window (50 turns), auto-compacted
- **Episodic memories**: notable events with importance scores (0-1), pruned by importance
- **Knowledge**: key-value semantic facts
- **Role notes**: evolution journal entries

Memory is injected into the system prompt via `get_context_for_prompt()`.

### Configuration

`AIvilizationConfig` (pydantic-settings) reads from env vars prefixed `AIVILIZATION_`. Key settings: `anthropic_api_key`, `default_model`, `max_agents` (50), `max_agent_depth` (10), `max_loop_iterations` (20), `max_cost_per_session_usd` (10.0), sandbox resource limits.

### Web Dashboard

Read-only FastAPI app (`web/server.py`) with Jinja2 templates. Serves HTML at `/` and JSON APIs at `/api/events`, `/api/agents`, `/api/tools`.

### Persistence

`Civilization.save()` serializes the full state (agents, custom tools, alliances) to `data/civilizations/{id}.json` with an index file. `Civilization.load()` reconstructs everything including sandbox snapshots.
