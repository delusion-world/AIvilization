from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from aivilization.core.memory import AgentMemory
from aivilization.core.tool import ToolScope

if TYPE_CHECKING:
    from aivilization.core.civilization import Civilization


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    WAITING_FOR_SUBAGENT = "waiting_for_subagent"
    ERROR = "error"


class AgentState(BaseModel):
    """Serializable agent state for persistence."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str
    system_prompt_base: str
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None  # Parent agent ID
    depth: int = 0  # Nesting depth in creation tree

    # Tool access
    tool_ids: list[str] = Field(default_factory=list)  # Private + shared tools

    # Memory
    memory: AgentMemory = Field(default_factory=AgentMemory)

    # Statistics
    total_api_calls: int = 0
    total_tokens_used: int = 0
    total_tools_created: int = 0
    total_agents_created: int = 0


class Agent:
    """
    A living agent in the civilization.

    Wraps AgentState (serializable) with runtime behavior
    (Claude API calls, tool execution, environment access).
    """

    def __init__(self, state: AgentState, civilization: Civilization) -> None:
        self.state = state
        self.civilization = civilization
        self._delegation_chain: list[str] = []

    @property
    def id(self) -> str:
        return self.state.id

    def build_system_prompt(self) -> str:
        """
        Assemble the full system prompt dynamically.
        Rebuilt before every API call to incorporate current state.
        """
        parts = [self.state.system_prompt_base]

        # Memory context
        memory_ctx = self.state.memory.get_context_for_prompt()
        if memory_ctx:
            parts.append(memory_ctx)

        # Civilization awareness
        other_agents = self.civilization.get_agent_directory(exclude=self.id)
        if other_agents:
            agent_lines = []
            for a in other_agents:
                agent_lines.append(f"- {a.name} (id: {a.id}): {a.role}")
            parts.append("## Other Agents in the Civilization\n" + "\n".join(agent_lines))

        # Alliance awareness
        alliances = self.civilization.get_agent_alliances(self.id)
        if alliances:
            alliance_lines = []
            for a in alliances:
                member_names = []
                for aid in a.agent_ids:
                    ag = self.civilization.get_agent(aid)
                    if ag:
                        member_names.append(ag.state.name)
                alliance_lines.append(
                    f"- {a.name}: {a.purpose} (members: {', '.join(member_names)})"
                )
            parts.append("## Your Alliances\n" + "\n".join(alliance_lines))

        # Meta-instructions for self-organization
        parts.append(
            "## Meta-Instructions\n"
            "You are part of AIvilization, a civilization of AI agents. "
            "As you handle more tasks, your role may evolve. "
            "Consider these strategies for efficiency:\n"
            "1. **Create a tool** (create_tool) when you notice repeated patterns\n"
            "2. **Create a specialist** (create_agent) when tasks need deep expertise\n"
            "3. **Delegate** (delegate_task) to existing agents when appropriate\n"
            "4. **Form alliances** (form_alliance) for ongoing collaboration\n"
            "5. **Evolve** (evolve) your role as your specialization crystallizes\n"
            "6. **Use your sandbox** for computation, file management, and experimentation\n"
            "Always think about building a thriving, efficient civilization."
        )

        return "\n\n".join(parts)

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get all tool definitions in Claude API format for this agent."""
        tools = []
        seen_names: set[str] = set()

        # Built-in tools
        for t in self.civilization.tool_registry.get_builtin_tools():
            tools.append(t.to_claude_tool_param())
            seen_names.add(t.name)

        # Agent's private tools
        for tool_id in self.state.tool_ids:
            tool_def = self.civilization.tool_registry.get(tool_id)
            if tool_def and tool_def.name not in seen_names:
                tools.append(tool_def.to_claude_tool_param())
                seen_names.add(tool_def.name)

        # All shared tools in civilization
        for t in self.civilization.tool_registry.get_shared_tools():
            if t.name not in seen_names:
                tools.append(t.to_claude_tool_param())
                seen_names.add(t.name)

        return tools

    async def process_message(self, user_message: str) -> str:
        """
        The main agent loop. Receives a message, thinks via Claude,
        executes tools as needed, returns final text response.

        Keeps calling Claude until stop_reason != "tool_use".
        """
        # Add user message to memory
        self.state.memory.add_conversation_turn("user", user_message)

        # Build messages for Claude API
        messages = self._build_api_messages()

        max_iterations = self.civilization.config.max_loop_iterations
        final_text = ""

        for iteration in range(max_iterations):
            self.state.status = AgentStatus.THINKING

            response = await self.civilization.claude_client.create_message(
                system=self.build_system_prompt(),
                messages=messages,
                tools=self.get_available_tools(),
                max_tokens=self.civilization.config.max_tokens_per_turn,
            )

            self.state.total_api_calls += 1
            self.state.total_tokens_used += (
                response.usage.input_tokens + response.usage.output_tokens
            )

            # Record assistant response
            assistant_content = [self._block_to_dict(b) for b in response.content]
            self.state.memory.add_conversation_turn("assistant", assistant_content)
            messages.append({"role": "assistant", "content": response.content})

            # If no tool use, we're done
            if response.stop_reason != "tool_use":
                final_text = self._extract_text(response.content)
                break

            # Execute all tool calls and collect results
            self.state.status = AgentStatus.EXECUTING_TOOL
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    result = await self._execute_tool(block.name, block.input, block.id)
                    tool_results.append(result)

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
            self.state.memory.add_conversation_turn("user", tool_results)

        self.state.status = AgentStatus.IDLE

        # Emit event for civilization tracking
        self.civilization.events.emit(
            "agent_responded",
            {
                "agent_id": self.id,
                "agent_name": self.state.name,
                "message_preview": final_text[:200],
                "iteration_count": iteration + 1 if "iteration" in dir() else 1,
            },
        )

        return final_text

    async def _execute_tool(
        self, tool_name: str, tool_input: dict[str, Any], tool_use_id: str
    ) -> dict[str, Any]:
        """Execute a single tool call, return a tool_result content block."""
        try:
            result = await self.civilization.tool_registry.execute(
                tool_name=tool_name,
                tool_input=tool_input,
                agent=self,
            )
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": str(result),
            }
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error executing tool '{tool_name}': {e}",
                "is_error": True,
            }

    def _build_api_messages(self) -> list[dict[str, Any]]:
        """Convert memory conversation history to Claude API message format."""
        return [
            {"role": turn["role"], "content": turn["content"]}
            for turn in self.state.memory.conversation_history
        ]

    @staticmethod
    def _extract_text(content_blocks: list) -> str:
        """Extract text from Claude response content blocks."""
        texts = []
        for block in content_blocks:
            if hasattr(block, "text") and block.text:
                texts.append(block.text)
        return "\n".join(texts)

    @staticmethod
    def _block_to_dict(block: Any) -> dict[str, Any]:
        """Convert a Claude content block to a serializable dict."""
        if hasattr(block, "model_dump"):
            return block.model_dump()
        if hasattr(block, "__dict__"):
            return {k: v for k, v in block.__dict__.items() if not k.startswith("_")}
        return {"type": "text", "text": str(block)}
