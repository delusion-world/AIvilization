from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EpisodicMemory(BaseModel):
    """A single episodic memory — a notable event the agent remembers."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str
    importance: float = 0.5  # 0-1 scale
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    related_agent_ids: list[str] = Field(default_factory=list)
    related_tool_ids: list[str] = Field(default_factory=list)


class AgentMemory(BaseModel):
    """All memory state for a single agent."""

    # Conversation history (recent, bounded)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    max_conversation_turns: int = 50

    # Episodic memories (distilled from conversations)
    episodic_memories: list[EpisodicMemory] = Field(default_factory=list)
    max_episodic_memories: int = 200

    # Semantic knowledge (key facts the agent has learned)
    knowledge: dict[str, str] = Field(default_factory=dict)

    # Role evolution notes
    role_notes: list[str] = Field(default_factory=list)

    def add_conversation_turn(self, role: str, content: Any) -> None:
        """Add a turn to conversation history, evicting oldest if at capacity."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_conversation_turns * 2:
            self._compact_history()

    def _compact_history(self) -> None:
        """Trim old conversation turns, preserving the most recent window."""
        if len(self.conversation_history) > self.max_conversation_turns + 2:
            self.conversation_history = (
                self.conversation_history[:2]
                + self.conversation_history[-(self.max_conversation_turns) :]
            )

    def add_knowledge(self, key: str, value: str) -> None:
        """Store a key fact."""
        self.knowledge[key] = value

    def add_episodic_memory(
        self,
        summary: str,
        importance: float = 0.5,
        related_agent_ids: list[str] | None = None,
        related_tool_ids: list[str] | None = None,
    ) -> EpisodicMemory:
        """Record a notable event."""
        memory = EpisodicMemory(
            summary=summary,
            importance=importance,
            related_agent_ids=related_agent_ids or [],
            related_tool_ids=related_tool_ids or [],
        )
        self.episodic_memories.append(memory)
        if len(self.episodic_memories) > self.max_episodic_memories:
            # Remove least important memories
            self.episodic_memories.sort(key=lambda m: m.importance, reverse=True)
            self.episodic_memories = self.episodic_memories[: self.max_episodic_memories]
        return memory

    def add_role_note(self, note: str) -> None:
        """Record a role evolution note."""
        self.role_notes.append(note)

    def get_context_for_prompt(self) -> str:
        """Build a memory summary string to inject into system prompt."""
        parts = []

        if self.role_notes:
            recent_notes = self.role_notes[-5:]
            parts.append(
                "## Your Role Evolution\n" + "\n".join(f"- {n}" for n in recent_notes)
            )

        if self.episodic_memories:
            top_memories = sorted(
                self.episodic_memories, key=lambda m: m.importance, reverse=True
            )[:10]
            parts.append(
                "## Key Memories\n"
                + "\n".join(f"- {m.summary}" for m in top_memories)
            )

        if self.knowledge:
            items = list(self.knowledge.items())[-10:]
            parts.append(
                "## Known Facts\n" + "\n".join(f"- {k}: {v}" for k, v in items)
            )

        return "\n\n".join(parts)
