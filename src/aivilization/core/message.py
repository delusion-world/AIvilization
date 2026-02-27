from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentBlock(BaseModel):
    """A single content block within a message. Mirrors Claude API content blocks."""

    type: str  # "text", "tool_use", "tool_result"
    text: str | None = None
    id: str | None = None  # tool_use id
    name: str | None = None  # tool name
    input: dict[str, Any] | None = None  # tool input
    tool_use_id: str | None = None  # for tool_result
    content: str | Any | None = None  # tool_result content
    is_error: bool = False


class Message(BaseModel):
    """A message in a conversation thread."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str | list[ContentBlock]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Message sent between agents via delegation or broadcast."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent_id: str
    to_agent_id: str
    content: str
    reply_to: str | None = None
    task_context: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # pending, delivered, read, replied
