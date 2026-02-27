from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolScope(str, Enum):
    BUILTIN = "builtin"  # Available to all agents, cannot be modified
    SHARED = "shared"  # Created by agent, shared across civilization
    PRIVATE = "private"  # Only available to the creating agent


class ToolDefinition(BaseModel):
    """The persistent, serializable definition of a tool."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # Must match ^[a-zA-Z0-9_-]{1,64}$
    description: str
    input_schema: dict[str, Any]
    scope: ToolScope = ToolScope.PRIVATE
    created_by_agent_id: str | None = None  # null for builtin
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_code: str | None = None  # Python source for custom tools
    version: int = 1
    usage_count: int = 0
    tags: list[str] = Field(default_factory=list)

    # Toolification provenance
    toolified_from_pattern: str | None = None
    example_invocations: list[dict[str, Any]] = Field(default_factory=list)

    def to_claude_tool_param(self) -> dict[str, Any]:
        """Convert to the format expected by Claude's tools parameter."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolExecutionRecord(BaseModel):
    """Records a single tool execution for history/toolification analysis."""

    tool_name: str
    agent_id: str
    input_params: dict[str, Any]
    output: str
    success: bool
    duration_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
