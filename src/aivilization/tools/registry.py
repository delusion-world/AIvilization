from __future__ import annotations

import time
from typing import Any, Callable, TYPE_CHECKING

from aivilization.core.tool import ToolDefinition, ToolExecutionRecord, ToolScope
from aivilization.tools.builtin import BUILTIN_TOOLS

if TYPE_CHECKING:
    from aivilization.core.agent import Agent


class ToolRegistry:
    """
    Central registry for all tools in the civilization.

    Manages built-in tools, custom tools, and tool execution.
    Records all executions for toolification analysis.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}  # id -> ToolDefinition
        self._tools_by_name: dict[str, ToolDefinition] = {}  # name -> ToolDefinition
        self._handlers: dict[str, Callable] = {}  # name -> async handler
        self._execution_log: list[ToolExecutionRecord] = []

        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in tools."""
        for name, (definition, handler) in BUILTIN_TOOLS.items():
            tool_def = ToolDefinition(
                name=definition["name"],
                description=definition["description"],
                input_schema=definition["input_schema"],
                scope=ToolScope.BUILTIN,
            )
            self._tools[tool_def.id] = tool_def
            self._tools_by_name[name] = tool_def
            if handler:
                self._handlers[name] = handler

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a new custom tool."""
        if tool_def.name in self._tools_by_name:
            raise ValueError(
                f"Tool '{tool_def.name}' already exists. Use edit_tool to modify it, "
                f"or choose a different name."
            )
        self._tools[tool_def.id] = tool_def
        self._tools_by_name[tool_def.name] = tool_def

        if tool_def.source_code:
            self._handlers[tool_def.name] = self._make_custom_handler(tool_def)

    def update(
        self, tool_id: str, updates: dict[str, Any], requesting_agent_id: str
    ) -> ToolDefinition:
        """Update an existing tool. Increments version."""
        tool = self._tools.get(tool_id)
        if tool is None:
            raise ValueError(f"No tool found with id '{tool_id}'.")

        if tool.scope == ToolScope.BUILTIN:
            raise PermissionError("Cannot edit built-in tools.")

        if tool.scope == ToolScope.PRIVATE and tool.created_by_agent_id != requesting_agent_id:
            raise PermissionError("Cannot edit another agent's private tool.")

        old_name = tool.name

        # Apply updates
        for key, value in updates.items():
            if hasattr(tool, key):
                setattr(tool, key, value)

        tool.version += 1

        # Update name mapping if name changed
        if "name" in updates and updates["name"] != old_name:
            del self._tools_by_name[old_name]
            self._tools_by_name[tool.name] = tool
            # Update handler mapping
            if old_name in self._handlers:
                self._handlers[tool.name] = self._handlers.pop(old_name)

        # Rebuild handler if implementation changed
        if "source_code" in updates and tool.source_code:
            self._handlers[tool.name] = self._make_custom_handler(tool)

        return tool

    def delete(self, tool_id: str, requesting_agent_id: str) -> str:
        """Delete a tool. Returns the deleted tool's name."""
        tool = self._tools.get(tool_id)
        if tool is None:
            raise ValueError(f"No tool found with id '{tool_id}'.")

        if tool.scope == ToolScope.BUILTIN:
            raise PermissionError("Cannot delete built-in tools.")

        if tool.scope == ToolScope.PRIVATE and tool.created_by_agent_id != requesting_agent_id:
            raise PermissionError("Cannot delete another agent's private tool.")

        name = tool.name
        del self._tools[tool_id]
        del self._tools_by_name[name]
        if name in self._handlers:
            del self._handlers[name]

        return name

    def _make_custom_handler(self, tool_def: ToolDefinition) -> Callable:
        """Create an async handler from a tool's source code."""

        async def handler(agent: Agent, input: dict[str, Any]) -> str:
            wrapper = f"params = {repr(input)}\n{tool_def.source_code}"
            result = agent.civilization.sandbox_manager.exec_python(agent.id, wrapper)
            if result.error:
                raise RuntimeError(result.error)
            if result.exit_code != 0 and result.stderr:
                raise RuntimeError(result.stderr)
            return result.stdout.strip() if result.stdout else "(no output)"

        return handler

    async def execute(
        self, tool_name: str, tool_input: dict[str, Any], agent: Agent
    ) -> str:
        """Execute a tool by name, record the execution, return result."""
        if tool_name not in self._handlers:
            raise ValueError(f"No handler registered for tool '{tool_name}'.")

        start = time.monotonic()
        try:
            result = await self._handlers[tool_name](agent, tool_input)
            duration = (time.monotonic() - start) * 1000
            success = True
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            result = str(e)
            success = False

        # Record execution
        record = ToolExecutionRecord(
            tool_name=tool_name,
            agent_id=agent.id,
            input_params=tool_input,
            output=result[:1000],
            success=success,
            duration_ms=duration,
        )
        self._execution_log.append(record)

        # Update usage count
        if tool_name in self._tools_by_name:
            self._tools_by_name[tool_name].usage_count += 1

        if not success:
            raise RuntimeError(result)

        return result

    def get(self, tool_id: str) -> ToolDefinition | None:
        return self._tools.get(tool_id)

    def get_by_name(self, name: str) -> ToolDefinition | None:
        return self._tools_by_name.get(name)

    def get_builtin_tools(self) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if t.scope == ToolScope.BUILTIN]

    def get_shared_tools(self) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if t.scope == ToolScope.SHARED]

    def get_all_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def search(self, query: str) -> list[ToolDefinition]:
        """Search tools by name or description."""
        query_lower = query.lower()
        return [
            t
            for t in self._tools.values()
            if query_lower in t.name.lower() or query_lower in t.description.lower()
        ]

    @property
    def execution_log(self) -> list[ToolExecutionRecord]:
        return self._execution_log
