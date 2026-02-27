"""Tests for the Tool system (registry, CRUD, definitions)."""
from __future__ import annotations

import pytest

from aivilization.core.tool import ToolDefinition, ToolScope
from aivilization.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


def test_registry_has_builtins(registry):
    """All 10 built-in tools should be registered."""
    builtins = registry.get_builtin_tools()
    builtin_names = {t.name for t in builtins}

    expected = {
        "create_agent",
        "delegate_task",
        "broadcast",
        "create_tool",
        "edit_tool",
        "delete_tool",
        "sandbox",
        "evolve",
        "query_civilization",
        "form_alliance",
    }
    assert expected == builtin_names


def test_register_custom_tool(registry):
    """Test registering a custom tool."""
    tool = ToolDefinition(
        name="my_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        scope=ToolScope.SHARED,
        source_code="print(params['x'])",
        created_by_agent_id="agent-1",
    )
    registry.register(tool)

    assert registry.get_by_name("my_tool") is not None
    assert registry.get(tool.id) is not None


def test_register_duplicate_name_fails(registry):
    """Cannot register two tools with the same name."""
    tool1 = ToolDefinition(
        name="dup_tool",
        description="First",
        input_schema={"type": "object"},
        scope=ToolScope.SHARED,
    )
    tool2 = ToolDefinition(
        name="dup_tool",
        description="Second",
        input_schema={"type": "object"},
        scope=ToolScope.SHARED,
    )
    registry.register(tool1)
    with pytest.raises(ValueError, match="already exists"):
        registry.register(tool2)


def test_update_custom_tool(registry):
    """Test updating a custom tool."""
    tool = ToolDefinition(
        name="editable_tool",
        description="Original",
        input_schema={"type": "object"},
        scope=ToolScope.SHARED,
        created_by_agent_id="agent-1",
    )
    registry.register(tool)

    updated = registry.update(
        tool.id,
        {"description": "Updated description"},
        requesting_agent_id="agent-1",
    )

    assert updated.description == "Updated description"
    assert updated.version == 2


def test_cannot_update_builtin(registry):
    """Cannot update a built-in tool."""
    builtin = registry.get_builtin_tools()[0]
    with pytest.raises(PermissionError, match="built-in"):
        registry.update(builtin.id, {"description": "hacked"}, requesting_agent_id="x")


def test_delete_custom_tool(registry):
    """Test deleting a custom tool."""
    tool = ToolDefinition(
        name="deletable_tool",
        description="Will be deleted",
        input_schema={"type": "object"},
        scope=ToolScope.SHARED,
        created_by_agent_id="agent-1",
    )
    registry.register(tool)
    assert registry.get_by_name("deletable_tool") is not None

    name = registry.delete(tool.id, requesting_agent_id="agent-1")
    assert name == "deletable_tool"
    assert registry.get_by_name("deletable_tool") is None


def test_cannot_delete_builtin(registry):
    """Cannot delete a built-in tool."""
    builtin = registry.get_builtin_tools()[0]
    with pytest.raises(PermissionError, match="built-in"):
        registry.delete(builtin.id, requesting_agent_id="x")


def test_cannot_delete_others_private_tool(registry):
    """Cannot delete another agent's private tool."""
    tool = ToolDefinition(
        name="private_tool",
        description="My private tool",
        input_schema={"type": "object"},
        scope=ToolScope.PRIVATE,
        created_by_agent_id="agent-1",
    )
    registry.register(tool)

    with pytest.raises(PermissionError, match="another agent"):
        registry.delete(tool.id, requesting_agent_id="agent-2")


def test_search_tools(registry):
    """Test searching tools by name/description."""
    tool = ToolDefinition(
        name="csv_parser",
        description="Parses CSV files and returns data",
        input_schema={"type": "object"},
        scope=ToolScope.SHARED,
    )
    registry.register(tool)

    results = registry.search("csv")
    assert any(t.name == "csv_parser" for t in results)

    results = registry.search("Parses")
    assert any(t.name == "csv_parser" for t in results)


def test_tool_definition_to_claude_format():
    """Test converting tool definition to Claude API format."""
    tool = ToolDefinition(
        name="my_tool",
        description="Does something",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    )
    claude_param = tool.to_claude_tool_param()

    assert claude_param["name"] == "my_tool"
    assert claude_param["description"] == "Does something"
    assert claude_param["input_schema"]["type"] == "object"
    assert "x" in claude_param["input_schema"]["properties"]


def test_get_shared_tools(registry):
    """Test getting shared tools."""
    tool = ToolDefinition(
        name="shared_tool",
        description="Shared",
        input_schema={"type": "object"},
        scope=ToolScope.SHARED,
    )
    registry.register(tool)

    shared = registry.get_shared_tools()
    assert any(t.name == "shared_tool" for t in shared)
