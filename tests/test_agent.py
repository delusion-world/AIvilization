"""Tests for the Agent class and agent loop."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aivilization.config import AIvilizationConfig
from aivilization.core.agent import Agent, AgentState, AgentStatus
from aivilization.core.civilization import Civilization
from aivilization.core.memory import AgentMemory


@pytest.fixture
def config():
    return AIvilizationConfig(anthropic_api_key="test-key")


@pytest.fixture
def civilization(config):
    civ = Civilization(config, name="Test Civilization")
    # Mock the Claude client to avoid real API calls
    civ.claude_client = MagicMock()
    return civ


def test_agent_state_creation():
    """Test that AgentState creates with proper defaults."""
    state = AgentState(
        name="TestAgent",
        role="Testing",
        system_prompt_base="You are a test agent.",
    )
    assert state.name == "TestAgent"
    assert state.role == "Testing"
    assert state.status == AgentStatus.IDLE
    assert state.depth == 0
    assert state.created_by is None
    assert state.total_api_calls == 0
    assert isinstance(state.memory, AgentMemory)
    assert state.id  # Should have auto-generated ID


def test_agent_build_system_prompt(civilization):
    """Test that system prompt is built correctly."""
    agent = civilization.create_agent(
        name="TestAgent",
        role="Testing",
        system_prompt_base="You are a test agent.",
    )

    prompt = agent.build_system_prompt()
    assert "You are a test agent." in prompt
    assert "Meta-Instructions" in prompt
    assert "AIvilization" in prompt


def test_agent_build_system_prompt_with_other_agents(civilization):
    """Test that system prompt includes info about other agents."""
    agent1 = civilization.create_agent(
        name="Agent1",
        role="Role1",
        system_prompt_base="Base1",
    )
    agent2 = civilization.create_agent(
        name="Agent2",
        role="Role2",
        system_prompt_base="Base2",
    )

    prompt = agent1.build_system_prompt()
    assert "Agent2" in prompt
    assert "Role2" in prompt


def test_agent_get_available_tools(civilization):
    """Test that available tools include builtins."""
    agent = civilization.create_agent(
        name="TestAgent",
        role="Testing",
        system_prompt_base="Base",
    )

    tools = agent.get_available_tools()
    tool_names = [t["name"] for t in tools]

    # Check all 10 built-in tools
    assert "create_agent" in tool_names
    assert "delegate_task" in tool_names
    assert "broadcast" in tool_names
    assert "create_tool" in tool_names
    assert "edit_tool" in tool_names
    assert "delete_tool" in tool_names
    assert "sandbox" in tool_names
    assert "evolve" in tool_names
    assert "query_civilization" in tool_names
    assert "form_alliance" in tool_names


def test_agent_memory_in_prompt(civilization):
    """Test that memory context appears in system prompt."""
    agent = civilization.create_agent(
        name="TestAgent",
        role="Testing",
        system_prompt_base="Base",
    )

    agent.state.memory.add_knowledge("test_key", "test_value")
    agent.state.memory.add_role_note("Became specialized in testing")

    prompt = agent.build_system_prompt()
    assert "test_key" in prompt
    assert "test_value" in prompt
    assert "specialized in testing" in prompt


def test_agent_alliance_in_prompt(civilization):
    """Test that alliance info appears in system prompt."""
    agent1 = civilization.create_agent(
        name="Agent1", role="Role1", system_prompt_base="Base1"
    )
    agent2 = civilization.create_agent(
        name="Agent2", role="Role2", system_prompt_base="Base2"
    )

    civilization.create_alliance(
        name="TestAlliance",
        agent_ids=[agent1.id, agent2.id],
        purpose="Testing together",
        created_by=agent1.id,
    )

    prompt = agent1.build_system_prompt()
    assert "TestAlliance" in prompt
    assert "Testing together" in prompt
