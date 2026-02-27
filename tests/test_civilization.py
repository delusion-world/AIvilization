"""Tests for the Civilization orchestrator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aivilization.config import AIvilizationConfig
from aivilization.core.civilization import Alliance, Civilization


@pytest.fixture
def config(tmp_path):
    return AIvilizationConfig(
        anthropic_api_key="test-key",
        data_dir=tmp_path / "civilizations",
    )


@pytest.fixture
def civilization(config):
    return Civilization(config, name="Test Civilization")


def test_create_civilization(civilization):
    """Test creating a new civilization."""
    assert civilization.state.name == "Test Civilization"
    assert len(civilization.agents) == 0
    assert civilization.get_primary_agent() is None


def test_create_agent(civilization):
    """Test creating agents in a civilization."""
    agent = civilization.create_agent(
        name="Agent1",
        role="Testing",
        system_prompt_base="You are a test agent.",
    )

    assert agent.state.name == "Agent1"
    assert agent.state.role == "Testing"
    assert len(civilization.agents) == 1
    assert civilization.get_primary_agent() is agent  # First agent is primary


def test_create_multiple_agents(civilization):
    """Test creating multiple agents with parent-child relationship."""
    parent = civilization.create_agent(
        name="Parent",
        role="Parent role",
        system_prompt_base="Parent prompt",
    )
    child = civilization.create_agent(
        name="Child",
        role="Child role",
        system_prompt_base="Child prompt",
        created_by=parent.id,
        depth=1,
    )

    assert len(civilization.agents) == 2
    assert child.state.created_by == parent.id
    assert child.state.depth == 1
    assert child.id in civilization.state.creation_graph.get(parent.id, [])


def test_get_agent_directory(civilization):
    """Test getting agent directory excluding self."""
    agent1 = civilization.create_agent(
        name="A1", role="R1", system_prompt_base="P1"
    )
    agent2 = civilization.create_agent(
        name="A2", role="R2", system_prompt_base="P2"
    )

    directory = civilization.get_agent_directory(exclude=agent1.id)
    assert len(directory) == 1
    assert directory[0].name == "A2"


def test_alliance_management(civilization):
    """Test creating and querying alliances."""
    a1 = civilization.create_agent(name="A1", role="R1", system_prompt_base="P1")
    a2 = civilization.create_agent(name="A2", role="R2", system_prompt_base="P2")

    alliance = civilization.create_alliance(
        name="TestAlliance",
        agent_ids=[a1.id, a2.id],
        purpose="Testing",
        created_by=a1.id,
    )

    assert alliance.name == "TestAlliance"
    assert len(alliance.agent_ids) == 2

    # Query by name
    found = civilization.get_alliance_by_name("TestAlliance")
    assert found is not None
    assert found.id == alliance.id

    # Query by agent
    agent_alliances = civilization.get_agent_alliances(a1.id)
    assert len(agent_alliances) == 1
    assert agent_alliances[0].name == "TestAlliance"

    # All alliances
    assert len(civilization.get_all_alliances()) == 1


def test_save_and_load(config):
    """Test saving and loading a civilization."""
    # Create and populate
    civ = Civilization(config, name="SaveTest")
    agent = civ.create_agent(
        name="SavedAgent",
        role="Saved Role",
        system_prompt_base="Saved prompt",
    )
    agent.state.memory.add_knowledge("key1", "value1")

    civ.create_alliance(
        name="SavedAlliance",
        agent_ids=[agent.id],
        purpose="Saved purpose",
        created_by=agent.id,
    )

    # Save
    path = civ.save()
    assert path.exists()

    # Load
    loaded = Civilization.load(config, civ.state.id)
    assert loaded.state.name == "SaveTest"
    assert len(loaded.agents) == 1

    loaded_agent = loaded.get_primary_agent()
    assert loaded_agent.state.name == "SavedAgent"
    assert loaded_agent.state.memory.knowledge.get("key1") == "value1"

    assert len(loaded.get_all_alliances()) == 1
    assert loaded.get_all_alliances()[0].name == "SavedAlliance"


def test_list_saved_civilizations(config):
    """Test listing saved civilizations."""
    civ = Civilization(config, name="Listed Civ")
    civ.create_agent(name="A", role="R", system_prompt_base="P")
    civ.save()

    # Create a new Civilization to list
    civ2 = Civilization(config)
    saved = civ2.list_saved()
    assert len(saved) >= 1
    assert any(s["name"] == "Listed Civ" for s in saved)


def test_events_emitted_on_agent_creation(civilization):
    """Test that events are emitted when agents are created."""
    civilization.create_agent(name="A1", role="R1", system_prompt_base="P1")

    events = civilization.events.get_events_by_type("agent_created")
    assert len(events) == 1
    assert events[0].data["name"] == "A1"


def test_events_emitted_on_alliance_formation(civilization):
    """Test that events are emitted when alliances are formed."""
    a1 = civilization.create_agent(name="A1", role="R1", system_prompt_base="P1")
    a2 = civilization.create_agent(name="A2", role="R2", system_prompt_base="P2")

    civilization.create_alliance(
        name="TestAlliance",
        agent_ids=[a1.id, a2.id],
        purpose="Testing",
        created_by=a1.id,
    )

    events = civilization.events.get_events_by_type("alliance_formed")
    assert len(events) == 1
    assert events[0].data["name"] == "TestAlliance"
