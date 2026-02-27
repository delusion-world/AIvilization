from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aivilization.config import AIvilizationConfig
from aivilization.core.agent import Agent, AgentState, AgentMemory
from aivilization.core.events import EventBus
from aivilization.core.sandbox import SandboxManager
from aivilization.core.tool import ToolScope
from aivilization.llm.claude import ClaudeClient
from aivilization.storage.json_store import JsonStore
from aivilization.tools.registry import ToolRegistry


class Alliance(BaseModel):
    """A persistent collaboration group between agents."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    purpose: str
    agent_ids: list[str]
    shared_context: str = ""
    shared_memory: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""  # agent_id that formed it


class CivilizationState(BaseModel):
    """Top-level serializable state of the entire civilization."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Civilization"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    primary_agent_id: str | None = None
    agent_states: dict[str, AgentState] = Field(default_factory=dict)
    tool_definitions: list[dict[str, Any]] = Field(default_factory=list)
    alliances: list[dict[str, Any]] = Field(default_factory=list)
    creation_graph: dict[str, list[str]] = Field(default_factory=dict)
    snapshot_tags: dict[str, str] = Field(default_factory=dict)  # agent_id -> image tag
    metadata: dict[str, Any] = Field(default_factory=dict)


class Civilization:
    """
    The top-level orchestrator. Owns all agents, the tool registry,
    the event bus, the Claude client, sandbox manager, and persistence.
    """

    def __init__(self, config: AIvilizationConfig, name: str = "New Civilization") -> None:
        self.config = config
        self.data_dir = config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.claude_client = ClaudeClient(config)
        self.tool_registry = ToolRegistry()
        self.events = EventBus()
        self.sandbox_manager = SandboxManager(config)
        self.store = JsonStore(self.data_dir)

        self._agents: dict[str, Agent] = {}
        self._alliances: dict[str, Alliance] = {}
        self._state = CivilizationState(name=name)

    # ── Agent Management ──

    def create_agent(
        self,
        name: str,
        role: str,
        system_prompt_base: str,
        created_by: str | None = None,
        depth: int = 0,
    ) -> Agent:
        """Create a new agent and add it to the civilization."""
        state = AgentState(
            name=name,
            role=role,
            system_prompt_base=system_prompt_base,
            created_by=created_by,
            depth=depth,
            memory=AgentMemory(),
        )

        agent = Agent(state=state, civilization=self)
        self._agents[agent.id] = agent
        self._state.agent_states[agent.id] = state

        # Update creation graph
        if created_by:
            if created_by not in self._state.creation_graph:
                self._state.creation_graph[created_by] = []
            self._state.creation_graph[created_by].append(agent.id)

        # First agent becomes primary
        if self._state.primary_agent_id is None:
            self._state.primary_agent_id = agent.id

        self.events.emit(
            "agent_created",
            {
                "agent_id": agent.id,
                "name": name,
                "role": role,
                "created_by": created_by,
            },
        )

        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def get_primary_agent(self) -> Agent | None:
        if self._state.primary_agent_id:
            return self._agents.get(self._state.primary_agent_id)
        return None

    def get_all_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def get_agent_directory(self, exclude: str | None = None) -> list[AgentState]:
        """Get agent states for directory listing (used in system prompts)."""
        return [a.state for a in self._agents.values() if a.id != exclude]

    @property
    def agents(self) -> dict[str, Agent]:
        return self._agents

    # ── Alliance Management ──

    def create_alliance(
        self,
        name: str,
        agent_ids: list[str],
        purpose: str,
        shared_context: str = "",
        created_by: str = "",
    ) -> Alliance:
        """Create a new alliance."""
        alliance = Alliance(
            name=name,
            agent_ids=agent_ids,
            purpose=purpose,
            shared_context=shared_context,
            created_by=created_by,
        )
        self._alliances[alliance.id] = alliance

        self.events.emit(
            "alliance_formed",
            {
                "alliance_id": alliance.id,
                "name": name,
                "agent_ids": agent_ids,
                "purpose": purpose,
            },
        )

        return alliance

    def get_alliance(self, alliance_id: str) -> Alliance | None:
        return self._alliances.get(alliance_id)

    def get_alliance_by_name(self, name: str) -> Alliance | None:
        for alliance in self._alliances.values():
            if alliance.name == name:
                return alliance
        return None

    def get_all_alliances(self) -> list[Alliance]:
        return list(self._alliances.values())

    def get_agent_alliances(self, agent_id: str) -> list[Alliance]:
        """Get all alliances an agent belongs to."""
        return [a for a in self._alliances.values() if agent_id in a.agent_ids]

    # ── Persistence ──

    def save(self) -> Path:
        """Save the entire civilization state to JSON."""
        # Update tool definitions in state
        self._state.tool_definitions = [
            t.model_dump()
            for t in self.tool_registry.get_all_tools()
            if t.scope != ToolScope.BUILTIN
        ]

        # Update alliances in state
        self._state.alliances = [a.model_dump() for a in self._alliances.values()]

        # Update agent states
        for agent_id, agent in self._agents.items():
            self._state.agent_states[agent_id] = agent.state

        state_data = self._state.model_dump(mode="json")
        return self.store.save_civilization(state_data)

    @classmethod
    def load(cls, config: AIvilizationConfig, civilization_id: str) -> Civilization:
        """Load a civilization from saved state."""
        store = JsonStore(config.data_dir)
        state_data = store.load_civilization(civilization_id)

        civ_state = CivilizationState(**state_data)
        civ = cls(config, name=civ_state.name)
        civ._state = civ_state

        # Reconstruct agents
        for agent_id, agent_state_data in state_data.get("agent_states", {}).items():
            if isinstance(agent_state_data, dict):
                agent_state = AgentState(**agent_state_data)
            else:
                agent_state = agent_state_data
            agent = Agent(state=agent_state, civilization=civ)
            civ._agents[agent_id] = agent

        # Reconstruct custom tools
        from aivilization.core.tool import ToolDefinition

        for tool_data in state_data.get("tool_definitions", []):
            tool_def = ToolDefinition(**tool_data)
            try:
                civ.tool_registry.register(tool_def)
            except ValueError:
                pass  # Already registered (e.g., name conflict)

        # Reconstruct alliances
        for alliance_data in state_data.get("alliances", []):
            alliance = Alliance(**alliance_data)
            civ._alliances[alliance.id] = alliance

        # Restore sandbox snapshots if available
        for agent_id, image_tag in civ_state.snapshot_tags.items():
            try:
                civ.sandbox_manager.restore(agent_id, image_tag)
            except Exception:
                pass  # Snapshot may not exist

        return civ

    def list_saved(self) -> list[dict[str, Any]]:
        """List all saved civilizations."""
        return self.store.list_civilizations()

    # ── Cleanup ──

    def shutdown(self) -> None:
        """Clean up all resources."""
        self.sandbox_manager.destroy_all()

    @property
    def state(self) -> CivilizationState:
        return self._state
