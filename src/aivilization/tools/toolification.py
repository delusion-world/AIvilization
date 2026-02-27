"""
Toolification Engine — detects repeated patterns in agent behavior
and proposes turning them into reusable tools.

Detection strategies:
1. Repeated exec_python calls with structurally similar code (skeleton matching)
2. Repeated sequences of tool calls (n-gram detection)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from aivilization.core.tool import ToolExecutionRecord

if TYPE_CHECKING:
    from aivilization.core.agent import Agent
    from aivilization.tools.registry import ToolRegistry


class ToolificationCandidate(BaseModel):
    """A detected pattern that could become a tool."""

    pattern_description: str
    agent_id: str
    frequency: int
    example_invocations: list[dict[str, Any]] = Field(default_factory=list)
    suggested_name: str
    suggested_description: str
    suggested_schema: dict[str, Any] = Field(default_factory=dict)
    suggested_implementation: str = ""


class ToolificationEngine:
    """
    Analyzes agent execution history to detect repeated patterns
    that should be turned into reusable tools.
    """

    def __init__(self, registry: ToolRegistry, threshold: int = 3) -> None:
        self.registry = registry
        self.threshold = threshold

    def analyze(self, agent: Agent) -> list[ToolificationCandidate]:
        """Analyze an agent's execution history and return toolification candidates."""
        candidates: list[ToolificationCandidate] = []
        candidates.extend(self._detect_code_patterns(agent))
        candidates.extend(self._detect_sequence_patterns(agent))
        return candidates

    def _detect_code_patterns(self, agent: Agent) -> list[ToolificationCandidate]:
        """Detect repeated code execution patterns via skeleton matching."""
        # Get all sandbox exec_python calls by this agent
        code_executions = [
            r
            for r in self.registry.execution_log
            if r.agent_id == agent.id
            and r.tool_name == "sandbox"
            and r.success
            and r.input_params.get("action") == "exec_python"
        ]

        if len(code_executions) < self.threshold:
            return []

        # Group by code skeleton
        skeleton_groups: dict[str, list[ToolExecutionRecord]] = defaultdict(list)
        for record in code_executions:
            code = record.input_params.get("code", "")
            skeleton = self._extract_skeleton(code)
            if skeleton:  # Skip empty skeletons
                skeleton_groups[skeleton].append(record)

        candidates = []
        for skeleton, records in skeleton_groups.items():
            if len(records) >= self.threshold:
                candidates.append(
                    self._build_candidate_from_code(skeleton, records, agent)
                )

        return candidates

    def _detect_sequence_patterns(self, agent: Agent) -> list[ToolificationCandidate]:
        """Detect repeated sequences of tool calls."""
        agent_records = [
            r
            for r in self.registry.execution_log
            if r.agent_id == agent.id and r.success
        ]

        if len(agent_records) < self.threshold * 2:
            return []

        # Build bigrams of tool call sequences
        bigram_counter: Counter = Counter()
        for i in range(len(agent_records) - 1):
            bigram = (agent_records[i].tool_name, agent_records[i + 1].tool_name)
            # Skip self-loops and sandbox-only sequences
            if bigram[0] != bigram[1]:
                bigram_counter[bigram] += 1

        candidates = []
        for sequence, count in bigram_counter.items():
            if count >= self.threshold:
                candidates.append(
                    self._build_candidate_from_sequence(sequence, count, agent)
                )

        return candidates

    def _extract_skeleton(self, code: str) -> str:
        """Reduce code to its structural skeleton by replacing literals with placeholders."""
        # Replace string literals
        skeleton = re.sub(r'"[^"]*"', '"<STR>"', code)
        skeleton = re.sub(r"'[^']*'", "'<STR>'", skeleton)
        # Replace numbers
        skeleton = re.sub(r"\b\d+\.?\d*\b", "<NUM>", skeleton)
        # Normalize whitespace
        skeleton = re.sub(r"\s+", " ", skeleton).strip()
        return skeleton

    def _build_candidate_from_code(
        self,
        skeleton: str,
        records: list[ToolExecutionRecord],
        agent: Agent,
    ) -> ToolificationCandidate:
        """Build a toolification candidate from repeated code patterns."""
        tool_count = len(self.registry.get_all_tools())
        return ToolificationCandidate(
            pattern_description=f"Repeated code pattern detected ({len(records)} occurrences)",
            agent_id=agent.id,
            frequency=len(records),
            example_invocations=[r.input_params for r in records[:3]],
            suggested_name=f"auto_{agent.state.name.lower().replace(' ', '_')}_tool_{tool_count}",
            suggested_description="Auto-detected repeated code pattern — refine before use",
            suggested_schema={"type": "object", "properties": {}},
            suggested_implementation=records[0].input_params.get("code", ""),
        )

    def _build_candidate_from_sequence(
        self,
        sequence: tuple[str, ...],
        count: int,
        agent: Agent,
    ) -> ToolificationCandidate:
        """Build a candidate from repeated tool call sequences."""
        tool_names = " → ".join(sequence)
        return ToolificationCandidate(
            pattern_description=f"Repeated tool sequence: {tool_names} ({count} occurrences)",
            agent_id=agent.id,
            frequency=count,
            suggested_name=f"workflow_{'_then_'.join(sequence)}",
            suggested_description=f"Automated workflow: {tool_names}",
            suggested_schema={"type": "object", "properties": {}},
            suggested_implementation="# Composite workflow — needs refinement",
        )

    async def propose_toolification(
        self, agent: Agent, candidate: ToolificationCandidate
    ) -> str:
        """
        Ask the agent to refine a toolification candidate into a proper tool.
        The agent decides whether the pattern merits toolification.
        """
        proposal_msg = (
            f"I've detected a repeated pattern in your work:\n\n"
            f"**{candidate.pattern_description}**\n\n"
            f"Examples:\n"
        )
        for i, ex in enumerate(candidate.example_invocations[:3], 1):
            proposal_msg += f"  {i}. {ex}\n"

        proposal_msg += (
            f"\nSuggested tool name: `{candidate.suggested_name}`\n\n"
            f"Would you like to create a reusable tool from this pattern? "
            f"If so, use the `create_tool` function to formalize it with "
            f"proper parameters, description, and implementation."
        )

        response = await agent.process_message(proposal_msg)
        return response
