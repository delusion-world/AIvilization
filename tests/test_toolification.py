"""Tests for the Toolification Engine."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aivilization.config import AIvilizationConfig
from aivilization.core.agent import Agent, AgentState
from aivilization.core.memory import AgentMemory
from aivilization.core.tool import ToolExecutionRecord
from aivilization.tools.registry import ToolRegistry
from aivilization.tools.toolification import ToolificationEngine


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def engine(registry):
    return ToolificationEngine(registry, threshold=3)


@pytest.fixture
def mock_agent():
    state = AgentState(
        name="TestAgent",
        role="Testing",
        system_prompt_base="Test",
        memory=AgentMemory(),
    )
    # Create a mock agent with the state
    agent = MagicMock()
    agent.id = state.id
    agent.state = state
    return agent


def test_no_candidates_with_few_executions(engine, mock_agent):
    """No candidates when there are too few executions."""
    candidates = engine.analyze(mock_agent)
    assert len(candidates) == 0


def test_detect_code_patterns(engine, registry, mock_agent):
    """Detect repeated code execution patterns."""
    # Add 3 similar code executions
    for i in range(3):
        record = ToolExecutionRecord(
            tool_name="sandbox",
            agent_id=mock_agent.id,
            input_params={"action": "exec_python", "code": f'data = load_csv("file{i}.csv")\nprint(len(data))'},
            output=f"{i * 10}",
            success=True,
            duration_ms=100,
        )
        registry._execution_log.append(record)

    candidates = engine.analyze(mock_agent)
    code_candidates = [c for c in candidates if "code pattern" in c.pattern_description]
    assert len(code_candidates) == 1
    assert code_candidates[0].frequency >= 3


def test_detect_sequence_patterns(engine, registry, mock_agent):
    """Detect repeated tool call sequences."""
    # Add repeated bigram: create_agent -> delegate_task
    for _ in range(4):
        registry._execution_log.append(
            ToolExecutionRecord(
                tool_name="create_agent",
                agent_id=mock_agent.id,
                input_params={"name": "x"},
                output="ok",
                success=True,
                duration_ms=50,
            )
        )
        registry._execution_log.append(
            ToolExecutionRecord(
                tool_name="delegate_task",
                agent_id=mock_agent.id,
                input_params={"agent_id": "y", "message": "do stuff"},
                output="done",
                success=True,
                duration_ms=50,
            )
        )

    candidates = engine.analyze(mock_agent)
    seq_candidates = [c for c in candidates if "sequence" in c.pattern_description]
    assert len(seq_candidates) >= 1


def test_skeleton_extraction(engine):
    """Test code skeleton extraction."""
    code = '''
data = load("my_file.csv")
result = process(data, 42)
print(f"Result: {result}")
'''
    skeleton = engine._extract_skeleton(code)
    assert "<STR>" in skeleton
    assert "<NUM>" in skeleton
    assert "load" in skeleton


def test_ignores_other_agents(engine, registry, mock_agent):
    """Only detects patterns for the specified agent."""
    # Add executions from a different agent
    for i in range(5):
        record = ToolExecutionRecord(
            tool_name="sandbox",
            agent_id="other-agent",
            input_params={"action": "exec_python", "code": f'print("hello {i}")'},
            output="hello",
            success=True,
            duration_ms=50,
        )
        registry._execution_log.append(record)

    candidates = engine.analyze(mock_agent)
    assert len(candidates) == 0
