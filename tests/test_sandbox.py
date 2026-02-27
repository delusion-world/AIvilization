"""Tests for the SandboxManager (Docker container management).

Note: These tests require Docker to be available. They are marked
to be skipped if Docker is not accessible.
"""
from __future__ import annotations

import pytest

from aivilization.core.sandbox import ExecResult, SandboxManager
from aivilization.config import AIvilizationConfig


def test_exec_result_model():
    """Test ExecResult model creation."""
    result = ExecResult(
        stdout="hello",
        stderr="",
        exit_code=0,
        duration_ms=42.5,
    )
    assert result.stdout == "hello"
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.error is None


def test_exec_result_with_error():
    """Test ExecResult with error state."""
    result = ExecResult(
        error="Connection refused",
        exit_code=-1,
        timed_out=True,
    )
    assert result.error == "Connection refused"
    assert result.timed_out is True
    assert result.exit_code == -1
