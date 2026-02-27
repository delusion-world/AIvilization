from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class AIvilizationConfig(BaseSettings):
    """Global configuration loaded from environment / .env file."""

    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-20250514"
    expensive_model: str = "claude-opus-4-6"
    max_tokens_per_turn: int = 4096
    max_agent_depth: int = 10
    max_agents: int = 50
    max_loop_iterations: int = 20
    data_dir: Path = Path("data/civilizations")

    # Docker sandbox settings
    sandbox_base_image: str = "aivilization-sandbox:latest"
    sandbox_memory_mb: int = 256
    sandbox_cpu_fraction: float = 0.5
    sandbox_timeout_seconds: int = 30
    sandbox_pids_limit: int = 64

    # Toolification
    toolification_threshold: int = 3

    # Web dashboard
    web_host: str = "127.0.0.1"
    web_port: int = 8420

    # Cost control
    max_cost_per_session_usd: float = 10.0

    model_config = {"env_prefix": "AIVILIZATION_"}
