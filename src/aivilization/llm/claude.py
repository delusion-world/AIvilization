from __future__ import annotations

from typing import Any

import anthropic

from aivilization.config import AIvilizationConfig


class ClaudeClient:
    """
    Wraps the Anthropic Python SDK for AIvilization.

    Handles API initialization, message creation with tool definitions,
    token tracking, and cost estimation.
    """

    def __init__(self, config: AIvilizationConfig) -> None:
        self.config = config
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_api_calls: int = 0

    async def create_message(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> anthropic.types.Message:
        """
        Create a message using the Claude API.

        The agentic loop (calling repeatedly until stop_reason != "tool_use")
        is handled by Agent.process_message(), not here.
        """
        model = model or self.config.default_model
        max_tokens = max_tokens or self.config.max_tokens_per_turn

        # Check budget
        if self.config.max_cost_per_session_usd > 0:
            current_cost = self._estimate_cost()
            if current_cost >= self.config.max_cost_per_session_usd:
                raise RuntimeError(
                    f"Session cost limit reached (${current_cost:.2f} >= "
                    f"${self.config.max_cost_per_session_usd:.2f})"
                )

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        # Track usage
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_api_calls += 1

        return response

    def get_usage_summary(self) -> dict[str, Any]:
        """Return a summary of API usage."""
        return {
            "total_api_calls": self.total_api_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }

    def _estimate_cost(self) -> float:
        """Rough cost estimate based on Claude Sonnet pricing."""
        input_cost = self.total_input_tokens * 3.0 / 1_000_000
        output_cost = self.total_output_tokens * 15.0 / 1_000_000
        return round(input_cost + output_cost, 4)
