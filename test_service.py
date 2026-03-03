"""
Test script: exercise the AIvilization service programmatically.

Sends a task to the Coordinator agent asking it to design a 3D mechanism,
and observes how the civilization handles it.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env file if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from aivilization.config import AIvilizationConfig
from aivilization.core.civilization import Civilization


async def main() -> None:
    # Ensure API key is available - check both var names
    api_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("AIVILIZATION_ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env or environment")
        sys.exit(1)

    config = AIvilizationConfig(
        anthropic_api_key=api_key,
        default_model="claude-sonnet-4-6",
    )
    print(f"Model: {config.default_model}")
    print(f"Max agents: {config.max_agents}")
    print(f"Budget: ${config.max_cost_per_session_usd}")
    print()

    # Create a civilization
    civ = Civilization(config, name="MechanismDesign Civilization")
    civ.create_agent(
        name="Coordinator",
        role="Primary coordinator that helps the user and delegates to specialists",
        system_prompt_base=(
            "You are the Coordinator, the primary AI agent in AIvilization — "
            "a civilization of AI agents that autonomously collaborate and evolve.\n\n"
            "You are the user's main point of contact. Your responsibilities:\n"
            "1. Understand what the user needs and handle simple requests directly\n"
            "2. Create specialized agents (create_agent) when tasks need deep expertise\n"
            "3. Delegate to existing agents (delegate_task) when appropriate\n"
            "4. Create reusable tools (create_tool) when you notice repeated patterns\n"
            "5. Form alliances (form_alliance) for ongoing multi-agent collaboration\n"
            "6. Use your sandbox for computation, code execution, and file management\n"
            "7. Query the civilization state (query_civilization) to stay informed\n\n"
            "Be proactive about building your civilization. When you see an opportunity "
            "to specialize, create an agent. When you see a repeated pattern, create a tool. "
            "When agents should work together regularly, form an alliance.\n\n"
            "Always think about efficiency, specialization, and the growth of the civilization."
        ),
    )

    primary = civ.get_primary_agent()
    print(f"Civilization: {civ.state.name}")
    print(f"Primary agent: {primary.state.name} (id: {primary.id[:8]}...)")
    print(f"Tools available: {len(primary.get_available_tools())}")
    print()

    # Send the 3D mechanism design request
    task = (
        "Design a simple gear mechanism with two interlocking gears. "
        "Describe the parameters needed (teeth count, module, pressure angle, shaft distance) "
        "and generate Python code that computes the gear geometry "
        "(tooth profiles as coordinate lists). "
        "Output the result so it could be used for 3D modeling."
    )

    print(f"TASK: {task}")
    print("=" * 60)
    print("Processing...")
    print()

    try:
        response = await primary.process_message(task)
        print("RESPONSE:")
        print("=" * 60)
        print(response)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Print stats
    print()
    print("=" * 60)
    print("STATS:")
    usage = civ.claude_client.get_usage_summary()
    print(f"  API calls: {usage['total_api_calls']}")
    print(f"  Input tokens: {usage['total_input_tokens']:,}")
    print(f"  Output tokens: {usage['total_output_tokens']:,}")
    print(f"  Cost: ${usage['estimated_cost_usd']:.4f}")
    print(f"  Agents: {len(civ.get_all_agents())}")
    print(f"  Events: {len(civ.events.get_recent_events(100))}")

    # Show events
    events = civ.events.get_recent_events(20)
    if events:
        print()
        print("EVENTS:")
        for e in events:
            ts = e.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {e.event_type}: {e.data}")

    # Save and cleanup
    try:
        path = civ.save()
        print(f"\nSaved to: {path}")
    except Exception as e:
        print(f"\nSave error (non-critical): {e}")

    civ.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
