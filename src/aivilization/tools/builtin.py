"""
Built-in tools available to every agent in the civilization.

Each tool is defined as a (definition_dict, async_handler) pair.
The handler receives (agent, input_dict) and returns a string result.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aivilization.core.agent import Agent

# ──────────────────────────────────────────────
# 1. create_agent
# ──────────────────────────────────────────────
CREATE_AGENT_DEFINITION = {
    "name": "create_agent",
    "description": (
        "Create a new specialized AI agent in the civilization. "
        "The new agent will have its own Docker container, memory, tools, and workspace. "
        "Use this when a task requires a dedicated specialist, or when you find yourself "
        "repeatedly handling a type of work that would benefit from delegation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "A short, descriptive name (e.g., 'DataAnalyst', 'CodeReviewer')",
            },
            "role": {
                "type": "string",
                "description": "Brief description of the agent's specialization",
            },
            "system_prompt": {
                "type": "string",
                "description": "Detailed system prompt defining the agent's behavior and expertise",
            },
        },
        "required": ["name", "role", "system_prompt"],
    },
}


async def execute_create_agent(agent: Agent, input: dict[str, Any]) -> str:
    civ = agent.civilization
    if len(civ.agents) >= civ.config.max_agents:
        return f"Error: civilization has reached the maximum of {civ.config.max_agents} agents."
    if agent.state.depth >= civ.config.max_agent_depth:
        return f"Error: maximum agent nesting depth ({civ.config.max_agent_depth}) reached."

    new_agent = civ.create_agent(
        name=input["name"],
        role=input["role"],
        system_prompt_base=input["system_prompt"],
        created_by=agent.id,
        depth=agent.state.depth + 1,
    )
    agent.state.total_agents_created += 1

    return (
        f"Successfully created agent '{new_agent.state.name}' "
        f"(id: {new_agent.id}). "
        f"You can now delegate tasks to it using delegate_task."
    )


# ──────────────────────────────────────────────
# 2. delegate_task
# ──────────────────────────────────────────────
DELEGATE_TASK_DEFINITION = {
    "name": "delegate_task",
    "description": (
        "Send a task or message to another agent and receive their response. "
        "This is synchronous — you will wait for the response. "
        "Use this to delegate work, ask for information, or collaborate. "
        "The target agent will process your message with full context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "The ID of the target agent",
            },
            "message": {
                "type": "string",
                "description": "The task description or message to send",
            },
        },
        "required": ["agent_id", "message"],
    },
}


async def execute_delegate_task(agent: Agent, input: dict[str, Any]) -> str:
    target_id = input["agent_id"]
    message = input["message"]

    target = agent.civilization.get_agent(target_id)
    if target is None:
        available = agent.civilization.get_all_agents()
        agent_list = ", ".join(f"'{a.state.name}' (id: {a.id})" for a in available if a.id != agent.id)
        return f"Error: no agent found with id '{target_id}'. Available agents: {agent_list}"

    # Check for delegation cycles
    delegation_chain = getattr(agent, "_delegation_chain", [])
    if target_id in delegation_chain:
        return f"Error: delegation cycle detected. Chain: {' -> '.join(delegation_chain)} -> {target_id}"

    # Pass delegation chain to target
    target._delegation_chain = delegation_chain + [agent.id]

    delegated_msg = (
        f"[Task delegated from '{agent.state.name}' (id: {agent.id})]\n\n"
        f"{message}"
    )

    try:
        response = await target.process_message(delegated_msg)
    finally:
        target._delegation_chain = []

    return f"Response from '{target.state.name}':\n{response}"


# ──────────────────────────────────────────────
# 3. broadcast
# ──────────────────────────────────────────────
BROADCAST_DEFINITION = {
    "name": "broadcast",
    "description": (
        "Send a message to multiple agents simultaneously. "
        "Target all agents, or specify an alliance name to broadcast only to its members. "
        "Each agent processes the message independently and their responses are collected."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to broadcast",
            },
            "alliance_name": {
                "type": "string",
                "description": "Optional: broadcast only to members of this alliance. If omitted, broadcasts to all agents.",
            },
        },
        "required": ["message"],
    },
}


async def execute_broadcast(agent: Agent, input: dict[str, Any]) -> str:
    message = input["message"]
    alliance_name = input.get("alliance_name")

    civ = agent.civilization

    if alliance_name:
        alliance = civ.get_alliance_by_name(alliance_name)
        if alliance is None:
            return f"Error: no alliance found with name '{alliance_name}'."
        target_ids = [aid for aid in alliance.agent_ids if aid != agent.id]
    else:
        target_ids = [a.id for a in civ.get_all_agents() if a.id != agent.id]

    if not target_ids:
        return "No other agents to broadcast to."

    broadcast_msg = (
        f"[Broadcast from '{agent.state.name}' (id: {agent.id})]\n\n"
        f"{message}"
    )

    responses = []
    for target_id in target_ids:
        target = civ.get_agent(target_id)
        if target:
            try:
                resp = await target.process_message(broadcast_msg)
                responses.append(f"**{target.state.name}**: {resp}")
            except Exception as e:
                responses.append(f"**{target.state.name}**: Error - {e}")

    return "Broadcast responses:\n\n" + "\n\n---\n\n".join(responses)


# ──────────────────────────────────────────────
# 4. create_tool
# ──────────────────────────────────────────────
CREATE_TOOL_DEFINITION = {
    "name": "create_tool",
    "description": (
        "Create a new reusable tool that you or other agents can use. "
        "Define the tool's name, description, input parameters (as JSON Schema), "
        "and a Python implementation. The implementation is a Python code block that "
        "receives a 'params' dict and should print its output to stdout."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Tool name (alphanumeric, hyphens, underscores, max 64 chars)",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of what the tool does",
            },
            "parameters": {
                "type": "object",
                "description": "JSON Schema defining the tool's input parameters",
            },
            "implementation": {
                "type": "string",
                "description": "Python code. Receives 'params' dict, prints output to stdout.",
            },
            "scope": {
                "type": "string",
                "enum": ["private", "shared"],
                "description": "private = only you can use it; shared = all agents can use it",
                "default": "shared",
            },
        },
        "required": ["name", "description", "parameters", "implementation"],
    },
}


async def execute_create_tool(agent: Agent, input: dict[str, Any]) -> str:
    from aivilization.core.tool import ToolDefinition, ToolScope

    scope = ToolScope.SHARED if input.get("scope", "shared") == "shared" else ToolScope.PRIVATE

    tool_def = ToolDefinition(
        name=input["name"],
        description=input["description"],
        input_schema=input["parameters"],
        scope=scope,
        created_by_agent_id=agent.id,
        source_code=input["implementation"],
    )

    try:
        agent.civilization.tool_registry.register(tool_def)
    except ValueError as e:
        return f"Error creating tool: {e}"

    agent.state.tool_ids.append(tool_def.id)
    agent.state.total_tools_created += 1

    scope_msg = "shared with all agents" if scope == ToolScope.SHARED else "private to you"
    return (
        f"Tool '{tool_def.name}' created successfully (id: {tool_def.id}). "
        f"Scope: {scope_msg}. It is now available for use."
    )


# ──────────────────────────────────────────────
# 5. edit_tool
# ──────────────────────────────────────────────
EDIT_TOOL_DEFINITION = {
    "name": "edit_tool",
    "description": (
        "Modify an existing tool's definition, parameters, or implementation. "
        "You can update any combination of fields. Version is auto-incremented. "
        "You can only edit tools you created, or shared tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_id": {
                "type": "string",
                "description": "ID of the tool to edit",
            },
            "name": {
                "type": "string",
                "description": "New name (optional)",
            },
            "description": {
                "type": "string",
                "description": "New description (optional)",
            },
            "parameters": {
                "type": "object",
                "description": "New JSON Schema for parameters (optional)",
            },
            "implementation": {
                "type": "string",
                "description": "New Python implementation (optional)",
            },
        },
        "required": ["tool_id"],
    },
}


async def execute_edit_tool(agent: Agent, input: dict[str, Any]) -> str:
    tool_id = input["tool_id"]
    registry = agent.civilization.tool_registry

    updates = {}
    for field in ("name", "description", "parameters", "implementation"):
        if field in input and input[field] is not None:
            key = "input_schema" if field == "parameters" else (
                "source_code" if field == "implementation" else field
            )
            updates[key] = input[field]

    if not updates:
        return "No changes specified."

    try:
        updated = registry.update(tool_id, updates, requesting_agent_id=agent.id)
        return f"Tool '{updated.name}' updated to version {updated.version}."
    except (ValueError, PermissionError) as e:
        return f"Error: {e}"


# ──────────────────────────────────────────────
# 6. delete_tool
# ──────────────────────────────────────────────
DELETE_TOOL_DEFINITION = {
    "name": "delete_tool",
    "description": (
        "Delete a tool from the civilization. "
        "Built-in tools cannot be deleted. "
        "You can only delete tools you created, unless you're the primary agent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_id": {
                "type": "string",
                "description": "ID of the tool to delete",
            },
        },
        "required": ["tool_id"],
    },
}


async def execute_delete_tool(agent: Agent, input: dict[str, Any]) -> str:
    tool_id = input["tool_id"]
    registry = agent.civilization.tool_registry

    try:
        name = registry.delete(tool_id, requesting_agent_id=agent.id)
        # Remove from agent's tool_ids if present
        if tool_id in agent.state.tool_ids:
            agent.state.tool_ids.remove(tool_id)
        return f"Tool '{name}' has been deleted."
    except (ValueError, PermissionError) as e:
        return f"Error: {e}"


# ──────────────────────────────────────────────
# 7. sandbox
# ──────────────────────────────────────────────
SANDBOX_DEFINITION = {
    "name": "sandbox",
    "description": (
        "Execute operations in your isolated Docker container environment. "
        "Actions: exec_python (run Python code), exec_shell (run shell command), "
        "read_file, write_file, list_files, install_package. "
        "Your container has Python 3.12 with common packages pre-installed. "
        "Files persist between operations in /workspace."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "exec_python",
                    "exec_shell",
                    "read_file",
                    "write_file",
                    "list_files",
                    "install_package",
                ],
                "description": "The type of operation to perform",
            },
            "code": {
                "type": "string",
                "description": "Python code to execute (for exec_python)",
            },
            "command": {
                "type": "string",
                "description": "Shell command to execute (for exec_shell)",
            },
            "path": {
                "type": "string",
                "description": "File path relative to /workspace (for read/write/list)",
            },
            "content": {
                "type": "string",
                "description": "File content to write (for write_file)",
            },
            "package": {
                "type": "string",
                "description": "Package name to install (for install_package)",
            },
        },
        "required": ["action"],
    },
}


async def execute_sandbox(agent: Agent, input: dict[str, Any]) -> str:
    action = input["action"]
    sandbox = agent.civilization.sandbox_manager

    if action == "exec_python":
        code = input.get("code", "")
        if not code:
            return "Error: 'code' is required for exec_python."
        result = sandbox.exec_python(agent.id, code)
        parts = []
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        if result.error:
            parts.append(f"error: {result.error}")
        if result.timed_out:
            parts.append("(execution timed out)")
        if result.exit_code != 0:
            parts.append(f"exit code: {result.exit_code}")
        return "\n".join(parts) if parts else "(no output)"

    elif action == "exec_shell":
        command = input.get("command", "")
        if not command:
            return "Error: 'command' is required for exec_shell."
        result = sandbox.exec_shell(agent.id, command)
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"stderr: {result.stderr}")
        if result.exit_code != 0:
            parts.append(f"exit code: {result.exit_code}")
        return "\n".join(parts) if parts else "(no output)"

    elif action == "read_file":
        path = input.get("path", "")
        if not path:
            return "Error: 'path' is required for read_file."
        try:
            content = sandbox.read_file(agent.id, path)
            return content
        except (FileNotFoundError, PermissionError) as e:
            return f"Error: {e}"

    elif action == "write_file":
        path = input.get("path", "")
        content = input.get("content", "")
        if not path:
            return "Error: 'path' is required for write_file."
        try:
            sandbox.write_file(agent.id, path, content)
            return f"File written: {path}"
        except PermissionError as e:
            return f"Error: {e}"

    elif action == "list_files":
        path = input.get("path", "/workspace")
        files = sandbox.list_files(agent.id, path)
        if not files:
            return "No files in workspace."
        return "Files:\n" + "\n".join(f"  - {f}" for f in files)

    elif action == "install_package":
        package = input.get("package", "")
        if not package:
            return "Error: 'package' is required for install_package."
        result = sandbox.install_package(agent.id, package)
        if result.exit_code == 0:
            return f"Package '{package}' installed successfully."
        return f"Error installing '{package}':\n{result.stderr}"

    return f"Unknown action: {action}"


# ──────────────────────────────────────────────
# 8. evolve
# ──────────────────────────────────────────────
EVOLVE_DEFINITION = {
    "name": "evolve",
    "description": (
        "Evolve your role and store important knowledge. "
        "Use this as your specialization crystallizes through experience. "
        "You can update your role description, store key insights to long-term memory, "
        "and add episodic memories of significant events."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "new_role": {
                "type": "string",
                "description": "Updated role description (optional)",
            },
            "role_note": {
                "type": "string",
                "description": "A note about why your role is evolving (optional)",
            },
            "knowledge_key": {
                "type": "string",
                "description": "Key for a fact to remember (optional)",
            },
            "knowledge_value": {
                "type": "string",
                "description": "Value for the fact to remember (optional)",
            },
            "memory_summary": {
                "type": "string",
                "description": "Summary of a significant event to remember (optional)",
            },
            "memory_importance": {
                "type": "number",
                "description": "Importance of the memory (0-1 scale, default 0.5)",
                "default": 0.5,
            },
        },
    },
}


async def execute_evolve(agent: Agent, input: dict[str, Any]) -> str:
    results = []

    if input.get("new_role"):
        old_role = agent.state.role
        agent.state.role = input["new_role"]
        results.append(f"Role updated: '{old_role}' → '{input['new_role']}'")

    if input.get("role_note"):
        agent.state.memory.add_role_note(input["role_note"])
        results.append(f"Role note added: {input['role_note']}")

    if input.get("knowledge_key") and input.get("knowledge_value"):
        agent.state.memory.add_knowledge(input["knowledge_key"], input["knowledge_value"])
        results.append(f"Knowledge stored: {input['knowledge_key']}")

    if input.get("memory_summary"):
        importance = input.get("memory_importance", 0.5)
        agent.state.memory.add_episodic_memory(
            summary=input["memory_summary"],
            importance=importance,
        )
        results.append(f"Memory recorded (importance: {importance})")

    if not results:
        return "No changes specified. Provide at least one field to update."

    agent.civilization.events.emit("agent_evolved", {
        "agent_id": agent.id,
        "changes": results,
    })

    return "Evolution complete:\n" + "\n".join(f"  - {r}" for r in results)


# ──────────────────────────────────────────────
# 9. query_civilization
# ──────────────────────────────────────────────
QUERY_CIVILIZATION_DEFINITION = {
    "name": "query_civilization",
    "description": (
        "Query the civilization's state. "
        "Types: 'agents' (list all agents), 'tools' (list all tools), "
        "'alliances' (list all alliances), 'history' (recent events), "
        "'knowledge_search' (search across all agents' knowledge)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["agents", "tools", "alliances", "history", "knowledge_search"],
                "description": "What to query",
            },
            "search_term": {
                "type": "string",
                "description": "Search term (for knowledge_search)",
            },
            "agent_id": {
                "type": "string",
                "description": "Filter by specific agent (optional)",
            },
        },
        "required": ["query_type"],
    },
}


async def execute_query_civilization(agent: Agent, input: dict[str, Any]) -> str:
    query_type = input["query_type"]
    civ = agent.civilization

    if query_type == "agents":
        agents = civ.get_all_agents()
        if not agents:
            return "No agents in the civilization."
        lines = []
        for a in agents:
            marker = " (you)" if a.id == agent.id else ""
            lines.append(
                f"- **{a.state.name}**{marker}\n"
                f"  id: {a.id}\n"
                f"  role: {a.state.role}\n"
                f"  status: {a.state.status.value}\n"
                f"  tools created: {a.state.total_tools_created}\n"
                f"  agents created: {a.state.total_agents_created}"
            )
        return "## Agents\n\n" + "\n\n".join(lines)

    elif query_type == "tools":
        registry = civ.tool_registry
        sections = []

        builtins = registry.get_builtin_tools()
        sections.append(
            "### Built-in Tools\n"
            + "\n".join(f"- **{t.name}**: {t.description[:80]}..." for t in builtins)
        )

        custom = [t for t in registry.get_all_tools() if t.scope != ToolScope.BUILTIN]
        if custom:
            from aivilization.core.tool import ToolScope

            for t in custom:
                scope_label = "shared" if t.scope == ToolScope.SHARED else "private"
                sections.append(
                    f"- **{t.name}** (id: {t.id}, {scope_label}, v{t.version})\n"
                    f"  {t.description[:100]}\n"
                    f"  uses: {t.usage_count}"
                )

        return "## Tools\n\n" + "\n\n".join(sections)

    elif query_type == "alliances":
        alliances = civ.get_all_alliances()
        if not alliances:
            return "No alliances formed yet."
        lines = []
        for a in alliances:
            member_names = []
            for aid in a.agent_ids:
                ag = civ.get_agent(aid)
                member_names.append(ag.state.name if ag else aid)
            lines.append(
                f"- **{a.name}** (id: {a.id})\n"
                f"  purpose: {a.purpose}\n"
                f"  members: {', '.join(member_names)}"
            )
        return "## Alliances\n\n" + "\n\n".join(lines)

    elif query_type == "history":
        events = civ.events.get_recent_events(20)
        if not events:
            return "No events recorded yet."
        lines = []
        for e in events:
            ts = e.timestamp.strftime("%H:%M:%S")
            lines.append(f"- [{ts}] {e.event_type}: {e.data}")
        return "## Recent History\n\n" + "\n".join(lines)

    elif query_type == "knowledge_search":
        search_term = input.get("search_term", "").lower()
        if not search_term:
            return "Error: 'search_term' is required for knowledge_search."

        results = []
        target_agents = [civ.get_agent(input["agent_id"])] if input.get("agent_id") else civ.get_all_agents()

        for a in target_agents:
            if a is None:
                continue
            for k, v in a.state.memory.knowledge.items():
                if search_term in k.lower() or search_term in v.lower():
                    results.append(f"- [{a.state.name}] {k}: {v}")

            for mem in a.state.memory.episodic_memories:
                if search_term in mem.summary.lower():
                    results.append(f"- [{a.state.name}] Memory: {mem.summary}")

        if not results:
            return f"No knowledge found matching '{search_term}'."
        return f"## Knowledge Search: '{search_term}'\n\n" + "\n".join(results)

    return f"Unknown query type: {query_type}"


# ──────────────────────────────────────────────
# 10. form_alliance
# ──────────────────────────────────────────────
FORM_ALLIANCE_DEFINITION = {
    "name": "form_alliance",
    "description": (
        "Create a persistent collaboration group between agents. "
        "Alliance members share context, can broadcast within the group, "
        "and have a shared knowledge store. "
        "The creating agent is automatically included."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Alliance name",
            },
            "agent_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agent IDs to include in the alliance",
            },
            "purpose": {
                "type": "string",
                "description": "Shared purpose/goal of the alliance",
            },
            "shared_context": {
                "type": "string",
                "description": "Initial shared context for all members (optional)",
            },
        },
        "required": ["name", "agent_ids", "purpose"],
    },
}


async def execute_form_alliance(agent: Agent, input: dict[str, Any]) -> str:
    civ = agent.civilization

    agent_ids = list(set(input["agent_ids"] + [agent.id]))

    # Validate all agent IDs exist
    for aid in agent_ids:
        if civ.get_agent(aid) is None:
            return f"Error: no agent found with id '{aid}'."

    alliance = civ.create_alliance(
        name=input["name"],
        agent_ids=agent_ids,
        purpose=input["purpose"],
        shared_context=input.get("shared_context", ""),
        created_by=agent.id,
    )

    member_names = [civ.get_agent(aid).state.name for aid in agent_ids]

    return (
        f"Alliance '{alliance.name}' formed (id: {alliance.id}).\n"
        f"Members: {', '.join(member_names)}\n"
        f"Purpose: {alliance.purpose}\n"
        f"Use broadcast with alliance_name='{alliance.name}' to message the group."
    )


# ──────────────────────────────────────────────
# Master registry
# ──────────────────────────────────────────────
BUILTIN_TOOLS: dict[str, tuple[dict[str, Any], Any]] = {
    "create_agent": (CREATE_AGENT_DEFINITION, execute_create_agent),
    "delegate_task": (DELEGATE_TASK_DEFINITION, execute_delegate_task),
    "broadcast": (BROADCAST_DEFINITION, execute_broadcast),
    "create_tool": (CREATE_TOOL_DEFINITION, execute_create_tool),
    "edit_tool": (EDIT_TOOL_DEFINITION, execute_edit_tool),
    "delete_tool": (DELETE_TOOL_DEFINITION, execute_delete_tool),
    "sandbox": (SANDBOX_DEFINITION, execute_sandbox),
    "evolve": (EVOLVE_DEFINITION, execute_evolve),
    "query_civilization": (QUERY_CIVILIZATION_DEFINITION, execute_query_civilization),
    "form_alliance": (FORM_ALLIANCE_DEFINITION, execute_form_alliance),
}
