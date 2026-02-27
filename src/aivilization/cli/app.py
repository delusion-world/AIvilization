"""
Rich-based CLI for AIvilization.

Commands:
  (default)           Chat with the active agent
  /agents             List all agents
  /tools              List all tools
  /alliances          List all alliances
  /agent <id|name>    Switch to chatting with a specific agent
  /status             Show civilization status
  /save               Save civilization state
  /load <id>          Load a saved civilization
  /new                Start a new civilization
  /graph              Show agent creation graph
  /cost               Show API usage and estimated cost
  /history            Show recent events
  /quit               Exit
"""
from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from aivilization.config import AIvilizationConfig
from aivilization.core.civilization import Civilization


class AIvilizationCLI:
    """Interactive CLI for the AIvilization system."""

    def __init__(self, config: AIvilizationConfig | None = None) -> None:
        self.console = Console()
        self.config = config or AIvilizationConfig()
        self.civilization: Civilization | None = None
        self.active_agent_id: str | None = None

    async def run(self) -> None:
        """Main CLI loop."""
        self._show_banner()
        self.civilization = self._initialize_civilization()
        self.active_agent_id = self.civilization.state.primary_agent_id

        primary = self.civilization.get_primary_agent()
        if primary:
            self.console.print(
                f"[dim]Primary agent: [bold]{primary.state.name}[/bold] — {primary.state.role}[/dim]\n"
            )

        while True:
            try:
                agent = self.civilization.get_agent(self.active_agent_id)
                agent_name = agent.state.name if agent else "???"
                user_input = self.console.input(
                    f"[bold cyan]{agent_name}>[/] "
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    should_quit = await self._handle_command(user_input)
                    if should_quit:
                        break
                else:
                    await self._chat(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Interrupted. Use /quit to exit.[/dim]")
            except EOFError:
                break

        self.console.print("\n[dim]Saving civilization...[/dim]")
        self.civilization.save()
        self.civilization.shutdown()
        self.console.print("[dim]Goodbye.[/dim]")

    async def _chat(self, message: str) -> None:
        """Send a message to the active agent and display the response."""
        agent = self.civilization.get_agent(self.active_agent_id)
        if not agent:
            self.console.print("[red]No active agent.[/red]")
            return

        with self.console.status(
            f"[bold green]{agent.state.name} is thinking...[/]",
            spinner="dots",
        ):
            try:
                response = await agent.process_message(message)
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                return

        self.console.print(
            Panel(
                Markdown(response),
                title=f"[bold green]{agent.state.name}[/]",
                border_style="green",
                padding=(1, 2),
            )
        )

    async def _handle_command(self, cmd: str) -> bool:
        """Route slash commands. Returns True if should quit."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command == "/quit" or command == "/exit":
            return True
        elif command == "/agents":
            self._cmd_agents()
        elif command == "/tools":
            self._cmd_tools()
        elif command == "/alliances":
            self._cmd_alliances()
        elif command == "/agent":
            self._cmd_switch_agent(args)
        elif command == "/status":
            self._cmd_status()
        elif command == "/save":
            self._cmd_save()
        elif command == "/load":
            self._cmd_load(args)
        elif command == "/new":
            self._cmd_new()
        elif command == "/graph":
            self._cmd_graph()
        elif command == "/cost":
            self._cmd_cost()
        elif command == "/history":
            self._cmd_history()
        elif command == "/help":
            self._cmd_help()
        else:
            self.console.print(f"[red]Unknown command: {command}. Type /help for available commands.[/red]")

        return False

    def _cmd_agents(self) -> None:
        """List all agents."""
        agents = self.civilization.get_all_agents()
        if not agents:
            self.console.print("[dim]No agents yet.[/dim]")
            return

        table = Table(title="Agents in Civilization")
        table.add_column("Name", style="bold")
        table.add_column("ID", style="dim")
        table.add_column("Role")
        table.add_column("Status")
        table.add_column("Tools", justify="right")
        table.add_column("Agents", justify="right")
        table.add_column("API Calls", justify="right")

        for a in agents:
            marker = " *" if a.id == self.active_agent_id else ""
            table.add_row(
                a.state.name + marker,
                a.id[:8] + "...",
                a.state.role[:40],
                a.state.status.value,
                str(a.state.total_tools_created),
                str(a.state.total_agents_created),
                str(a.state.total_api_calls),
            )

        self.console.print(table)

    def _cmd_tools(self) -> None:
        """List all tools."""
        registry = self.civilization.tool_registry

        table = Table(title="Tools")
        table.add_column("Name", style="bold")
        table.add_column("Scope")
        table.add_column("Version", justify="right")
        table.add_column("Uses", justify="right")
        table.add_column("Description")

        for t in registry.get_all_tools():
            table.add_row(
                t.name,
                t.scope.value,
                str(t.version),
                str(t.usage_count),
                t.description[:50] + "..." if len(t.description) > 50 else t.description,
            )

        self.console.print(table)

    def _cmd_alliances(self) -> None:
        """List all alliances."""
        alliances = self.civilization.get_all_alliances()
        if not alliances:
            self.console.print("[dim]No alliances formed yet.[/dim]")
            return

        for a in alliances:
            members = []
            for aid in a.agent_ids:
                ag = self.civilization.get_agent(aid)
                members.append(ag.state.name if ag else aid[:8])
            self.console.print(
                Panel(
                    f"Purpose: {a.purpose}\nMembers: {', '.join(members)}",
                    title=f"[bold]{a.name}[/]",
                    border_style="blue",
                )
            )

    def _cmd_switch_agent(self, args: str) -> None:
        """Switch active agent by ID or name."""
        if not args:
            self.console.print("[red]Usage: /agent <id or name>[/red]")
            return

        # Try by ID first
        agent = self.civilization.get_agent(args)
        if not agent:
            # Try by name
            for a in self.civilization.get_all_agents():
                if a.state.name.lower() == args.lower():
                    agent = a
                    break

        if agent:
            self.active_agent_id = agent.id
            self.console.print(
                f"[green]Switched to agent: {agent.state.name} ({agent.state.role})[/green]"
            )
        else:
            self.console.print(f"[red]Agent not found: {args}[/red]")

    def _cmd_status(self) -> None:
        """Show civilization status."""
        civ = self.civilization
        usage = civ.claude_client.get_usage_summary()

        self.console.print(
            Panel(
                f"Name: {civ.state.name}\n"
                f"ID: {civ.state.id}\n"
                f"Agents: {len(civ.agents)}\n"
                f"Custom Tools: {len([t for t in civ.tool_registry.get_all_tools() if t.scope.value != 'builtin'])}\n"
                f"Alliances: {len(civ.get_all_alliances())}\n"
                f"API Calls: {usage['total_api_calls']}\n"
                f"Tokens: {usage['total_input_tokens'] + usage['total_output_tokens']:,}\n"
                f"Est. Cost: ${usage['estimated_cost_usd']:.4f}",
                title="[bold]Civilization Status[/]",
                border_style="cyan",
            )
        )

    def _cmd_save(self) -> None:
        """Save civilization state."""
        path = self.civilization.save()
        self.console.print(f"[green]Civilization saved to {path}[/green]")

    def _cmd_load(self, args: str) -> None:
        """Load a saved civilization."""
        if not args:
            # List available civilizations
            saved = self.civilization.list_saved()
            if not saved:
                self.console.print("[dim]No saved civilizations found.[/dim]")
                return
            self.console.print("[bold]Saved civilizations:[/bold]")
            for s in saved:
                self.console.print(
                    f"  - {s['name']} (id: {s['id']}, agents: {s.get('agent_count', '?')})"
                )
            self.console.print("[dim]Use /load <id> to load one.[/dim]")
            return

        try:
            self.civilization.shutdown()
            self.civilization = Civilization.load(self.config, args)
            self.active_agent_id = self.civilization.state.primary_agent_id
            self.console.print(
                f"[green]Loaded civilization: {self.civilization.state.name}[/green]"
            )
        except FileNotFoundError:
            self.console.print(f"[red]Civilization not found: {args}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error loading: {e}[/red]")

    def _cmd_new(self) -> None:
        """Start a new civilization."""
        if self.civilization:
            self.civilization.save()
            self.civilization.shutdown()
        self.civilization = self._initialize_civilization()
        self.active_agent_id = self.civilization.state.primary_agent_id
        self.console.print("[green]New civilization created![/green]")

    def _cmd_graph(self) -> None:
        """Display the agent creation tree."""
        tree = Tree("[bold]Civilization Agent Graph[/bold]")
        primary = self.civilization.get_primary_agent()
        if primary:
            self._build_tree(tree, primary.id)
        else:
            tree.add("[dim]No agents[/dim]")
        self.console.print(tree)

    def _build_tree(self, parent_node: Any, agent_id: str) -> None:
        """Recursively build the agent tree."""
        agent = self.civilization.get_agent(agent_id)
        if not agent:
            return

        marker = " *" if agent_id == self.active_agent_id else ""
        label = (
            f"[bold]{agent.state.name}[/]{marker} "
            f"[dim]({agent.state.role})[/dim]"
        )
        node = parent_node.add(label)

        children = self.civilization.state.creation_graph.get(agent_id, [])
        for child_id in children:
            self._build_tree(node, child_id)

    def _cmd_cost(self) -> None:
        """Show API usage and estimated cost."""
        usage = self.civilization.claude_client.get_usage_summary()
        self.console.print(
            Panel(
                f"API Calls: {usage['total_api_calls']}\n"
                f"Input Tokens: {usage['total_input_tokens']:,}\n"
                f"Output Tokens: {usage['total_output_tokens']:,}\n"
                f"Estimated Cost: ${usage['estimated_cost_usd']:.4f}\n"
                f"Budget: ${self.config.max_cost_per_session_usd:.2f}",
                title="[bold]API Usage[/]",
                border_style="yellow",
            )
        )

    def _cmd_history(self) -> None:
        """Show recent events."""
        events = self.civilization.events.get_recent_events(20)
        if not events:
            self.console.print("[dim]No events recorded yet.[/dim]")
            return

        for e in events:
            ts = e.timestamp.strftime("%H:%M:%S")
            self.console.print(f"  [{ts}] [bold]{e.event_type}[/] {e.data}")

    def _cmd_help(self) -> None:
        """Show available commands."""
        self.console.print(
            Panel(
                "  (text)              Chat with the active agent\n"
                "  /agents             List all agents\n"
                "  /tools              List all tools\n"
                "  /alliances          List all alliances\n"
                "  /agent <id|name>    Switch to a specific agent\n"
                "  /status             Show civilization status\n"
                "  /save               Save civilization state\n"
                "  /load [id]          Load a saved civilization\n"
                "  /new                Start a new civilization\n"
                "  /graph              Show agent creation tree\n"
                "  /cost               Show API usage and cost\n"
                "  /history            Show recent events\n"
                "  /quit               Exit",
                title="[bold]Commands[/]",
                border_style="cyan",
            )
        )

    def _show_banner(self) -> None:
        """Display the startup banner."""
        self.console.print(
            Panel(
                "[bold]AIvilization[/bold]\n"
                "[dim]A multi-agent AI civilization where agents autonomously\n"
                "collaborate, create tools, and evolve[/dim]",
                border_style="bright_blue",
                padding=(1, 4),
            )
        )

    def _initialize_civilization(self) -> Civilization:
        """Create a new civilization with the primary agent."""
        civ = Civilization(self.config, name="My Civilization")
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
        return civ
