"""
Read-only FastAPI web dashboard for viewing civilization state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aivilization.core.civilization import Civilization


def create_app(civilization: Civilization) -> FastAPI:
    """Create the FastAPI app with routes bound to a civilization instance."""
    app = FastAPI(title="AIvilization Dashboard")

    templates_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    templates = Jinja2Templates(directory=str(templates_dir))

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        agents = civilization.get_all_agents()
        tools = civilization.tool_registry.get_all_tools()
        events = civilization.events.get_recent_events(50)
        usage = civilization.claude_client.get_usage_summary()
        alliances = civilization.get_all_alliances()

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "civilization": civilization,
                "agents": agents,
                "tools": tools,
                "events": events,
                "usage": usage,
                "alliances": alliances,
                "creation_graph": civilization.state.creation_graph,
            },
        )

    @app.get("/agent/{agent_id}", response_class=HTMLResponse)
    async def agent_detail(request: Request, agent_id: str) -> HTMLResponse:
        agent = civilization.get_agent(agent_id)
        if not agent:
            return HTMLResponse("Agent not found", status_code=404)

        return templates.TemplateResponse(
            "agent_detail.html",
            {
                "request": request,
                "agent": agent,
                "memory": agent.state.memory,
                "alliances": civilization.get_agent_alliances(agent_id),
            },
        )

    @app.get("/api/events")
    async def api_events(limit: int = 50) -> list[dict[str, Any]]:
        events = civilization.events.get_recent_events(limit)
        return [e.model_dump(mode="json") for e in events]

    @app.get("/api/agents")
    async def api_agents() -> list[dict[str, Any]]:
        return [
            {
                "id": a.id,
                "name": a.state.name,
                "role": a.state.role,
                "status": a.state.status.value,
                "created_by": a.state.created_by,
                "tools_created": a.state.total_tools_created,
                "agents_created": a.state.total_agents_created,
                "api_calls": a.state.total_api_calls,
            }
            for a in civilization.get_all_agents()
        ]

    @app.get("/api/tools")
    async def api_tools() -> list[dict[str, Any]]:
        return [
            {
                "id": t.id,
                "name": t.name,
                "scope": t.scope.value,
                "version": t.version,
                "usage_count": t.usage_count,
                "description": t.description,
            }
            for t in civilization.tool_registry.get_all_tools()
        ]

    return app
