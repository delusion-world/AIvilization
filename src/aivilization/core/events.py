from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field


class CivilizationEvent(BaseModel):
    """A single event in the civilization's history."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    data: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Simple synchronous event bus for civilization-wide events."""

    def __init__(self, max_history: int = 1000) -> None:
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[CivilizationEvent] = []
        self._max_history = max_history

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        self._handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if handler in self._handlers.get(event_type, []):
            self._handlers[event_type].remove(handler)

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event, calling all registered handlers."""
        event = CivilizationEvent(event_type=event_type, data=data)
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        for handler in self._handlers.get(event_type, []):
            handler(event)

    def get_recent_events(self, n: int = 50) -> list[CivilizationEvent]:
        """Get the most recent N events."""
        return self._history[-n:]

    def get_events_by_type(self, event_type: str, n: int = 50) -> list[CivilizationEvent]:
        """Get the most recent N events of a specific type."""
        matching = [e for e in self._history if e.event_type == event_type]
        return matching[-n:]

    @property
    def history(self) -> list[CivilizationEvent]:
        return self._history
