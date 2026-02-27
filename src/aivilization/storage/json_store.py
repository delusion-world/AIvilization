from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class JsonStore:
    """File-based JSON persistence for civilization state."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_civilization(self, state_data: dict[str, Any]) -> Path:
        """Save civilization state to a JSON file."""
        civ_id = state_data["id"]
        filepath = self.data_dir / f"{civ_id}.json"
        filepath.write_text(json.dumps(state_data, cls=DateTimeEncoder, indent=2))
        self._update_index(state_data)
        return filepath

    def load_civilization(self, civilization_id: str) -> dict[str, Any]:
        """Load civilization state from JSON."""
        filepath = self.data_dir / f"{civilization_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(
                f"No civilization found with id '{civilization_id}'"
            )
        return json.loads(filepath.read_text())

    def list_civilizations(self) -> list[dict[str, Any]]:
        """List all saved civilizations from the index."""
        index_path = self.data_dir / "index.json"
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text())

    def delete_civilization(self, civilization_id: str) -> None:
        """Delete a saved civilization."""
        filepath = self.data_dir / f"{civilization_id}.json"
        if filepath.exists():
            filepath.unlink()

        # Update index
        index_path = self.data_dir / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            index = [e for e in index if e["id"] != civilization_id]
            index_path.write_text(json.dumps(index, indent=2))

    def _update_index(self, state_data: dict[str, Any]) -> None:
        """Update the civilization index file."""
        index_path = self.data_dir / "index.json"
        index: list[dict[str, Any]] = []
        if index_path.exists():
            index = json.loads(index_path.read_text())

        civ_id = state_data["id"]
        entry = {
            "id": civ_id,
            "name": state_data.get("name", "Unnamed"),
            "created_at": state_data.get("created_at", ""),
            "agent_count": len(state_data.get("agent_states", {})),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Upsert
        existing_idx = next(
            (i for i, e in enumerate(index) if e["id"] == civ_id), None
        )
        if existing_idx is not None:
            index[existing_idx] = entry
        else:
            index.append(entry)

        index_path.write_text(json.dumps(index, indent=2))
