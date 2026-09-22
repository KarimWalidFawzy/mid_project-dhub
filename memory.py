"""Persistent conversation memory for the knowledge assistant."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class Message:
    role: str
    content: str
    timestamp: str


class ConversationMemory:
    """A small JSON-backed buffer memory with a stable, inspectable format."""

    def __init__(self, path: str | Path = "data/memory.json") -> None:
        self.path = Path(path)
        self.messages: list[Message] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw_messages = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.messages = []
            return
        self.messages = [Message(**item) for item in raw_messages if self._valid(item)]

    @staticmethod
    def _valid(item: object) -> bool:
        return (
            isinstance(item, dict)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("content"), str)
            and isinstance(item.get("timestamp"), str)
        )

    def add(self, role: str, content: str) -> None:
        self.messages.append(
            Message(
                role=role,
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(message) for message in self.messages], indent=2),
            encoding="utf-8",
        )

    def recent(self, limit: int = 8) -> list[Message]:
        return self.messages[-limit:]

    def context(self, limit: int = 8) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in self.recent(limit))

    def clear(self) -> None:
        self.messages = []
        self.save()

    def __iter__(self) -> Iterable[Message]:
        return iter(self.messages)
